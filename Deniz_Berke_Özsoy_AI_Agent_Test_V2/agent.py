"""Hierarchical multi-agent data analysis assistant.

Architecture:
1. GuardrailValidator blocks unsafe prompts before planning.
2. PlannerAgent writes an English step-by-step plan.
3. ExecutorAgent is the only component allowed to call tools.
4. EditorCriticAgent verifies observations and writes the final Turkish answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools import create_vehicle_consumption_chart, fetch_live_weather_api, query_holidays, query_vehicles, query_weather
from utils import AgentRunResult, AgentTraceStep, ConversationMemory, DynamicSemanticMemory, GuardrailValidator, ReflectionReport, ToolCall, compact_json, dataclass_to_dict, normalize_text, safe_json_loads, setup_logger, truncate_text


@dataclass(frozen=True)
class RouteDecision:
    """Semantic routing decision.

    Attributes:
        intent: Selected intent.
        dataset: Logical data source.
        confidence: Confidence score.
        reason: English explanation.
        guardrail_triggered: Whether a security guardrail blocked the request.
    """

    intent: str
    dataset: str
    confidence: float
    reason: str
    guardrail_triggered: bool = False


@dataclass
class ExecutionPlan:
    """Planner output.

    Attributes:
        user_query: Original user query.
        route: Route decision.
        steps: Ordered English execution steps.
        required_tools: Tools expected by the plan.
        memory_updates: User profile updates extracted before execution.
        contextual_preferences: Preferences that should guide execution.
    """

    user_query: str
    route: RouteDecision
    steps: List[str]
    required_tools: List[str] = field(default_factory=list)
    memory_updates: Dict[str, Any] = field(default_factory=dict)
    contextual_preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutorOutput:
    """Executor output.

    Attributes:
        trace: ReAct-style trace steps.
        raw_observations: Parsed tool outputs.
        completed: Whether execution completed successfully.
    """

    trace: List[AgentTraceStep]
    raw_observations: List[Dict[str, Any]]
    completed: bool


class PlannerAgent:
    """Planner agent that creates explicit English execution plans."""

    def create_plan(self, user_query: str, route: RouteDecision, memory_updates: Dict[str, Any], contextual_preferences: Dict[str, Any]) -> ExecutionPlan:
        """Create a step-by-step plan.

        Args:
            user_query: User query.
            route: Semantic route.
            memory_updates: Newly extracted user preference updates.
            contextual_preferences: Previously stored preferences.

        Returns:
            Execution plan.
        """
        if route.intent == "vehicle_lookup":
            steps = ["Identify whether the user specified a vehicle type.", "If no explicit vehicle type is provided, use the stored vehicle preference if available.", "Call the vehicle query tool and sort by fuel consumption in ascending order.", "Return only data-supported facts to the Editor/Critic agent."]
            required_tools = ["query_vehicles"]
        elif route.intent == "vehicle_chart":
            steps = ["Identify the requested vehicle subset if any.", "If the user did not specify a type, optionally use the stored vehicle preference.", "Call the publication-quality charting tool.", "Return chart paths and plotted records to the Editor/Critic agent."]
            required_tools = ["create_vehicle_consumption_chart"]
        elif route.intent == "holiday_lookup":
            steps = ["Extract the requested Turkish date expression.", "Call the official-holiday Excel tool with the extracted date.", "Return the matched holiday record to the Editor/Critic agent."]
            required_tools = ["query_holidays"]
        elif route.intent == "holiday_list":
            steps = ["Call the official-holiday Excel tool in list-all mode.", "Return all available records to the Editor/Critic agent."]
            required_tools = ["query_holidays"]
        elif route.intent == "weather_forecast":
            steps = ["Recognize that the Excel weather file contains historical averages only.", "Use the external weather fallback tool for future forecast queries.", "Return forecast records and source limitations to the Editor/Critic agent."]
            required_tools = ["fetch_live_weather_api"]
        elif route.intent == "weather_historical":
            steps = ["Infer the requested city, month, and weather metric.", "Call the historical weather Excel tool.", "Return the historical average and clearly mark it as non-forecast data."]
            required_tools = ["query_weather"]
        elif route.intent == "preference_update":
            steps = ["Do not call external tools.", "Confirm that the durable user preference was saved.", "Explain how it will be used in future vehicle-related questions."]
            required_tools = []
        else:
            steps = ["Do not call tools because the query cannot be confidently mapped.", "Ask the user to rephrase around supported domains."]
            required_tools = []

        return ExecutionPlan(user_query=user_query, route=route, steps=steps, required_tools=required_tools, memory_updates=memory_updates, contextual_preferences=contextual_preferences)


class ExecutorAgent:
    """Executor agent that is exclusively allowed to call tools."""

    def __init__(self, data_dir: str, output_dir: str) -> None:
        """Initialize executor.

        Args:
            data_dir: Excel data directory.
            output_dir: Chart output directory.
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.logger = setup_logger()
        self.tool_registry: Dict[str, Callable[..., str]] = {"query_vehicles": query_vehicles, "query_holidays": query_holidays, "query_weather": query_weather, "fetch_live_weather_api": fetch_live_weather_api, "create_vehicle_consumption_chart": create_vehicle_consumption_chart}

    def execute(self, plan: ExecutionPlan, normalized_query: str) -> ExecutorOutput:
        """Execute a plan using tools.

        Args:
            plan: Planner output.
            normalized_query: Normalized user query.

        Returns:
            Executor output.
        """
        trace: List[AgentTraceStep] = []
        observations: List[Dict[str, Any]] = []
        route = plan.route
        step_index = 1
        planner_observation = {"status": "success", "agent": "PlannerAgent", "plan_steps": plan.steps, "required_tools": plan.required_tools, "memory_updates": plan.memory_updates, "contextual_preferences": plan.contextual_preferences}
        trace.append(AgentTraceStep(step_index=step_index, agent_name="PlannerAgent", thought="The PlannerAgent decomposed the user request into an explicit execution plan before any tool call.", action=None, observation=planner_observation))
        step_index += 1

        if route.intent == "vehicle_lookup":
            explicit_vehicle_type = self._extract_vehicle_type(normalized_query)
            memory_vehicle_type = plan.contextual_preferences.get("preferred_vehicle_type")
            vehicle_type = explicit_vehicle_type or memory_vehicle_type

            tool_call = ToolCall(
                name="query_vehicles",
                arguments={
                    "vehicle_type": vehicle_type,
                    "sort_by": "consumption",
                    "ascending": True,
                    "top_n": 1,
                    "data_dir": self.data_dir,
                },
            )
            observation = self._execute_tool_call(tool_call)
            observations.append(observation)
            trace.append(
                AgentTraceStep(
                    step_index=step_index,
                    agent_name="ExecutorAgent",
                    thought="The ExecutorAgent is calling the vehicle tool because vehicle fuel consumption requires Excel-backed analysis.",
                    action=tool_call,
                    observation=observation,
                )
            )

            # Academic robustness enhancement:
            # If a stored preference such as SUV does not exist in the dataset, the Executor retries
            # without the preference instead of failing the whole conversation.
            if (
                observation.get("status") == "error"
                and observation.get("error", {}).get("code") == "NO_MATCHING_VEHICLES"
                and explicit_vehicle_type is None
                and memory_vehicle_type is not None
            ):
                step_index += 1
                fallback_call = ToolCall(
                    name="query_vehicles",
                    arguments={
                        "vehicle_type": None,
                        "sort_by": "consumption",
                        "ascending": True,
                        "top_n": 1,
                        "data_dir": self.data_dir,
                    },
                )
                fallback_observation = self._execute_tool_call(fallback_call)
                fallback_observation.setdefault("data", {})["preference_fallback"] = {
                    "requested_preference": memory_vehicle_type,
                    "reason": "The preferred vehicle type was not present in the Excel dataset, so the query was retried without that filter.",
                }
                observations.append(fallback_observation)
                trace.append(
                    AgentTraceStep(
                        step_index=step_index,
                        agent_name="ExecutorAgent",
                        thought="The stored vehicle preference did not match any row, so the Executor retried the query without the preference filter.",
                        action=fallback_call,
                        observation=fallback_observation,
                    )
                )
        elif route.intent == "vehicle_chart":
            explicit_vehicle_type = self._extract_vehicle_type(normalized_query)
            memory_vehicle_type = plan.contextual_preferences.get("preferred_vehicle_type")
            vehicle_type = explicit_vehicle_type or memory_vehicle_type

            tool_call = ToolCall(
                name="create_vehicle_consumption_chart",
                arguments={
                    "vehicle_type": vehicle_type,
                    "output_dir": self.output_dir,
                    "data_dir": self.data_dir,
                },
            )
            observation = self._execute_tool_call(tool_call)
            observations.append(observation)
            trace.append(
                AgentTraceStep(
                    step_index=step_index,
                    agent_name="ExecutorAgent",
                    thought="The ExecutorAgent is generating a research-grade chart because the user requested visual comparison.",
                    action=tool_call,
                    observation=observation,
                )
            )

            # Robust visualization fallback:
            # A stored preference can be too narrow for small datasets. If it causes an empty chart,
            # retry with all vehicles so the UI still produces a useful visual artifact.
            if (
                observation.get("status") == "error"
                and observation.get("error", {}).get("code") == "NO_CHART_DATA"
                and explicit_vehicle_type is None
                and memory_vehicle_type is not None
            ):
                step_index += 1
                fallback_call = ToolCall(
                    name="create_vehicle_consumption_chart",
                    arguments={
                        "vehicle_type": None,
                        "output_dir": self.output_dir,
                        "data_dir": self.data_dir,
                    },
                )
                fallback_observation = self._execute_tool_call(fallback_call)
                fallback_observation.setdefault("data", {})["preference_fallback"] = {
                    "requested_preference": memory_vehicle_type,
                    "reason": "The preferred vehicle type was not present in the Excel dataset, so the chart was generated for all vehicles.",
                }
                observations.append(fallback_observation)
                trace.append(
                    AgentTraceStep(
                        step_index=step_index,
                        agent_name="ExecutorAgent",
                        thought="The stored vehicle preference produced no chart data, so the Executor generated the comparison chart for all vehicles.",
                        action=fallback_call,
                        observation=fallback_observation,
                    )
                )
        elif route.intent == "holiday_lookup":
            date_text = self._extract_holiday_date(plan.user_query) or plan.user_query
            tool_call = ToolCall(name="query_holidays", arguments={"date_text": date_text, "holiday_name": None, "list_all": False, "data_dir": self.data_dir})
            observation = self._execute_tool_call(tool_call)
            observations.append(observation)
            trace.append(AgentTraceStep(step_index=step_index, agent_name="ExecutorAgent", thought="The ExecutorAgent is querying the holiday dataset by date.", action=tool_call, observation=observation))
        elif route.intent == "holiday_list":
            tool_call = ToolCall(name="query_holidays", arguments={"date_text": None, "holiday_name": None, "list_all": True, "data_dir": self.data_dir})
            observation = self._execute_tool_call(tool_call)
            observations.append(observation)
            trace.append(AgentTraceStep(step_index=step_index, agent_name="ExecutorAgent", thought="The ExecutorAgent is listing all holiday records from the Excel dataset.", action=tool_call, observation=observation))
        elif route.intent == "weather_forecast":
            city = self._extract_city(normalized_query)
            days = self._extract_forecast_days(normalized_query)
            tool_call = ToolCall(name="fetch_live_weather_api", arguments={"city": city, "days": days})
            observation = self._execute_tool_call(tool_call)
            observations.append(observation)
            trace.append(AgentTraceStep(step_index=step_index, agent_name="ExecutorAgent", thought="The ExecutorAgent uses the external fallback because future weather cannot be answered from historical averages.", action=tool_call, observation=observation))
        elif route.intent == "weather_historical":
            city = self._extract_city(normalized_query)
            month = self._extract_month(normalized_query)
            metric = self._extract_weather_metric(normalized_query)
            tool_call = ToolCall(name="query_weather", arguments={"city": "İSTANBUL" if normalize_text(city) == "istanbul" else city, "month": month, "metric": metric, "data_dir": self.data_dir})
            observation = self._execute_tool_call(tool_call)
            observations.append(observation)
            trace.append(AgentTraceStep(step_index=step_index, agent_name="ExecutorAgent", thought="The ExecutorAgent queries historical weather averages because the request is not future-oriented.", action=tool_call, observation=observation))
        elif route.intent == "preference_update":
            observation = {"status": "success", "tool": "semantic_memory", "data": {"memory_updates": plan.memory_updates, "contextual_preferences": plan.contextual_preferences}}
            observations.append(observation)
            trace.append(AgentTraceStep(step_index=step_index, agent_name="ExecutorAgent", thought="The ExecutorAgent does not call external tools for preference updates; the semantic memory layer has already saved the preference.", action=None, observation=observation))
        else:
            observation = {"status": "error", "error": {"code": "UNSUPPORTED_QUERY", "message": "No executable route was selected."}}
            observations.append(observation)
            trace.append(AgentTraceStep(step_index=step_index, agent_name="ExecutorAgent", thought="The ExecutorAgent cannot execute tools because the route is unsupported.", action=None, observation=observation))

        completed = any(item.get("status") == "success" for item in observations) if observations else False
        return ExecutorOutput(trace=trace, raw_observations=observations, completed=completed)

    def _execute_tool_call(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Execute a registered tool and parse JSON.

        Args:
            tool_call: Tool call.

        Returns:
            Parsed observation.
        """
        try:
            if tool_call.name not in self.tool_registry:
                return {"tool": tool_call.name, "status": "error", "error": {"code": "UNKNOWN_TOOL", "message": f"Tool '{tool_call.name}' is not registered."}}
            raw_observation = self.tool_registry[tool_call.name](**tool_call.arguments)
            parsed_observation = safe_json_loads(raw_observation)
            self.logger.info("Tool %s returned %s", tool_call.name, truncate_text(compact_json(parsed_observation), 500))
            return parsed_observation
        except Exception as exc:
            return {"tool": tool_call.name, "status": "error", "error": {"code": "TOOL_EXECUTION_FAILURE", "message": str(exc)}}

    def _extract_vehicle_type(self, normalized_query: str) -> Optional[str]:
        """Extract explicit vehicle type.

        Args:
            normalized_query: Normalized query.

        Returns:
            Vehicle type or None.
        """
        for candidate in ["suv", "sedan", "hatchback", "truck", "kamyon", "minivan", "van", "crossover"]:
            if candidate in normalized_query:
                return "truck" if candidate == "kamyon" else candidate
        return None

    def _extract_holiday_date(self, query: str) -> Optional[str]:
        """Extract Turkish date expression.

        Args:
            query: Original query.

        Returns:
            Date expression or None.
        """
        normalized = normalize_text(query)
        months = "ocak|subat|mart|nisan|mayis|haziran|temmuz|agustos|eylul|ekim|kasim|aralik"
        match = re.search(rf"\b(\d{{1,2}})\s+({months})\b", normalized)
        if match:
            return f"{int(match.group(1))} {match.group(2)}"
        return None

    def _extract_forecast_days(self, normalized_query: str) -> int:
        """Extract forecast horizon.

        Args:
            normalized_query: Normalized query.

        Returns:
            Number of forecast days.
        """
        if "hafta" in normalized_query:
            return 7
        match = re.search(r"\b(\d{1,2})\s*gun\b", normalized_query)
        if match:
            return max(1, min(14, int(match.group(1))))
        return 7

    def _extract_city(self, normalized_query: str) -> str:
        """Extract city.

        Args:
            normalized_query: Normalized query.

        Returns:
            City name.
        """
        if "istanbul" in normalized_query:
            return "İstanbul"
        return "İstanbul"

    def _extract_month(self, normalized_query: str) -> Optional[str]:
        """Extract month.

        Args:
            normalized_query: Normalized query.

        Returns:
            Month text or None.
        """
        for month in ["ocak", "subat", "mart", "nisan", "mayis", "haziran", "temmuz", "agustos", "eylul", "ekim", "kasim", "aralik", "yillik"]:
            if month in normalized_query:
                return month
        return None

    def _extract_weather_metric(self, normalized_query: str) -> str:
        """Infer weather metric.

        Args:
            normalized_query: Normalized query.

        Returns:
            Dataset metric name.
        """
        if "yagis" in normalized_query:
            return "Aylık Toplam Yağış Miktarı Ortalaması (mm)"
        if "en yuksek" in normalized_query or "maksimum" in normalized_query:
            return "Ortalama En Yüksek Sıcaklık (°C)"
        if "en dusuk" in normalized_query or "minimum" in normalized_query:
            return "Ortalama En Düşük Sıcaklık (°C)"
        return "Ortalama Sıcaklık (°C)"


class EditorCriticAgent:
    """Editor/Critic agent that verifies observations and writes Turkish responses."""

    def compose(self, plan: ExecutionPlan, executor_output: ExecutorOutput) -> ReflectionReport:
        """Create final Turkish answer with critic checks.

        Args:
            plan: Planner output.
            executor_output: Executor output.

        Returns:
            Reflection report.
        """
        route = plan.route
        observations = executor_output.raw_observations
        first_observation = next(
            (observation for observation in observations if observation.get("status") == "success"),
            observations[0] if observations else {},
        )
        draft_response = self._draft_response(route, first_observation, plan)
        critique_points, corrections, final_response = self._critique_and_revise(route, first_observation, draft_response, executor_output, plan)
        return ReflectionReport(draft_response=draft_response, critique_points=critique_points, corrections=corrections, final_response=final_response, passed=(len(critique_points) == 0 or len(corrections) > 0))

    def _draft_response(self, route: RouteDecision, observation: Dict[str, Any], plan: ExecutionPlan) -> str:
        """Draft a Turkish response.

        Args:
            route: Route decision.
            observation: Tool observation.
            plan: Execution plan.

        Returns:
            Turkish draft response.
        """
        if observation.get("status") != "success":
            return "İlgili araç veya veri kaynağı çalıştırılırken bir sorun oluştu."
        if route.intent == "vehicle_lookup":
            return self._draft_vehicle_response(observation, plan)
        if route.intent == "vehicle_chart":
            return self._draft_chart_response(observation)
        if route.intent in {"holiday_lookup", "holiday_list"}:
            return self._draft_holiday_response(observation)
        if route.intent == "weather_forecast":
            return self._draft_forecast_response(observation)
        if route.intent == "weather_historical":
            return self._draft_historical_weather_response(observation)
        if route.intent == "preference_update":
            updates = observation.get("data", {}).get("memory_updates", {})
            if updates.get("preferred_vehicle_type"):
                return f"Tercihinizi kaydettim: bundan sonraki araç sorularında mümkün olduğunda **{updates['preferred_vehicle_type'].upper()}** araç tipini bağlamsal tercih olarak dikkate alacağım."
            return "Tercih ifadeniz algılandı; ancak kaydedilecek net bir araç tipi bulunamadı."
        return "Sorunuzu mevcut veri kaynaklarıyla güvenilir biçimde eşleştiremedim. Araç yakıt tüketimi, resmî tatiller veya İstanbul hava durumu hakkında daha belirgin bir soru sorabilirsiniz."

    def _critique_and_revise(self, route: RouteDecision, observation: Dict[str, Any], draft_response: str, executor_output: ExecutorOutput, plan: ExecutionPlan) -> Tuple[List[str], List[str], str]:
        """Critique and revise the draft.

        Args:
            route: Route decision.
            observation: Tool observation.
            draft_response: Draft response.
            executor_output: Executor output.
            plan: Execution plan.

        Returns:
            Critique points, corrections, final response.
        """
        critique_points: List[str] = []
        corrections: List[str] = []
        final_response = draft_response
        if not draft_response.strip():
            critique_points.append("Draft response is empty.")
            final_response = "Cevap üretilemedi; lütfen sorunuzu daha açık biçimde yeniden yazın."
            corrections.append("Replaced empty draft with safe Turkish fallback.")
        if observation.get("status") == "error":
            critique_points.append("Executor returned a structured tool error.")
            final_response = self._build_error_response(observation)
            corrections.append("Converted structured error into user-safe Turkish explanation.")
        called_tools = [step.action.name for step in executor_output.trace if step.action is not None]
        for required_tool in plan.required_tools:
            if required_tool not in called_tools:
                critique_points.append(f"Required tool was not called: {required_tool}.")
        if route.intent == "weather_forecast" and "fetch_live_weather_api" not in called_tools:
            critique_points.append("Future weather query did not use external forecast fallback.")
        if route.intent == "weather_forecast" and "harici" not in final_response.lower():
            critique_points.append("Forecast answer does not clearly state the external fallback source.")
            final_response += " Bu cevap, tarihsel ortalama yerine harici tahmin aracından gelen gelecek gün verilerine dayanmaktadır."
            corrections.append("Added explicit external forecast-source statement.")
        if route.intent == "weather_historical" and "tahmin değildir" not in final_response.lower():
            critique_points.append("Historical weather answer may be mistaken for a forecast.")
            final_response += " Bu değer gelecek hava tahmini değildir."
            corrections.append("Added non-forecast limitation statement.")
        if route.intent == "vehicle_chart" and ".png" not in final_response:
            critique_points.append("Chart answer does not expose the PNG path.")
            corrections.append("Requested chart path visibility in final response.")
        return critique_points, corrections, final_response

    def _draft_vehicle_response(self, observation: Dict[str, Any], plan: ExecutionPlan) -> str:
        """Draft vehicle answer.

        Args:
            observation: Tool observation.
            plan: Execution plan.

        Returns:
            Turkish answer.
        """
        records = observation.get("data", {}).get("records", [])
        if not records:
            return "Uygun araç kaydı bulunamadı."
        vehicle = records[0]
        preferred_type = plan.contextual_preferences.get("preferred_vehicle_type")
        preference_note = ""
        if preferred_type and not self._query_explicitly_mentions_type(plan.user_query):
            preference_note = f" Kayıtlı tercihiniz nedeniyle analizde **{preferred_type.upper()}** araç tipi önceliklendirildi."
        return f"Veri setine göre en düşük yakıt tüketimine sahip uygun araç **{vehicle.get('brand')}** modelidir. Yakıt tüketimi **{vehicle.get('consumption')}** olarak kayıtlıdır. Araç tipi **{vehicle.get('type')}**, bagaj hacmi **{vehicle.get('luggage space(L)')} L**, oturma kapasitesi ise **{vehicle.get('seater')} kişidir**.{preference_note}"

    def _draft_chart_response(self, observation: Dict[str, Any]) -> str:
        """Draft chart answer.

        Args:
            observation: Tool observation.

        Returns:
            Turkish answer.
        """
        data = observation.get("data", {})
        fallback_info = data.get("preference_fallback", {})
        fallback_note = ""

        if fallback_info:
            fallback_note = (
                f" Kayıtlı **{fallback_info.get('requested_preference', '').upper()}** tercihiniz veri setinde bulunmadığı için "
                "grafik tüm araçlar üzerinden oluşturuldu."
            )

        return (
            f"Araçların yakıt tüketimlerini karşılaştıran akademik biçimli grafik başarıyla oluşturuldu. "
            f"PNG çıktısı: **{data.get('chart_path_png')}**. "
            f"PDF çıktısı: **{data.get('chart_path_pdf')}**. "
            f"Grafikte **{data.get('row_count')} araç** karşılaştırıldı."
            f"{fallback_note}"
        )

    def _draft_holiday_response(self, observation: Dict[str, Any]) -> str:
        """Draft holiday answer.

        Args:
            observation: Tool observation.

        Returns:
            Turkish answer.
        """
        records = observation.get("data", {}).get("records", [])
        if not records:
            return "İlgili tarih için resmî tatil kaydı bulunamadı."
        if len(records) == 1:
            holiday = records[0]
            return f"**{holiday.get('Tarih / Dönem')}** için veri setinde **{holiday.get('Tatil / Bayram')}** kaydı bulunmaktadır. Türü: **{holiday.get('Türü')}**. Süresi: **{holiday.get('Süre')}**."
        rows = "; ".join(f"{record.get('Tarih / Dönem')}: {record.get('Tatil / Bayram')} ({record.get('Süre')})" for record in records)
        return f"Veri setindeki resmî tatil kayıtları şunlardır: {rows}."

    def _draft_forecast_response(self, observation: Dict[str, Any]) -> str:
        """Draft forecast answer.

        Args:
            observation: Tool observation.

        Returns:
            Turkish answer.
        """
        data = observation.get("data", {})
        forecast = data.get("forecast", [])
        if not forecast:
            return "Tahmin verisi üretilemedi."
        forecast_lines = []
        for item in forecast[:7]:
            forecast_lines.append(f"{item.get('date')}: {item.get('condition_tr')}, {item.get('min_temp_c')}–{item.get('max_temp_c')}°C, yağış olasılığı %{item.get('precipitation_probability_percent')}")
        return f"Hava durumu Excel dosyası yalnızca tarihsel ortalamaları içerdiğinden, geleceğe yönelik bu soru için harici hava durumu tahmin aracı kullanıldı. **{data.get('city')}** için önümüzdeki **{data.get('days')} günün** özet tahmini: " + " | ".join(forecast_lines) + "."

    def _draft_historical_weather_response(self, observation: Dict[str, Any]) -> str:
        """Draft historical weather answer.

        Args:
            observation: Tool observation.

        Returns:
            Turkish answer.
        """
        data = observation.get("data", {})
        return f"**{data.get('city')}** için **{data.get('month')}** dönemindeki **{data.get('metric')}** değeri tarihsel Excel ortalamasına göre **{data.get('value')}** olarak kayıtlıdır. Bu değer gelecek hava tahmini değildir."

    def _build_error_response(self, observation: Dict[str, Any]) -> str:
        """Convert tool error to Turkish.

        Args:
            observation: Error observation.

        Returns:
            Turkish error response.
        """
        error = observation.get("error", {})
        code = error.get("code", "UNKNOWN_ERROR")
        message = error.get("message", "Unknown error")
        return f"İşlemi güvenilir şekilde tamamlayamadım; ancak sistem hatayı yakaladı ve uygulama çökmedi. Hata kodu: **{code}**. Teknik açıklama: **{message}**. Lütfen Excel dosyasının mevcut olduğunu ve beklenen sütun adlarının değiştirilmediğini kontrol edin."

    def _query_explicitly_mentions_type(self, query: str) -> bool:
        """Check if query explicitly mentions vehicle type.

        Args:
            query: User query.

        Returns:
            Whether a type is explicit.
        """
        normalized = normalize_text(query)
        return any(token in normalized for token in ["suv", "sedan", "hatchback", "truck", "kamyon", "van", "minivan"])


class DataAnalysisAgent:
    """Facade orchestrating Guardrail -> Planner -> Executor -> Editor/Critic."""

    def __init__(self, data_dir: str = "data", output_dir: str = "outputs", max_steps: int = 4, memory_size: int = 8, profile_path: str = "user_profile.json") -> None:
        """Initialize the multi-agent assistant.

        Args:
            data_dir: Dataset directory.
            output_dir: Chart output directory.
            max_steps: Reserved for future multi-step execution limits.
            memory_size: Short-term memory size.
            profile_path: Persistent user profile path.
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.max_steps = max(1, int(max_steps))
        self.guardrails = GuardrailValidator()
        self.memory = ConversationMemory(max_turns=memory_size)
        self.profile_memory = DynamicSemanticMemory(profile_path=profile_path)
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent(data_dir=data_dir, output_dir=output_dir)
        self.editor = EditorCriticAgent()

    def answer(self, user_query: str) -> str:
        """Return final Turkish response only.

        Args:
            user_query: User query.

        Returns:
            Final Turkish response.
        """
        return self.run(user_query).final_response

    def run(self, user_query: str) -> AgentRunResult:
        """Run the full hierarchical multi-agent pipeline.

        Args:
            user_query: User query.

        Returns:
            Complete agent run result.
        """
        guardrail_result = self.guardrails.validate(user_query)
        if not guardrail_result.is_allowed:
            route = RouteDecision(intent="blocked", dataset="none", confidence=1.0, reason=f"Guardrail blocked prompt: {guardrail_result.reason_code}.", guardrail_triggered=True)
            trace = [AgentTraceStep(step_index=1, agent_name="GuardrailValidator", thought="The prompt was blocked before planning because it matched a security guardrail.", action=None, observation={"status": "blocked", "guardrail": dataclass_to_dict(guardrail_result)})]
            reflection = ReflectionReport(draft_response=guardrail_result.message_tr, critique_points=[], corrections=[], final_response=guardrail_result.message_tr, passed=True)
            return AgentRunResult(user_query=user_query, route=dataclass_to_dict(route), trace=trace, reflection=reflection, final_response=guardrail_result.message_tr)

        normalized_query = normalize_text(user_query)
        memory_updates = self.profile_memory.update_from_query(user_query)
        route = self._semantic_route(normalized_query, memory_updates)
        contextual_preferences = {"preferred_vehicle_type": self.profile_memory.get_preferred_vehicle_type()}
        plan = self.planner.create_plan(user_query=user_query, route=route, memory_updates=memory_updates, contextual_preferences=contextual_preferences)
        executor_output = self.executor.execute(plan=plan, normalized_query=normalized_query)
        reflection = self.editor.compose(plan=plan, executor_output=executor_output)
        result = AgentRunResult(user_query=user_query, route=dataclass_to_dict(route), trace=executor_output.trace, reflection=reflection, final_response=reflection.final_response)
        self.memory.add_turn(user_query=user_query, assistant_response=reflection.final_response, metadata={"route": dataclass_to_dict(route), "plan": dataclass_to_dict(plan), "reflection": dataclass_to_dict(reflection)})
        return result

    def _semantic_route(self, normalized_query: str, memory_updates: Dict[str, Any]) -> RouteDecision:
        """Route query to an intent.

        Args:
            normalized_query: Normalized query.
            memory_updates: Extracted semantic memory updates.

        Returns:
            Route decision.
        """
        if memory_updates and not self._contains_data_request(normalized_query):
            return RouteDecision(intent="preference_update", dataset="semantic_memory", confidence=0.96, reason="The prompt primarily states a durable user preference.")
        vehicle_keywords = ["arac", "araba", "otomobil", "yakit", "tuketim", "sedan", "hatchback", "suv", "bagaj", "koltuk"]
        chart_keywords = ["grafik", "grafigi", "grafikleri", "ciz", "gorsel", "plot", "chart", "karsilastiran", "karsilastir"]
        holiday_keywords = ["tatil", "bayram", "resmi", "23 nisan", "19 mayis", "29 ekim", "30 agustos", "ramazan", "kurban"]
        weather_keywords = ["hava", "sicaklik", "yagis", "istanbul", "meteoroloji", "tahmin"]
        future_keywords = ["onumuzdeki", "gelecek", "yarin", "haftaya", "tahmin", "forecast", "sonraki"]
        if any(keyword in normalized_query for keyword in vehicle_keywords) and any(keyword in normalized_query for keyword in chart_keywords):
            return RouteDecision(intent="vehicle_chart", dataset="vehicles", confidence=0.97, reason="The query requests a visual comparison of vehicle fuel consumption.")
        if any(keyword in normalized_query for keyword in holiday_keywords):
            if "liste" in normalized_query or "nelerdir" in normalized_query:
                return RouteDecision(intent="holiday_list", dataset="holidays", confidence=0.94, reason="The query asks to list official holidays.")
            return RouteDecision(intent="holiday_lookup", dataset="holidays", confidence=0.96, reason="The query asks about a specific official holiday or date.")
        if any(keyword in normalized_query for keyword in weather_keywords):
            if any(keyword in normalized_query for keyword in future_keywords):
                return RouteDecision(intent="weather_forecast", dataset="external_weather_api", confidence=0.98, reason="Future weather requires the external forecast fallback because Excel contains historical averages only.")
            return RouteDecision(intent="weather_historical", dataset="weather", confidence=0.91, reason="The query can be answered from historical weather averages.")
        if any(keyword in normalized_query for keyword in vehicle_keywords):
            return RouteDecision(intent="vehicle_lookup", dataset="vehicles", confidence=0.95, reason="The query asks for vehicle fuel-consumption analysis.")
        return RouteDecision(intent="unknown", dataset="none", confidence=0.20, reason="The query does not clearly match vehicles, official holidays, or Istanbul weather data.")

    def _contains_data_request(self, normalized_query: str) -> bool:
        """Check whether the prompt asks for analysis rather than only stating a preference.

        Args:
            normalized_query: Normalized query.

        Returns:
            Whether the query asks for data analysis.
        """
        request_markers = ["hangisi", "nedir", "kac", "liste", "goster", "ciz", "grafik", "karsilastir", "hava nasil", "tatil mi"]
        return any(marker in normalized_query for marker in request_markers)
