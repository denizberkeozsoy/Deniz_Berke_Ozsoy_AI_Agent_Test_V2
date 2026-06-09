"""Utility layer for the multi-agent data analysis assistant.

This module provides safe JSON handling, Turkish-aware text normalization,
structured trace dataclasses, security guardrails, and persistent semantic memory.
"""

from __future__ import annotations

import html
import json
import logging
import re
import traceback
import unicodedata
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ToolCall:
    """A structured tool/function call.

    Attributes:
        name: Tool name.
        arguments: JSON-serializable tool arguments.
    """

    name: str
    arguments: Dict[str, Any]


@dataclass
class AgentTraceStep:
    """A visible multi-agent trace step.

    Attributes:
        step_index: Step number.
        thought: English high-level reasoning statement.
        action: Optional tool call.
        observation: Optional parsed tool observation.
        agent_name: Name of the specialized agent that produced the step.
    """

    step_index: int
    thought: str
    action: Optional[ToolCall] = None
    observation: Optional[Dict[str, Any]] = None
    agent_name: str = "ExecutorAgent"


@dataclass
class ReflectionReport:
    """Editor/Critic reflection report.

    Attributes:
        draft_response: Initial Turkish draft.
        critique_points: English critique points.
        corrections: English correction actions.
        final_response: Final professional Turkish response.
        passed: Whether the answer passed critic checks.
    """

    draft_response: str
    critique_points: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)
    final_response: str = ""
    passed: bool = False


@dataclass
class AgentRunResult:
    """Complete output returned by DataAnalysisAgent.

    Attributes:
        user_query: Original user query.
        route: Semantic route metadata.
        trace: Planner/Executor trace steps.
        reflection: Editor/Critic report.
        final_response: Final Turkish answer.
        created_at: UTC timestamp.
    """

    user_query: str
    route: Dict[str, Any]
    trace: List[AgentTraceStep]
    reflection: ReflectionReport
    final_response: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ConversationTurn:
    """A conversation memory turn.

    Attributes:
        user_query: User query.
        assistant_response: Final response.
        metadata: Optional metadata.
        created_at: UTC timestamp.
    """

    user_query: str
    assistant_response: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class GuardrailResult:
    """Security validation result.

    Attributes:
        is_allowed: Whether the prompt is safe to process.
        reason_code: Machine-readable reason.
        message_tr: Turkish user-facing message.
        matched_terms: Matched suspicious terms.
    """

    is_allowed: bool
    reason_code: str
    message_tr: str
    matched_terms: List[str] = field(default_factory=list)


class ConversationMemory:
    """Bounded short-term memory for recent turns."""

    def __init__(self, max_turns: int = 8) -> None:
        """Initialize memory.

        Args:
            max_turns: Maximum number of conversation turns.
        """
        self.max_turns = max(1, int(max_turns))
        self._turns: Deque[ConversationTurn] = deque(maxlen=self.max_turns)

    def add_turn(self, user_query: str, assistant_response: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a conversation turn.

        Args:
            user_query: User query.
            assistant_response: Assistant answer.
            metadata: Optional metadata.
        """
        self._turns.append(
            ConversationTurn(
                user_query=user_query,
                assistant_response=assistant_response,
                metadata=metadata or {},
            )
        )

    def recent_turns(self, limit: Optional[int] = None) -> List[ConversationTurn]:
        """Return recent turns.

        Args:
            limit: Optional number of turns.

        Returns:
            Conversation turns in chronological order.
        """
        turns = list(self._turns)
        if limit is None:
            return turns
        return turns[-max(1, int(limit)):]

    def clear(self) -> None:
        """Clear short-term memory."""
        self._turns.clear()

    def as_prompt_context(self, limit: int = 4) -> str:
        """Serialize memory as compact text.

        Args:
            limit: Maximum number of turns.

        Returns:
            Textual memory context.
        """
        blocks: List[str] = []
        for turn in self.recent_turns(limit):
            blocks.append(f"User: {turn.user_query}\nAssistant: {turn.assistant_response}")
        return "\n---\n".join(blocks)


class GuardrailValidator:
    """Prompt-level security validator.

    The validator blocks prompt injection, destructive file operations,
    credential exfiltration attempts, and requests to bypass system rules.
    """

    def __init__(self) -> None:
        """Initialize guardrail patterns."""
        self.block_patterns: Dict[str, List[str]] = {
            "PROMPT_INJECTION": [
                r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
                r"forget\s+(all\s+)?(previous|prior|above)\s+instructions",
                r"system\s+prompt",
                r"developer\s+message",
                r"jailbreak",
                r"bypass\s+(the\s+)?rules",
                r"act\s+as\s+dan",
                r"reveal\s+(your\s+)?hidden",
                r"show\s+(your\s+)?chain\s+of\s+thought",
            ],
            "DESTRUCTIVE_OPERATION": [
                r"delete\s+system\s+files",
                r"remove\s+all\s+files",
                r"format\s+(the\s+)?disk",
                r"rm\s+-rf",
                r"shutil\.rmtree",
                r"os\.remove",
                r"del\s+/f",
            ],
            "CREDENTIAL_EXFILTRATION": [
                r"print\s+os\.environ",
                r"show\s+api\s+key",
                r"leak\s+api\s+key",
                r"steal\s+(password|token|credential)",
                r"read\s+\.env",
                r"cat\s+\.env",
                r"exfiltrate",
            ],
        }

    def validate(self, prompt: str) -> GuardrailResult:
        """Validate a user prompt before planning.

        Args:
            prompt: Raw user prompt.

        Returns:
            Guardrail validation result.
        """
        normalized = normalize_text(prompt)

        for reason_code, patterns in self.block_patterns.items():
            matched_terms: List[str] = []
            for pattern in patterns:
                if re.search(pattern, normalized, flags=re.IGNORECASE):
                    matched_terms.append(pattern)

            if matched_terms:
                return GuardrailResult(
                    is_allowed=False,
                    reason_code=reason_code,
                    matched_terms=matched_terms,
                    message_tr=(
                        "Bu isteği güvenlik nedeniyle işleyemem. "
                        "Sistem talimatlarını aşmaya, gizli bilgileri açığa çıkarmaya veya dosya sistemi üzerinde "
                        "zararlı işlem yapmaya yönelik talepler desteklenmez. "
                        "Araç verileri, resmî tatiller veya İstanbul hava durumu hakkında güvenli bir soru sorabilirsiniz."
                    ),
                )

        return GuardrailResult(
            is_allowed=True,
            reason_code="SAFE",
            matched_terms=[],
            message_tr="İstek güvenli görünüyor.",
        )


class DynamicSemanticMemory:
    """Persistent user preference memory stored in JSON format."""

    def __init__(self, profile_path: str = "user_profile.json") -> None:
        """Initialize dynamic profile memory.

        Args:
            profile_path: JSON file path for persistent preferences.
        """
        self.profile_path = Path(profile_path)
        self.profile: Dict[str, Any] = self._load_profile()

    def _load_profile(self) -> Dict[str, Any]:
        """Load profile from disk.

        Returns:
            User profile dictionary.
        """
        if not self.profile_path.exists():
            return self._default_profile()

        try:
            with self.profile_path.open("r", encoding="utf-8") as file:
                loaded = json.load(file)
            if isinstance(loaded, dict):
                return loaded
            return self._default_profile()
        except Exception:
            return self._default_profile()

    def _default_profile(self) -> Dict[str, Any]:
        """Return default profile.

        Returns:
            Empty profile schema.
        """
        return {
            "vehicle_preferences": {},
            "weather_preferences": {},
            "updated_at": None,
        }

    def save(self) -> None:
        """Persist profile to disk."""
        self.profile["updated_at"] = datetime.now(timezone.utc).isoformat()
        with self.profile_path.open("w", encoding="utf-8") as file:
            json.dump(self.profile, file, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        """Clear and persist user profile."""
        self.profile = self._default_profile()
        self.save()

    def get_preferred_vehicle_type(self) -> Optional[str]:
        """Return preferred vehicle type if available.

        Returns:
            Preferred vehicle type or None.
        """
        value = self.profile.get("vehicle_preferences", {}).get("preferred_type")
        return str(value) if value else None

    def update_from_query(self, query: str) -> Dict[str, Any]:
        """Extract durable user preferences from a natural-language query.

        Args:
            query: User query.

        Returns:
            Dictionary describing extracted updates.
        """
        normalized = normalize_text(query)
        updates: Dict[str, Any] = {}

        preference_markers = [
            "usually", "prefer", "preference", "genellikle", "tercih", "kiralarim",
            "kiraliyorum", "severim", "favorim",
        ]

        vehicle_type_map = {
            "suv": "suv",
            "sedan": "sedan",
            "hatchback": "hatchback",
            "minivan": "minivan",
            "van": "van",
            "truck": "truck",
            "kamyon": "truck",
            "crossover": "crossover",
        }

        if any(marker in normalized for marker in preference_markers):
            for token, canonical_type in vehicle_type_map.items():
                if token in normalized:
                    self.profile.setdefault("vehicle_preferences", {})["preferred_type"] = canonical_type
                    updates["preferred_vehicle_type"] = canonical_type
                    break

        if "istanbul" in normalized and any(marker in normalized for marker in preference_markers):
            self.profile.setdefault("weather_preferences", {})["preferred_city"] = "İstanbul"
            updates["preferred_city"] = "İstanbul"

        if updates:
            self.save()

        return updates


def setup_logger(name: str = "multi_agent_data_assistant", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger.

    Args:
        name: Logger name.
        log_file: Optional log file.
        level: Logging level.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def normalize_text(value: Any) -> str:
    """Normalize text for Turkish-aware semantic matching.

    Args:
        value: Input value.

    Returns:
        Lowercase, accent-folded text.
    """
    if value is None:
        return ""

    raw = str(value).strip().lower()
    replacements = {"ı": "i", "İ": "i", "ğ": "g", "ü": "u", "ş": "s", "ö": "o", "ç": "c"}
    for source, target in replacements.items():
        raw = raw.replace(source, target)

    raw = unicodedata.normalize("NFKD", raw)
    raw = "".join(character for character in raw if not unicodedata.combining(character))
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def safe_json_loads(payload: Any) -> Dict[str, Any]:
    """Safely parse JSON.

    Args:
        payload: JSON string or dictionary.

    Returns:
        Parsed dictionary or structured error.
    """
    if isinstance(payload, dict):
        return payload

    if not isinstance(payload, str):
        return {"status": "error", "error": {"code": "INVALID_PAYLOAD_TYPE", "message": f"Expected JSON string, received {type(payload).__name__}."}}

    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
        return {"status": "error", "error": {"code": "INVALID_JSON_ROOT", "message": "The JSON root must be an object."}}
    except json.JSONDecodeError as exc:
        return {"status": "error", "error": {"code": "JSON_DECODE_ERROR", "message": str(exc), "raw_payload": payload[:500]}}


def compact_json(data: Dict[str, Any]) -> str:
    """Serialize a dictionary compactly.

    Args:
        data: Dictionary.

    Returns:
        JSON string.
    """
    return json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)


def truncate_text(text: str, max_length: int = 1200) -> str:
    """Truncate long text.

    Args:
        text: Input text.
        max_length: Maximum length.

    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def html_escape(value: Any) -> str:
    """Escape text for HTML display.

    Args:
        value: Input value.

    Returns:
        HTML-safe string.
    """
    return html.escape(str(value), quote=True)


def safe_execute(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[bool, Any]:
    """Execute a function with exception capture.

    Args:
        function: Callable.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Tuple of success flag and result/error object.
    """
    try:
        return True, function(*args, **kwargs)
    except Exception as exc:
        return False, {"status": "error", "error": {"code": "SAFE_EXECUTION_ERROR", "message": str(exc), "traceback": traceback.format_exc(limit=3)}}


def dataclass_to_dict(instance: Any) -> Dict[str, Any]:
    """Convert dataclass to dictionary safely.

    Args:
        instance: Dataclass instance.

    Returns:
        Dictionary representation.
    """
    try:
        return asdict(instance)
    except Exception:
        return {"repr": repr(instance)}
