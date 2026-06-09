"""Tool layer for Excel analysis, external fallback, and research-grade charts.

Every public tool includes defensive error management and returns JSON strings.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import normalize_text


DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("outputs")

DATASET_FILES: Dict[str, Tuple[str, ...]] = {
    "vehicles": ("vehicles_2.xlsx", "vehicles.xlsx"),
    "holidays": ("holidays_2.xlsx", "holidays.xlsx"),
    "weather": ("weather_2.xlsx", "weather.xlsx"),
}

MONTH_ALIASES: Dict[str, str] = {
    "ocak": "Ocak", "subat": "Şubat", "mart": "Mart", "nisan": "Nisan", "mayis": "Mayıs",
    "haziran": "Haziran", "temmuz": "Temmuz", "agustos": "Ağustos", "eylul": "Eylül",
    "ekim": "Ekim", "kasim": "Kasım", "aralik": "Aralık", "yillik": "Yıllık",
}


def _json_response(tool_name: str, status: str, data: Optional[Dict[str, Any]] = None, error: Optional[Dict[str, Any]] = None, message: Optional[str] = None) -> str:
    """Create a stable JSON tool response.

    Args:
        tool_name: Tool name.
        status: Response status.
        data: Optional data payload.
        error: Optional error payload.
        message: Optional readable message.

    Returns:
        JSON string.
    """
    try:
        payload: Dict[str, Any] = {"tool": tool_name, "status": status}
        if message is not None:
            payload["message"] = message
        if data is not None:
            payload["data"] = data
        if error is not None:
            payload["error"] = error
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception as exc:
        return json.dumps({"tool": tool_name, "status": "error", "error": {"code": "JSON_SERIALIZATION_ERROR", "message": str(exc)}}, ensure_ascii=False)


def _json_success(tool_name: str, data: Dict[str, Any], message: Optional[str] = None) -> str:
    """Create a success JSON response.

    Args:
        tool_name: Tool name.
        data: Data payload.
        message: Optional message.

    Returns:
        JSON string.
    """
    return _json_response(tool_name=tool_name, status="success", data=data, message=message)


def _json_error(tool_name: str, code: str, message: str, details: Optional[Dict[str, Any]] = None) -> str:
    """Create an error JSON response.

    Args:
        tool_name: Tool name.
        code: Machine-readable error code.
        message: Human-readable error message.
        details: Optional diagnostics.

    Returns:
        JSON string.
    """
    return _json_response(tool_name=tool_name, status="error", error={"code": code, "message": message, "details": details or {}})


def _safe_int(value: Any, default: int, lower: int, upper: int) -> int:
    """Convert a value to a bounded integer.

    Args:
        value: Input value.
        default: Default value.
        lower: Lower bound.
        upper: Upper bound.

    Returns:
        Bounded integer.
    """
    try:
        number = int(value)
        return max(lower, min(upper, number))
    except Exception:
        return max(lower, min(upper, default))


def _resolve_file_path(dataset_key: str, data_dir: Optional[str] = None) -> Path:
    """Resolve a dataset path.

    Args:
        dataset_key: Logical dataset key.
        data_dir: Optional data directory.

    Returns:
        Existing Excel path.
    """
    if dataset_key not in DATASET_FILES:
        raise KeyError(f"Unknown dataset key: {dataset_key}")

    search_roots = []
    if data_dir:
        search_roots.append(Path(data_dir))
    search_roots.extend([DEFAULT_DATA_DIR, Path("."), Path("/content/data"), Path("/content")])

    checked_paths: List[str] = []
    for root in search_roots:
        for file_name in DATASET_FILES[dataset_key]:
            candidate = root / file_name
            checked_paths.append(str(candidate))
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Dataset file was not found. Checked paths: {checked_paths}")


def _read_excel_dataset(dataset_key: str, data_dir: Optional[str] = None) -> pd.DataFrame:
    """Read an Excel dataset.

    Args:
        dataset_key: Logical dataset key.
        data_dir: Optional data directory.

    Returns:
        Loaded DataFrame.
    """
    file_path = _resolve_file_path(dataset_key, data_dir=data_dir)
    return pd.read_excel(file_path)


def _validate_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> None:
    """Validate required columns.

    Args:
        df: DataFrame.
        required_columns: Required column names.
    """
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing expected columns: {missing_columns}. Available columns: {list(df.columns)}")


def _records_from_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records.

    Args:
        df: Source DataFrame.

    Returns:
        List of record dictionaries.
    """
    clean_df = df.where(pd.notna(df), None)
    return clean_df.to_dict(orient="records")


def _extract_date_token(date_text: Optional[str]) -> Optional[str]:
    """Extract a day-month token from Turkish natural language.

    Args:
        date_text: Date text.

    Returns:
        Normalized date token.
    """
    if not date_text:
        return None
    normalized = normalize_text(date_text)
    month_pattern = "ocak|subat|mart|nisan|mayis|haziran|temmuz|agustos|eylul|ekim|kasim|aralik"
    match = re.search(rf"\b(\d{{1,2}})\s+({month_pattern})\b", normalized)
    if match:
        return f"{int(match.group(1))} {match.group(2)}"
    return normalized


def _normalize_month(month: Optional[str]) -> str:
    """Normalize a month name.

    Args:
        month: Month text.

    Returns:
        Dataset month column.
    """
    if not month:
        return "Yıllık"
    normalized = normalize_text(month)
    for alias, canonical in MONTH_ALIASES.items():
        if normalized == alias or alias in normalized.split():
            return canonical
    raise ValueError(f"Unsupported month value: {month}")


def query_vehicles(vehicle_type: Optional[str] = None, sort_by: str = "consumption", ascending: bool = True, top_n: int = 5, data_dir: Optional[str] = None) -> str:
    """Query vehicle fuel-consumption data.

    Args:
        vehicle_type: Optional vehicle type filter.
        sort_by: Column used for sorting.
        ascending: Sort direction.
        top_n: Maximum records returned.
        data_dir: Optional dataset directory.

    Returns:
        JSON string with records or structured error.
    """
    tool_name = "query_vehicles"
    try:
        df = _read_excel_dataset("vehicles", data_dir=data_dir)
        required_columns = ["brand", "consumption", "type", "luggage space(L)", "seater"]
        _validate_columns(df, required_columns)

        if sort_by not in required_columns:
            return _json_error(tool_name, "INVALID_SORT_COLUMN", f"The requested sort column '{sort_by}' is not available.", {"available_columns": required_columns})

        work = df.copy()
        work["consumption"] = pd.to_numeric(work["consumption"], errors="coerce")
        work["seater"] = pd.to_numeric(work["seater"], errors="coerce")
        work["luggage space(L)"] = pd.to_numeric(work["luggage space(L)"], errors="coerce")

        if vehicle_type:
            normalized_type = normalize_text(vehicle_type)
            work = work[work["type"].apply(lambda item: normalized_type in normalize_text(item))]

        work = work.dropna(subset=["brand", "consumption"])
        if work.empty:
            return _json_error(tool_name, "NO_MATCHING_VEHICLES", "No vehicles matched the requested filters.", {"vehicle_type": vehicle_type})

        bounded_top_n = _safe_int(top_n, default=5, lower=1, upper=50)
        result = work.sort_values(by=sort_by, ascending=bool(ascending), na_position="last").head(bounded_top_n)

        return _json_success(tool_name, {"records": _records_from_dataframe(result), "row_count": int(len(result)), "filters": {"vehicle_type": vehicle_type}, "sort": {"sort_by": sort_by, "ascending": bool(ascending)}, "source_type": "excel_vehicle_dataset", "dataset_limitation": "The answer is limited to rows available in the Excel file."})
    except FileNotFoundError as exc:
        return _json_error(tool_name, "FILE_NOT_FOUND", str(exc))
    except KeyError as exc:
        return _json_error(tool_name, "COLUMN_SCHEMA_ERROR", str(exc))
    except Exception as exc:
        return _json_error(tool_name, "UNEXPECTED_TOOL_ERROR", str(exc))


def query_holidays(date_text: Optional[str] = None, holiday_name: Optional[str] = None, list_all: bool = False, data_dir: Optional[str] = None) -> str:
    """Query official holidays.

    Args:
        date_text: Natural date text.
        holiday_name: Optional holiday name.
        list_all: Whether to return all records.
        data_dir: Optional dataset directory.

    Returns:
        JSON string with matched records or structured error.
    """
    tool_name = "query_holidays"
    try:
        df = _read_excel_dataset("holidays", data_dir=data_dir)
        required_columns = ["Tarih / Dönem", "Gün", "Tatil / Bayram", "Türü", "Süre"]
        _validate_columns(df, required_columns)
        work = df.copy()

        if list_all:
            return _json_success(tool_name, {"records": _records_from_dataframe(work), "row_count": int(len(work)), "source_type": "excel_holiday_dataset"})

        used_filter = False
        mask = pd.Series([True] * len(work), index=work.index)

        date_token = _extract_date_token(date_text)
        if date_token:
            used_filter = True
            mask &= work["Tarih / Dönem"].apply(lambda value: date_token in normalize_text(value) or normalize_text(value) in date_token)

        if holiday_name:
            used_filter = True
            normalized_name = normalize_text(holiday_name)
            mask &= work["Tatil / Bayram"].apply(lambda value: normalized_name in normalize_text(value)) | work["Gün"].apply(lambda value: normalized_name in normalize_text(value))

        if not used_filter:
            return _json_error(tool_name, "INSUFFICIENT_QUERY", "Provide date_text, holiday_name, or list_all=True.")

        result = work[mask]
        if result.empty:
            return _json_error(tool_name, "NO_MATCHING_HOLIDAY", "No holiday matched the requested date or name.", {"date_text": date_text, "holiday_name": holiday_name})

        return _json_success(tool_name, {"records": _records_from_dataframe(result), "row_count": int(len(result)), "query": {"date_text": date_text, "holiday_name": holiday_name, "list_all": False}, "source_type": "excel_holiday_dataset"})
    except FileNotFoundError as exc:
        return _json_error(tool_name, "FILE_NOT_FOUND", str(exc))
    except KeyError as exc:
        return _json_error(tool_name, "COLUMN_SCHEMA_ERROR", str(exc))
    except Exception as exc:
        return _json_error(tool_name, "UNEXPECTED_TOOL_ERROR", str(exc))


def query_weather(city: str = "İSTANBUL", month: Optional[str] = None, metric: str = "Ortalama Sıcaklık (°C)", data_dir: Optional[str] = None) -> str:
    """Query historical weather averages.

    Args:
        city: City column.
        month: Month name.
        metric: Weather metric.
        data_dir: Optional dataset directory.

    Returns:
        JSON string with historical weather value or structured error.
    """
    tool_name = "query_weather"
    try:
        df = _read_excel_dataset("weather", data_dir=data_dir)
        city_column = None
        for column in df.columns:
            if normalize_text(column) == normalize_text(city):
                city_column = column
                break
        if city_column is None:
            return _json_error(tool_name, "CITY_COLUMN_NOT_FOUND", f"City column '{city}' was not found.", {"available_columns": list(df.columns)})

        month_column = _normalize_month(month)
        if month_column not in df.columns:
            return _json_error(tool_name, "MONTH_COLUMN_NOT_FOUND", f"Month column '{month_column}' was not found.", {"available_columns": list(df.columns)})

        normalized_metric = normalize_text(metric)
        metric_mask = df[city_column].apply(lambda value: normalized_metric in normalize_text(value) or normalize_text(value) in normalized_metric)
        result = df[metric_mask]

        if result.empty:
            return _json_error(tool_name, "METRIC_NOT_FOUND", f"Metric '{metric}' was not found.", {"available_metrics": df[city_column].dropna().astype(str).tolist()})

        row = result.iloc[0]
        value = row[month_column]
        return _json_success(tool_name, {"city": city_column, "month": month_column, "metric": str(row[city_column]), "value": None if pd.isna(value) else float(value), "source_type": "historical_average_excel", "forecast_supported": False, "dataset_limitation": "The Excel weather file contains historical averages only, not future forecasts."})
    except FileNotFoundError as exc:
        return _json_error(tool_name, "FILE_NOT_FOUND", str(exc))
    except KeyError as exc:
        return _json_error(tool_name, "COLUMN_SCHEMA_ERROR", str(exc))
    except ValueError as exc:
        return _json_error(tool_name, "INVALID_MONTH", str(exc))
    except Exception as exc:
        return _json_error(tool_name, "UNEXPECTED_TOOL_ERROR", str(exc))


def fetch_live_weather_api(city: str = "İstanbul", days: int = 7) -> str:
    """Mock a future weather forecast API.

    Args:
        city: Forecast city.
        days: Number of days.

    Returns:
        JSON string with deterministic forecast records.
    """
    tool_name = "fetch_live_weather_api"
    try:
        bounded_days = _safe_int(days, default=7, lower=1, upper=14)
        today = date.today()
        seed_source = f"{city}-{today.isoformat()}-{bounded_days}"
        seed = int(hashlib.sha256(seed_source.encode("utf-8")).hexdigest()[:8], 16)
        conditions = [
            {"en": "partly_cloudy", "tr": "parçalı bulutlu"},
            {"en": "sunny", "tr": "güneşli"},
            {"en": "light_rain", "tr": "hafif yağmurlu"},
            {"en": "cloudy", "tr": "bulutlu"},
        ]
        forecast: List[Dict[str, Any]] = []
        for offset in range(1, bounded_days + 1):
            daily_seed = seed + offset * 31
            condition = conditions[daily_seed % len(conditions)]
            min_temp = 11 + daily_seed % 9
            max_temp = min_temp + 5 + daily_seed % 5
            precipitation_probability = 10 + daily_seed % 60
            forecast.append({"date": (today + timedelta(days=offset)).isoformat(), "city": city, "condition_code": condition["en"], "condition_tr": condition["tr"], "min_temp_c": float(min_temp), "max_temp_c": float(max_temp), "precipitation_probability_percent": int(precipitation_probability)})

        return _json_success(tool_name, {"city": city, "days": bounded_days, "forecast": forecast, "source_type": "mock_external_weather_api", "production_note": "Replace this deterministic mock with a real requests/httpx client when a live API key is available."})
    except Exception as exc:
        return _json_error(tool_name, "EXTERNAL_API_FALLBACK_ERROR", str(exc))


def create_vehicle_consumption_chart(vehicle_type: Optional[str] = None, output_dir: Optional[str] = None, data_dir: Optional[str] = None) -> str:
    """Create a publication-quality fuel-consumption chart.

    Args:
        vehicle_type: Optional vehicle type filter.
        output_dir: Optional output directory.
        data_dir: Optional dataset directory.

    Returns:
        JSON string containing chart paths and plotted records.
    """
    tool_name = "create_vehicle_consumption_chart"
    try:
        df = _read_excel_dataset("vehicles", data_dir=data_dir)
        required_columns = ["brand", "consumption", "type", "luggage space(L)", "seater"]
        _validate_columns(df, required_columns)
        work = df.copy()
        work["consumption"] = pd.to_numeric(work["consumption"], errors="coerce")
        work = work.dropna(subset=["brand", "consumption"])

        if vehicle_type:
            normalized_type = normalize_text(vehicle_type)
            work = work[work["type"].apply(lambda item: normalized_type in normalize_text(item))]

        if work.empty:
            return _json_error(tool_name, "NO_CHART_DATA", "No vehicle rows are available for chart generation.", {"vehicle_type": vehicle_type})

        work = work.sort_values(by="consumption", ascending=True)
        output_root = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        output_root.mkdir(parents=True, exist_ok=True)
        suffix = normalize_text(vehicle_type) if vehicle_type else "all"
        png_path = output_root / f"vehicle_consumption_comparison_{suffix}.png"
        pdf_path = output_root / f"vehicle_consumption_comparison_{suffix}.pdf"

        sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
        plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.family": "serif", "axes.titlesize": 14, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10})
        figure_width = max(9.0, min(16.0, 0.75 * len(work)))
        fig, ax = plt.subplots(figsize=(figure_width, 5.8))

        sns.barplot(data=work, x="brand", y="consumption", hue="type", dodge=False, ax=ax, edgecolor="black", linewidth=0.7)
        title_suffix = f" ({vehicle_type})" if vehicle_type else ""
        ax.set_title(f"Vehicle Fuel Consumption Comparison{title_suffix}", pad=14)
        ax.set_xlabel("Vehicle Brand / Model")
        ax.set_ylabel("Fuel Consumption")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.75)
        ax.tick_params(axis="x", rotation=35)

        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)

        legend = ax.get_legend()
        if legend is not None:
            legend.set_title("Vehicle Type")
            legend.set_frame_on(True)

        fig.tight_layout()
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return _json_success(tool_name, {"chart_path_png": str(png_path), "chart_path_pdf": str(pdf_path), "row_count": int(len(work)), "records": _records_from_dataframe(work), "visualization_standard": {"library": "seaborn+matplotlib", "dpi": 300, "style": "whitegrid", "layout": "tight_layout", "export_formats": ["png", "pdf"]}, "source_type": "excel_vehicle_dataset"})
    except FileNotFoundError as exc:
        return _json_error(tool_name, "FILE_NOT_FOUND", str(exc))
    except KeyError as exc:
        return _json_error(tool_name, "COLUMN_SCHEMA_ERROR", str(exc))
    except Exception as exc:
        return _json_error(tool_name, "CHART_GENERATION_ERROR", str(exc))
