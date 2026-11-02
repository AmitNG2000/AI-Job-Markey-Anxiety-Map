# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: ai_map_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Imports

# %%
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urlparse
import zipfile

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from tqdm import tqdm

# %% [markdown]
# # Parameters

# %%
# Notebook parameters. Override these in Papermill or with environment
# In BGU HPC cluster, downloading 2 months of gdelt data, took about 5 days.
START_DATE = datetime(2023, 1, 1)   # YYYY-MM-DD
END_DATE = datetime(2023, 3, 1)     # YYYY-MM-DD
AGGREGATION_PERIOD = "7D"
SAVE_INTERVAL = "1M"

# Optional environment variables:
# GDELT_DATE_START=2023-01-01
# GDELT_DATE_END=2023-01-31
# GDELT_AGGREGATION_PERIOD=7D
# GDELT_SAVE_INTERVAL=1M
# GDELT_JOB_ID=jan_2023_weekly

# %%
MAX_DOWNLOAD_WORKERS = 8

GDELT_MASTER_FILE_LIST_URLS = {
    "main": "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt",
    "translation": (
        "http://data.gdeltproject.org/gdeltv2/"
        "masterfilelist-translation.txt"
    ),
}
# Allow slow GDELT responses to remain idle for up to 10 minutes per attempt.
REQUEST_TIMEOUT_SECONDS = 600
DOWNLOAD_MAX_ATTEMPTS = 8
DOWNLOAD_RETRY_BACKOFF_SECONDS = 30
AGGREGATE_SCHEMA_VERSION = "location_theme_window_v3"

# %%
LABOR_THEME_TERMS = (
    "EMPLOY",
    "EMPLOYMENT",
    "UNEMPLOY",
    "UNEMPLOYMENT",
    "JOB",
    "JOBS",
    "LABOR",
    "LABOUR",
    "WORK",
    "WORKER",
    "WORKERS",
    "WORKFORCE",
    "HIRING",
    "RECRUIT",
    "RECRUITMENT",
    "LAYOFF",
    "LAYOFFS",
    "FIRED",
    "WAGE",
    "WAGES",
    "SALARY",
    "SALARIES",
    "PAY",
    "INCOME",
    "CAREER",
    "OCCUPATION",
    "PROFESSION",
    "STRIKE",
    "UNION",
    "GIG",
    "FREELANCE",
)
UNKNOWN_VALUE = "UNKNOWN"

# %%
GCAM_FEELING_DIMENSIONS = {
    "liwc_anger": "c5.32",
    "liwc_anxiety": "c5.33",
    "lm_uncertainty": "c6.6",
    "wordnet_anger": "c15.15",
    "wordnet_anxiety": "c15.22",
    "wordnet_fear": "c15.111",
    "wordnet_hope": "c15.212",
    "wordnet_optimism": "c15.203",
    "wordnet_feeling": "c17.13",
}
TONE_COLUMNS = [
    "tone",
    "positive_score",
    "negative_score",
    "polarity",
    "activity_reference_density",
    "self_group_reference_density",
    "word_count",
]
AVERAGE_SOURCE_COLUMNS = [
    "location_latitude",
    "location_longitude",
    *TONE_COLUMNS,
    *GCAM_FEELING_DIMENSIONS,
]
AVERAGE_OUTPUT_COLUMNS = [
    "location_latitude_avg",
    "location_longitude_avg",
    *[f"{column}_avg" for column in TONE_COLUMNS],
    *[f"{column}_avg" for column in GCAM_FEELING_DIMENSIONS],
]
AVERAGE_OUTPUT_BY_SOURCE = {
    "location_latitude": "location_latitude_avg",
    "location_longitude": "location_longitude_avg",
    **{column: f"{column}_avg" for column in TONE_COLUMNS},
    **{column: f"{column}_avg" for column in GCAM_FEELING_DIMENSIONS},
}
GROUP_COLUMNS = [
    "location_type",
    "location_name",
    "location_country_code",
    "location_admin1_code",
    "start_date",
    "end_date",
    "theme",
]
DOMINANT_FEELING_COUNT_COLUMNS = [
    "count_anger",
    "count_anxiety",
    "count_fear",
    "count_hope",
    "count_optimism",
    "count_uncertainty",
    "count_unknown",
]
ARCHIVE_SUMMARY_COLUMNS = [
    "url",
    "timestamp",
    "archive_date",
    "stream",
    "status",
    "failure_reason",
    "downloaded_bytes",
    "deleted_bytes",
    "article_count",
    "interaction_count",
    "elapsed_seconds",
]

# %% [markdown]
# # Setup

# %%
sns.set_theme(style="whitegrid", context="talk")
pd.options.display.float_format = "{:,.3f}".format


# %%
def find_project_dir(start_dir: Path) -> Path:
    """Find the Project directory from common launch locations.

    Args:
        start_dir: Directory from which the notebook or script was launched.

    Returns:
        Project directory containing the source and configuration folders.
    """

    resolved_start = start_dir.resolve()
    candidates = [
        resolved_start,
        resolved_start / "Project",
        resolved_start.parent,
    ]
    for candidate in candidates:
        if (candidate / "src").is_dir() and (candidate / "config").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find the Project directory. Run from the repository root, "
        "Project directory, or Project/src directory."
    )


# %%
def current_job_id() -> str:
    """Return a filesystem-safe identifier for the current collection job.

    Returns:
        Filesystem-safe GDELT job ID, Slurm job ID, or ``local``.
    """

    raw_job_id = (
        os.getenv("GDELT_JOB_ID")
        or os.getenv("SLURM_JOB_ID")
        or os.getenv("SLURM_JOBID")
        or "local"
    )
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_job_id)


# %%
def ensure_output_dirs(base_dir: Path) -> tuple[Path, Path, Path]:
    """Create folders used by the loader.

    Args:
        base_dir: Project folder containing this script.

    Returns:
        Data, result, and log directories.
    """

    data_dir = base_dir / "data"
    result_dir = base_dir / "result"
    logs_dir = base_dir / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, result_dir, logs_dir


# %%
def configured_worker_count(archive_count: int | None = None) -> int:
    """Choose a download worker count that respects the current job size.

    Args:
        archive_count: Number of archives to process, when known.

    Returns:
        Positive number of parallel download workers.
    """

    cpu_count = os.cpu_count() or 1
    slurm_cpu_count = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpu_count:
        cpu_count = min(cpu_count, pd.to_numeric(slurm_cpu_count, errors="coerce"))
        if pd.isna(cpu_count):
            cpu_count = os.cpu_count() or 1
        cpu_count = int(cpu_count)

    limits = [MAX_DOWNLOAD_WORKERS, cpu_count]
    if archive_count is not None:
        limits.append(max(1, archive_count))
    return max(1, min(limits))


# %%
base_dir = find_project_dir(Path.cwd())
job_id = current_job_id()
data_dir, result_dir, logs_dir = ensure_output_dirs(base_dir)
raw_data_dir = data_dir / "raw"
aggregate_data_dir = data_dir / "aggregated"
job_data_dir = data_dir / "mid_save"
job_result_dir = result_dir
temporary_archive_dir = raw_data_dir / "temporary_archives" / job_id

for directory in [
    raw_data_dir,
    aggregate_data_dir,
    job_data_dir,
    job_result_dir,
    temporary_archive_dir,
]:
    directory.mkdir(parents=True, exist_ok=True)

print(f"Project directory: {base_dir}")
print(f"Collection job ID: {job_id}")
print(f"Aggregated data directory: {aggregate_data_dir}")

# %% [markdown]
# # Configuration


# %%
def configured_date_range() -> tuple[datetime, datetime, str, str]:
    """Read the configured date range, aggregation period, and save interval.

    Returns:
        Start datetime, end datetime, aggregation period, and save interval.
    """

    start_text = os.getenv("GDELT_DATE_START")
    end_text = os.getenv("GDELT_DATE_END")
    period = os.getenv("GDELT_AGGREGATION_PERIOD", AGGREGATION_PERIOD)
    interval = os.getenv("GDELT_SAVE_INTERVAL", SAVE_INTERVAL)

    start = (
        datetime.strptime(start_text, "%Y-%m-%d")
        if start_text
        else START_DATE
    )
    end = (
        datetime.strptime(end_text, "%Y-%m-%d")
        if end_text
        else END_DATE
    )
    if end < start:
        raise ValueError("END_DATE must be on or after START_DATE.")
    return start, end, period, interval


# %%
def build_time_windows(
    start: datetime,
    end: datetime,
    period: str,
) -> pd.DataFrame:
    """Build inclusive analysis windows for the configured date range.

    Args:
        start: Inclusive start datetime.
        end: Inclusive end datetime.
        period: Pandas-compatible aggregation period or offset string.

    Returns:
        DataFrame with inclusive ``start_date`` and ``end_date`` columns.
    """

    offset = pd.tseries.frequencies.to_offset(period)
    current_start = pd.Timestamp(start).normalize()
    final_exclusive = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
    rows = []

    while current_start < final_exclusive:
        next_start = current_start + offset
        if next_start <= current_start:
            raise ValueError("AGGREGATION_PERIOD must move time forward.")
        current_end_exclusive = min(next_start, final_exclusive)
        rows.append(
            {
                "start_timestamp": current_start,
                "end_timestamp_exclusive": current_end_exclusive,
                "start_date": current_start.date(),
                "end_date": (
                    current_end_exclusive - pd.Timedelta(days=1)
                ).date(),
            }
        )
        current_start = current_end_exclusive

    return pd.DataFrame(rows)


# %%
def query_signature(start: datetime, end: datetime, period: str) -> str:
    """Create a short signature for the date range and archive filter.

    Args:
        start: Inclusive start date.
        end: Inclusive end date.
        period: Aggregation period.

    Returns:
        Stable signature used in metadata.
    """

    value = "|".join(
        [
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            period,
            *GDELT_MASTER_FILE_LIST_URLS.values(),
            *LABOR_THEME_TERMS,
            AGGREGATE_SCHEMA_VERSION,
        ]
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


# %%
def aggregate_output_paths(
    start: datetime,
    end: datetime,
    period: str,
    legacy_name: bool = False,
) -> tuple[Path, Path]:
    """Build final aggregate parquet and metadata paths.

    Args:
        start: Inclusive start date.
        end: Inclusive end date.
        period: Aggregation period.
        legacy_name: Whether to use the old ``gdelt_location_theme`` prefix.

    Returns:
        Parquet output path and JSON metadata path.
    """

    safe_period = re.sub(r"[^A-Za-z0-9_-]+", "_", period)
    date_label = f"{start:%Y%m%d}_{end:%Y%m%d}_{safe_period}"
    prefix = "gdelt_location_theme" if legacy_name else "gdelt"
    output_path = aggregate_data_dir / f"{prefix}_{date_label}.parquet"
    metadata_path = output_path.with_suffix(".metadata.json")
    return output_path, metadata_path


# %%
start_date, end_date, aggregation_period, save_interval = configured_date_range()
safe_aggregation_period = re.sub(
    r"[^A-Za-z0-9_-]+", "_", aggregation_period
)
output_file_stem = (
    f"gdelt_{start_date:%Y%m%d}_{end_date:%Y%m%d}_"
    f"{safe_aggregation_period}_jobid_{job_id}"
)
progress_report_path = job_data_dir / f"{output_file_stem}_progress_report.txt"
time_windows = build_time_windows(
    start_date,
    end_date,
    aggregation_period,
)
aggregate_output_path, aggregate_metadata_path = aggregate_output_paths(
    start_date,
    end_date,
    aggregation_period,
)
legacy_aggregate_output_path, legacy_aggregate_metadata_path = (
    aggregate_output_paths(
        start_date,
        end_date,
        aggregation_period,
        legacy_name=True,
    )
)

print(
    "Configured GDELT range: "
    f"{start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}; "
    f"aggregation period: {aggregation_period}; "
    f"save interval: {save_interval}"
)
print(f"Progress report: {progress_report_path}")
time_windows

# %% [markdown]
# # Progress Logging


# %%
def format_file_size(byte_count: int) -> str:
    """Format bytes using readable binary units.

    Args:
        byte_count: Number of bytes to format.

    Returns:
        Readable file size such as ``1.25 GB``.
    """

    size = float(byte_count)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:,.2f} {unit}"
        size /= 1024
    return f"{size:,.2f} TB"


# %%
def format_duration(seconds: float) -> str:
    """Format elapsed seconds as a human-readable duration.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Readable duration such as ``2 minutes 15.3 seconds``.
    """

    minutes, remaining_seconds = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    parts = []
    if hours:
        parts.append(f"{hours:,} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes:,} minute{'s' if minutes != 1 else ''}")
    parts.append(f"{remaining_seconds:,.1f} seconds")
    return " ".join(parts)


# %%
def report_progress(message: str) -> None:
    """Print a progress message and append it to the daily report file.

    Args:
        message: Human-readable progress message.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_line = f"[{timestamp}] {message}"
    print(report_line, flush=True)
    with progress_report_path.open("a", encoding="utf-8") as file:
        file.write(f"{report_line}\n")


# %%
report_progress(
    "CONFIG: "
    f"START_DATE={start_date:%Y-%m-%d}, "
    f"END_DATE={end_date:%Y-%m-%d}, "
    f"AGGREGATION_PERIOD={aggregation_period}, "
    f"SAVE_INTERVAL={save_interval}, "
    f"OUTPUT={aggregate_output_path}"
)

# %% [markdown]
# # GDELT Archive Index


# %%
def download_master_file_list(url: str, output_path: Path) -> Path:
    """Download and cache GDELT's public archive index.

    Args:
        url: Public GDELT master-file-list URL.
        output_path: Local text file used for the archive index.

    Returns:
        Path to the cached master file list.
    """

    if output_path.exists():
        print(f"Using cached GDELT archive index: {output_path}")
        return output_path

    temporary_path = output_path.with_name(f"{output_path.name}.{job_id}.part")
    try:
        with requests.get(
            url,
            stream=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()
            with temporary_path.open("wb") as file:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        file.write(block)
        temporary_path.replace(output_path)
    except (OSError, requests.RequestException):
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


# %%
def load_gkg_archive_urls(
    master_file_path: Path,
    start: datetime,
    end: datetime,
    stream: str,
) -> pd.DataFrame:
    """Select GKG archive URLs inside the configured date range.

    Args:
        master_file_path: Downloaded GDELT master file list.
        start: Inclusive start date.
        end: Inclusive end date.
        stream: Main or translated GKG stream label.

    Returns:
        Archive timestamps, URLs, sizes, and stream labels.
    """

    rows = []
    end_exclusive = end + timedelta(days=1)
    timestamp_pattern = re.compile(
        r"/(\d{14})(?:\.translation)?\.gkg\.csv\.zip$"
    )

    with master_file_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            if len(parts) != 3 or not parts[2].endswith(".gkg.csv.zip"):
                continue
            match = timestamp_pattern.search(parts[2])
            if not match:
                continue
            timestamp = datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
            if start <= timestamp < end_exclusive:
                rows.append(
                    {
                        "timestamp": timestamp,
                        "archive_date": timestamp.date(),
                        "stream": stream,
                        "compressed_bytes": int(parts[0]),
                        "md5": parts[1],
                        "url": parts[2],
                    }
                )

    columns = [
        "timestamp",
        "archive_date",
        "stream",
        "compressed_bytes",
        "md5",
        "url",
    ]
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["timestamp", "stream"])
        .reset_index(drop=True)
    )


# %%
def load_archive_index(start: datetime, end: datetime) -> pd.DataFrame:
    """Load all GKG archive URLs for the configured range.

    Args:
        start: Inclusive start date.
        end: Inclusive end date.

    Returns:
        Combined main and translation archive index.
    """

    archive_indexes = []
    for stream, url in GDELT_MASTER_FILE_LIST_URLS.items():
        master_file_path = download_master_file_list(
            url,
            raw_data_dir / f"gdelt_{stream}_masterfilelist.txt",
        )
        archive_indexes.append(
            load_gkg_archive_urls(master_file_path, start, end, stream)
        )

    archive_index = pd.concat(archive_indexes, ignore_index=True)
    archive_index = archive_index.sort_values(
        ["timestamp", "stream"]
    ).reset_index(drop=True)
    if archive_index.empty:
        raise ValueError("No GKG archives matched the configured date range.")
    return archive_index


# %% [markdown]
# # GKG Parsing Helpers

# %%
GKG_COLUMNS = (
    "gkg_record_id",
    "gdelt_date",
    "source_collection_identifier",
    "source_name",
    "article_url",
    "counts",
    "v2_counts",
    "themes",
    "v2_themes",
    "locations",
    "v2_locations",
    "persons",
    "v2_persons",
    "organizations",
    "v2_organizations",
    "v2_tone",
    "enhanced_dates",
    "gcam",
    "sharing_image",
    "related_images",
    "social_image_embeds",
    "social_video_embeds",
    "quotations",
    "all_names",
    "amounts",
    "translation_info",
    "extras_xml",
)
GKG_FIELD_INDEX = {name: index for index, name in enumerate(GKG_COLUMNS)}


# %%
def parse_gdelt_date(value: object) -> pd.Timestamp:
    """Parse a GDELT 14-digit timestamp.

    Args:
        value: GDELT date value from the public GKG archive.

    Returns:
        Parsed pandas timestamp or ``NaT``.
    """

    text = str(value).split(".")[0]
    return pd.to_datetime(text, format="%Y%m%d%H%M%S", errors="coerce")


# %%
def parse_v2_tone(value: object) -> dict[str, float]:
    """Parse GDELT's seven built-in V2Tone dimensions.

    Args:
        value: Comma-delimited V2Tone string.

    Returns:
        Mapping from readable tone names to numeric values.
    """

    values = str(value).split(",") if pd.notna(value) else []
    parsed = pd.to_numeric(pd.Series(values), errors="coerce").tolist()
    parsed.extend([float("nan")] * (len(TONE_COLUMNS) - len(parsed)))
    return dict(zip(TONE_COLUMNS, parsed[: len(TONE_COLUMNS)]))


# %%
def parse_locations(value: object) -> list[dict]:
    """Parse all GDELT V2Locations entries.

    Args:
        value: Semicolon-delimited V2Locations string.

    Returns:
        List of structured location dictionaries.
    """

    locations = []
    if pd.isna(value) or str(value) == "":
        return locations

    for item in str(value).split(";"):
        fields = item.split("#")
        if len(fields) < 7:
            continue
        latitude = pd.to_numeric(fields[4], errors="coerce")
        longitude = pd.to_numeric(fields[5], errors="coerce")
        locations.append(
            {
                "location_name": fields[1] or UNKNOWN_VALUE,
                "location_type": fields[0] or UNKNOWN_VALUE,
                "location_country_code": fields[2] or UNKNOWN_VALUE,
                "location_admin1_code": fields[3] or UNKNOWN_VALUE,
                "location_latitude": (
                    float(latitude) if pd.notna(latitude) else float("nan")
                ),
                "location_longitude": (
                    float(longitude) if pd.notna(longitude) else float("nan")
                ),
            }
        )
    return locations


# %%
def unknown_location() -> dict:
    """Return the fallback location used for records with no location.

    Returns:
        Structured location dictionary with UNKNOWN labels.
    """

    return {
        "location_name": UNKNOWN_VALUE,
        "location_type": UNKNOWN_VALUE,
        "location_country_code": UNKNOWN_VALUE,
        "location_admin1_code": UNKNOWN_VALUE,
        "location_latitude": float("nan"),
        "location_longitude": float("nan"),
    }


# %%
def parse_gcam(value: object) -> dict[str, float]:
    """Parse selected GCAM feeling dimensions.

    Args:
        value: Comma-delimited GCAM dimension/value pairs.

    Returns:
        Mapping from readable feeling names to numeric values.
    """

    raw_values = {}
    if pd.notna(value):
        for item in str(value).split(","):
            dimension, separator, score = item.partition(":")
            if separator:
                raw_values[dimension] = pd.to_numeric(score, errors="coerce")

    return {
        name: raw_values.get(dimension, float("nan"))
        for name, dimension in GCAM_FEELING_DIMENSIONS.items()
    }


# %%
def dominant_feeling(feelings: dict[str, float]) -> str:
    """Choose the strongest available selected GCAM feeling family.

    Args:
        feelings: Named GCAM feeling scores.

    Returns:
        Feeling family label or ``unknown``.
    """

    feeling_groups = {
        "anger": ["liwc_anger", "wordnet_anger"],
        "anxiety": ["liwc_anxiety", "wordnet_anxiety"],
        "fear": ["wordnet_fear"],
        "hope": ["wordnet_hope"],
        "optimism": ["wordnet_optimism"],
        "uncertainty": ["lm_uncertainty"],
    }
    group_scores = {}
    for group, columns in feeling_groups.items():
        values = [
            feelings[column]
            for column in columns
            if pd.notna(feelings.get(column)) and feelings[column] > 0
        ]
        if values:
            group_scores[group] = max(values)
    return max(group_scores, key=group_scores.get) if group_scores else "unknown"


# %%
def extract_matching_themes(v2_themes: object) -> list[str]:
    """Extract configured labor-market theme terms from GDELT V2Themes.

    Args:
        v2_themes: Semicolon-delimited GDELT V2Themes value.

    Returns:
        Sorted matching configured terms or ``UNKNOWN``.
    """

    if pd.isna(v2_themes):
        return [UNKNOWN_VALUE]

    upper_themes = str(v2_themes).upper()
    matches = sorted({term for term in LABOR_THEME_TERMS if term in upper_themes})
    return matches or [UNKNOWN_VALUE]


# %%
def is_labor_market_record(v2_themes: object) -> bool:
    """Check whether GDELT themes identify a labor-market article.

    Args:
        v2_themes: Raw GDELT V2Themes value.

    Returns:
        True when at least one configured labor-market term is present.
    """

    return extract_matching_themes(v2_themes) != [UNKNOWN_VALUE]


# %%
def find_time_window(
    timestamp: pd.Timestamp,
    windows: pd.DataFrame,
) -> dict | None:
    """Find the configured time window containing a timestamp.

    Args:
        timestamp: Parsed article timestamp.
        windows: DataFrame from ``build_time_windows``.

    Returns:
        Time-window row dictionary, or None when outside the range.
    """

    if pd.isna(timestamp):
        return None
    mask = (
        (windows["start_timestamp"] <= timestamp)
        & (timestamp < windows["end_timestamp_exclusive"])
    )
    if not mask.any():
        return None
    return windows.loc[mask].iloc[0].to_dict()


# %%
def record_to_interactions(
    fields: list[str],
    windows: pd.DataFrame,
) -> list[dict]:
    """Convert one matching GKG row into interaction rows.

    Args:
        fields: Raw tab-split GKG fields.
        windows: Configured time windows.

    Returns:
        Interaction rows for all locations and matching themes.
    """

    gdelt_timestamp = parse_gdelt_date(fields[GKG_FIELD_INDEX["gdelt_date"]])
    window = find_time_window(gdelt_timestamp, windows)
    if window is None:
        return []

    gkg_record_id = fields[GKG_FIELD_INDEX["gkg_record_id"]]
    v2_themes = fields[GKG_FIELD_INDEX["v2_themes"]]
    locations = parse_locations(fields[GKG_FIELD_INDEX["v2_locations"]])
    if not locations:
        locations = [unknown_location()]

    tone = parse_v2_tone(fields[GKG_FIELD_INDEX["v2_tone"]])
    feelings = parse_gcam(fields[GKG_FIELD_INDEX["gcam"]])
    feeling = dominant_feeling(feelings)
    themes = extract_matching_themes(v2_themes)

    interactions = []
    for location in locations:
        for theme in themes:
            interactions.append(
                {
                    "gkg_record_id": gkg_record_id,
                    "start_date": window["start_date"],
                    "end_date": window["end_date"],
                    "theme": theme,
                    "dominant_feeling": feeling,
                    **location,
                    **tone,
                    **feelings,
                }
            )
    return interactions


# %% [markdown]
# # Archive Processing


# %%
def download_archive_file(
    url: str,
    output_path: Path,
    expected_md5: str,
) -> int:
    """Download one public GDELT archive to a temporary local path.

    Args:
        url: Public GDELT ZIP URL.
        output_path: Temporary ZIP destination.
        expected_md5: MD5 checksum from GDELT's master file list.

    Returns:
        Number of bytes written to disk.
    """

    for attempt in range(1, DOWNLOAD_MAX_ATTEMPTS + 1):
        bytes_written = 0
        md5_hash = hashlib.md5(usedforsecurity=False)
        try:
            with requests.get(
                url,
                stream=True,
                timeout=REQUEST_TIMEOUT_SECONDS,
            ) as response:
                response.raise_for_status()
                with output_path.open("wb") as file:
                    for block in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if block:
                            bytes_written += len(block)
                            md5_hash.update(block)
                            file.write(block)
            if md5_hash.hexdigest().lower() != expected_md5.lower():
                raise ValueError(f"MD5 checksum mismatch for {Path(url).name}.")
            return bytes_written
        except (OSError, requests.RequestException, ValueError) as error:
            output_path.unlink(missing_ok=True)
            if attempt == DOWNLOAD_MAX_ATTEMPTS:
                raise
            wait_seconds = DOWNLOAD_RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1)
            print(
                f"\nDownload failed for {Path(url).name}: {error}; retrying in "
                f"{wait_seconds:,} seconds "
                f"({attempt:,}/{DOWNLOAD_MAX_ATTEMPTS:,}).",
                flush=True,
            )
            time.sleep(wait_seconds)
    return 0


# %%
def archive_zip_path(url: str) -> Path:
    """Build a job-safe temporary ZIP path for one archive URL.

    Args:
        url: Public GDELT ZIP URL.

    Returns:
        Temporary archive path unique to this process and URL.
    """

    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    archive_name = Path(urlparse(url).path).name
    return temporary_archive_dir / f"{os.getpid()}_{digest}_{archive_name}"


# %%
def aggregate_interactions(interactions: list[dict]) -> pd.DataFrame:
    """Aggregate interaction rows into the final grouped schema.

    Args:
        interactions: One row per article-location-theme interaction.

    Returns:
        Aggregated dataframe.
    """

    if not interactions:
        return empty_aggregate_frame()

    data = pd.DataFrame(interactions)
    for column in AVERAGE_SOURCE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    feeling_counts = pd.get_dummies(data["dominant_feeling"], prefix="count")
    for column in DOMINANT_FEELING_COUNT_COLUMNS:
        if column not in feeling_counts:
            feeling_counts[column] = 0

    data_with_counts = pd.concat(
        [data.drop(columns=["dominant_feeling"]), feeling_counts],
        axis=1,
    )
    grouped_rows = data_with_counts.groupby(GROUP_COLUMNS, dropna=False)
    aggregations = {
        "gkg_record_id": pd.Series.nunique,
        **{column: "sum" for column in DOMINANT_FEELING_COUNT_COLUMNS},
    }
    grouped = (
        grouped_rows.agg(aggregations)
        .reset_index()
        .rename(columns={"gkg_record_id": "article_count"})
    )
    grouped["interaction_count"] = grouped_rows.size().values
    for source_column, output_column in AVERAGE_OUTPUT_BY_SOURCE.items():
        grouped[output_column] = grouped_rows[source_column].mean().values
        grouped[f"__{output_column}_count"] = (
            grouped_rows[source_column].count().values
        )
    return set_aggregate_column_order(grouped, keep_internal=True)


# %%
def empty_aggregate_frame() -> pd.DataFrame:
    """Create an empty aggregate dataframe with the final columns.

    Returns:
        Empty dataframe with stable output columns.
    """

    columns = [
        *GROUP_COLUMNS,
        "article_count",
        "interaction_count",
        *AVERAGE_OUTPUT_COLUMNS,
        *DOMINANT_FEELING_COUNT_COLUMNS,
    ]
    return pd.DataFrame(columns=columns)


# %%
def set_aggregate_column_order(
    data: pd.DataFrame,
    keep_internal: bool = False,
) -> pd.DataFrame:
    """Set stable column order for aggregate output.

    Args:
        data: Aggregate dataframe.
        keep_internal: Whether to keep helper columns for later combining.

    Returns:
        Aggregate dataframe with expected columns.
    """

    ordered_columns = empty_aggregate_frame().columns.tolist()
    for column in ordered_columns:
        if column not in data:
            data[column] = 0 if column.startswith("count_") else pd.NA
    internal_columns = [
        column for column in data.columns if keep_internal and column.startswith("__")
    ]
    return data[ordered_columns + internal_columns]


# %%
def parse_archive_interactions(
    zip_path: Path,
    windows: pd.DataFrame,
) -> tuple[list[dict], int]:
    """Extract relevant interactions from one downloaded GKG ZIP archive.

    Args:
        zip_path: Downloaded GKG ZIP archive.
        windows: Configured time windows.

    Returns:
        Interaction rows and matching article count.
    """

    interactions = []
    article_ids = set()
    with zipfile.ZipFile(zip_path) as archive:
        csv_names = [
            name
            for name in archive.namelist()
            if not name.endswith("/") and name.endswith(".csv")
        ]
        if not csv_names:
            raise ValueError(f"No CSV file found in {zip_path.name}.")
        csv_name = csv_names[0]
        with archive.open(csv_name) as file:
            for byte_line in file:
                fields = byte_line.decode("utf-8", errors="replace").rstrip(
                    "\n"
                ).split("\t", maxsplit=len(GKG_COLUMNS) - 1)
                if len(fields) != len(GKG_COLUMNS):
                    continue

                v2_themes = fields[GKG_FIELD_INDEX["v2_themes"]]
                if not is_labor_market_record(v2_themes):
                    continue

                row_interactions = record_to_interactions(fields, windows)
                if row_interactions:
                    article_ids.add(fields[GKG_FIELD_INDEX["gkg_record_id"]])
                    interactions.extend(row_interactions)

    return interactions, len(article_ids)


# %%
def process_archive(
    archive: pd.Series,
    windows: pd.DataFrame,
) -> dict:
    """Download, parse, aggregate, and delete one GDELT archive.

    Args:
        archive: One archive-index row.
        windows: Configured time windows.

    Returns:
        Processing summary and aggregate dataframe.
    """

    zip_path = archive_zip_path(archive["url"])
    downloaded_bytes = 0
    deleted_bytes = 0
    interactions = []
    article_count = 0
    aggregate = empty_aggregate_frame()
    start_time = time.perf_counter()

    try:
        downloaded_bytes = download_archive_file(
            archive["url"],
            zip_path,
            archive["md5"],
        )
    except (OSError, requests.RequestException, ValueError) as error:
        return {
            "url": archive["url"],
            "timestamp": archive["timestamp"],
            "archive_date": archive["archive_date"],
            "stream": archive["stream"],
            "status": "download_failed",
            "failure_reason": f"{type(error).__name__}: {error}",
            "downloaded_bytes": 0,
            "deleted_bytes": 0,
            "article_count": 0,
            "interaction_count": 0,
            "elapsed_seconds": time.perf_counter() - start_time,
            "aggregate": aggregate,
        }

    try:
        interactions, article_count = parse_archive_interactions(
            zip_path,
            windows,
        )
        deleted_bytes = zip_path.stat().st_size if zip_path.exists() else 0
        aggregate = aggregate_interactions(interactions)
    finally:
        zip_path.unlink(missing_ok=True)

    return {
        "url": archive["url"],
        "timestamp": archive["timestamp"],
        "archive_date": archive["archive_date"],
        "stream": archive["stream"],
        "status": "completed",
        "failure_reason": "",
        "downloaded_bytes": downloaded_bytes,
        "deleted_bytes": deleted_bytes,
        "article_count": article_count,
        "interaction_count": len(interactions),
        "elapsed_seconds": time.perf_counter() - start_time,
        "aggregate": aggregate,
    }


# %% [markdown]
# # Final Aggregation


# %%
def combine_aggregates(
    aggregates: list[pd.DataFrame],
    keep_internal: bool = False,
) -> pd.DataFrame:
    """Combine archive-level aggregates into one final aggregate dataframe.

    Args:
        aggregates: Archive-level aggregate dataframes.
        keep_internal: Whether to retain counts needed for exact checkpoint resume.

    Returns:
        Final grouped aggregate dataframe.
    """

    nonempty = [frame for frame in aggregates if not frame.empty]
    if not nonempty:
        return empty_aggregate_frame()

    data = pd.concat(nonempty, ignore_index=True)
    sum_columns = [
        "article_count",
        "interaction_count",
        *DOMINANT_FEELING_COUNT_COLUMNS,
    ]
    # Weighted means preserve archive-level averages without storing all rows.
    for column in AVERAGE_OUTPUT_COLUMNS:
        count_column = f"__{column}_count"
        if count_column not in data:
            data[count_column] = data["interaction_count"]
        data[f"{column}_weighted"] = (
            pd.to_numeric(data[column], errors="coerce")
            * data[count_column]
        )

    aggregations = {
        **{column: "sum" for column in sum_columns},
        **{
            f"__{column}_count": "sum"
            for column in AVERAGE_OUTPUT_COLUMNS
        },
        **{
            f"{column}_weighted": "sum"
            for column in AVERAGE_OUTPUT_COLUMNS
        },
    }
    combined = (
        data.groupby(GROUP_COLUMNS, dropna=False)
        .agg(aggregations)
        .reset_index()
    )
    for column in AVERAGE_OUTPUT_COLUMNS:
        count_column = f"__{column}_count"
        combined[column] = (
            combined[f"{column}_weighted"]
            / combined[count_column].replace(0, pd.NA)
        )
        combined = combined.drop(columns=f"{column}_weighted")
        if not keep_internal:
            combined = combined.drop(columns=count_column)

    return set_aggregate_column_order(combined, keep_internal=keep_internal)


# %%
def write_parquet_atomically(data: pd.DataFrame, output_path: Path) -> None:
    """Write a parquet file through a temporary path before replacing.

    Args:
        data: Dataframe to write.
        output_path: Final parquet path.
    """

    temporary_path = output_path.with_suffix(".parquet.part")
    data.to_parquet(temporary_path, index=False, engine="pyarrow")
    temporary_path.replace(output_path)


# %%
def write_json_atomically(data: dict, output_path: Path) -> None:
    """Write a JSON file through a temporary path before replacing.

    Args:
        data: Dictionary to save.
        output_path: Final JSON path.
    """

    temporary_path = output_path.with_suffix(".json.part")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)
    temporary_path.replace(output_path)


# %%
def write_text_atomically(text: str, output_path: Path) -> None:
    """Write a text file through a temporary path before replacing.

    Args:
        text: Text content to save.
        output_path: Final text path.
    """

    temporary_path = output_path.with_suffix(f"{output_path.suffix}.part")
    with temporary_path.open("w", encoding="utf-8") as file:
        file.write(text)
    temporary_path.replace(output_path)


# %%
def next_save_timestamp(
    current_timestamp: pd.Timestamp,
    interval: str,
) -> pd.Timestamp:
    """Move a timestamp forward by one configured save interval.

    Args:
        current_timestamp: Current checkpoint boundary.
        interval: Pandas-compatible duration or offset string.

    Returns:
        Next checkpoint boundary.
    """

    offset = pd.tseries.frequencies.to_offset(interval)
    next_timestamp = current_timestamp + offset
    if next_timestamp <= current_timestamp:
        raise ValueError("SAVE_INTERVAL must move time forward.")
    return next_timestamp


# %%
def write_mid_save_summary(
    output_path: Path,
    summary_path: Path,
    text_path: Path,
    aggregate: pd.DataFrame,
    summary: pd.DataFrame,
    save_boundary: pd.Timestamp,
    latest_archive_timestamp: pd.Timestamp,
    elapsed_seconds: float,
) -> None:
    """Save a readable checkpoint summary next to checkpoint data files.

    Args:
        output_path: Checkpoint aggregate parquet path.
        summary_path: Checkpoint archive-summary parquet path.
        text_path: Checkpoint text summary path.
        aggregate: Current aggregate dataframe.
        summary: Current one-row-per-archive summary dataframe.
        save_boundary: Dataset-time boundary that triggered this save.
        latest_archive_timestamp: Latest archive timestamp completed so far.
        elapsed_seconds: Seconds elapsed since the job started.
    """

    downloaded_bytes = int(summary["downloaded_bytes"].sum())
    article_count = int(summary["article_count"].sum())
    table_columns = "\n".join(f"- {column}" for column in aggregate.columns)
    text = (
        "GDELT aggregation mid-save summary\n"
        "Contents: all archives processed so far\n"
        f"Save boundary: {save_boundary:%Y-%m-%d %H:%M:%S}\n"
        f"Latest completed archive: {latest_archive_timestamp:%Y-%m-%d %H:%M:%S}\n"
        f"Saved at: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"Elapsed time: {format_duration(elapsed_seconds)}\n"
        f"Aggregate path: {output_path}\n"
        f"Archive summary path: {summary_path}\n"
        f"Downloaded data: {format_file_size(downloaded_bytes)}\n"
        f"Processed articles: {article_count:,}\n"
        f"Aggregate rows: {len(aggregate):,}\n"
        f"Archive files processed: {len(summary):,}\n"
        "Table columns:\n"
        f"{table_columns}\n"
    )
    write_text_atomically(text, text_path)


# %%
def save_mid_process_results(
    aggregates: list[pd.DataFrame],
    archive_summaries: list[dict],
    checkpoint_index: int,
    save_boundary: pd.Timestamp,
    latest_archive_timestamp: pd.Timestamp,
    elapsed_seconds: float,
) -> None:
    """Save current aggregate results during a long GDELT run.

    Args:
        aggregates: Archive-level aggregates completed so far.
        archive_summaries: Archive-level processing summaries completed so far.
        checkpoint_index: One-based checkpoint counter.
        save_boundary: Dataset-time boundary that triggered this save.
        latest_archive_timestamp: Latest archive timestamp completed so far.
        elapsed_seconds: Seconds elapsed since the job started.
    """

    if not archive_summaries:
        return

    current_aggregate = combine_aggregates(aggregates, keep_internal=True)
    current_summary = pd.DataFrame(archive_summaries).sort_values(
        ["timestamp", "stream"]
    )
    output_path = job_data_dir / (
        f"{output_file_stem}_checkpoint_{checkpoint_index:03d}.parquet"
    )
    summary_path = (
        job_data_dir
        / f"{output_file_stem}_checkpoint_{checkpoint_index:03d}_summary.parquet"
    )
    text_path = job_data_dir / f"{output_file_stem}_mid_save_summary.txt"

    write_parquet_atomically(current_summary, summary_path)
    write_parquet_atomically(current_aggregate, output_path)
    write_mid_save_summary(
        output_path,
        summary_path,
        text_path,
        current_aggregate,
        current_summary,
        save_boundary,
        latest_archive_timestamp,
        elapsed_seconds,
    )
    downloaded_text = format_file_size(
        int(current_summary["downloaded_bytes"].sum())
    )
    report_progress(
        "MID SAVE: "
        f"checkpoint={checkpoint_index:,}, "
        f"boundary={save_boundary:%Y-%m-%d %H:%M:%S}, "
        f"latest_archive={latest_archive_timestamp:%Y-%m-%d %H:%M:%S}, "
        f"archives={len(current_summary):,}, "
        f"records={int(current_summary['article_count'].sum()):,}, "
        f"downloaded={downloaded_text}, "
        f"output={output_path}."
    )


# %%
def output_is_complete(
    output_path: Path,
    metadata_path: Path,
    signature: str,
) -> bool:
    """Check whether this aggregate output already exists and is complete.

    Args:
        output_path: Aggregate parquet path.
        metadata_path: Aggregate metadata path.
        signature: Expected query signature.

    Returns:
        True when metadata says the output is complete for this query.
    """

    if not output_path.exists() or not metadata_path.exists():
        return False
    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, ValueError):
        return False
    return (
        metadata.get("complete") is True
        and metadata.get("query_signature") == signature
        and metadata.get("schema_version") == AGGREGATE_SCHEMA_VERSION
    )


# %%
def completed_output_paths(signature: str) -> tuple[Path, Path] | None:
    """Find a complete output using either the new or legacy filename.

    Args:
        signature: Expected query signature.

    Returns:
        Complete aggregate and metadata paths, or ``None`` if neither exists.
    """

    candidates = [
        (aggregate_output_path, aggregate_metadata_path),
        (legacy_aggregate_output_path, legacy_aggregate_metadata_path),
    ]
    for output_path, metadata_path in candidates:
        if output_is_complete(output_path, metadata_path, signature):
            return output_path, metadata_path
    return None


# %%
def completed_summary_path(metadata_path: Path) -> Path:
    """Read the saved archive-summary path from aggregate metadata.

    Args:
        metadata_path: Aggregate metadata path.

    Returns:
        Summary parquet path from metadata or the current job default.
    """

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    summary_path = metadata.get("summary_output_path")
    if summary_path:
        configured_path = Path(summary_path)
        if configured_path.exists():
            return configured_path

        relocated_path = job_data_dir / configured_path.name
        if relocated_path.exists():
            return relocated_path
        return configured_path
    return job_data_dir / f"{output_file_stem}_summary.parquet"


# %%
def empty_archive_summary() -> pd.DataFrame:
    """Create an empty archive summary with the expected columns.

    Returns:
        Empty archive-summary dataframe.
    """

    return pd.DataFrame(columns=ARCHIVE_SUMMARY_COLUMNS)


# %%
def acquire_output_lock(output_path: Path) -> tuple[object, Path]:
    """Acquire a non-blocking file lock for final output writes.

    Args:
        output_path: Final parquet output path.

    Returns:
        Open lock file and its path.

    Raises:
        RuntimeError: If another process currently holds the lock.
    """

    lock_directory = output_path.parent / "locks"
    lock_directory.mkdir(parents=True, exist_ok=True)
    lock_path = lock_directory / output_path.with_suffix(".lock").name
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_file.close()
        raise RuntimeError(
            f"Another process is currently writing {output_path}."
        ) from error
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"pid={os.getpid()}\nacquired={datetime.now().isoformat()}\n")
    lock_file.flush()
    return lock_file, lock_path


# %%
def release_output_lock(lock_file: object, lock_path: Path) -> None:
    """Release and delete a lock file created by ``acquire_output_lock``.

    Args:
        lock_file: Open file that owns the advisory lock.
        lock_path: Lock file path to delete.
    """

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()
    lock_path.unlink(missing_ok=True)


# %%
def delete_temporary_archives() -> None:
    """Delete temporary archive files created by the current job."""

    shutil.rmtree(temporary_archive_dir, ignore_errors=True)
    report_progress(
        f"CLEANUP: deleted temporary archive directory {temporary_archive_dir}."
    )


# %%
def parse_checkpoint_filename(path: Path) -> dict | None:
    """Parse query details stored in a checkpoint filename.

    Args:
        path: Aggregate checkpoint parquet path.

    Returns:
        Parsed checkpoint details, or ``None`` when the filename does not use
        the expected checkpoint naming convention.
    """

    match = re.fullmatch(
        r"gdelt_(\d{8})_(\d{8})_(.+?)_jobid_(.+)_checkpoint_(\d+)",
        path.stem,
    )
    if not match:
        return None
    try:
        checkpoint_start = datetime.strptime(match.group(1), "%Y%m%d")
        checkpoint_end = datetime.strptime(match.group(2), "%Y%m%d")
    except ValueError:
        return None
    return {
        "start_date": checkpoint_start,
        "end_date": checkpoint_end,
        "aggregation_period": match.group(3),
        "job_id": match.group(4),
        "checkpoint_index": int(match.group(5)),
    }


# %%
def load_latest_checkpoint() -> tuple[list[pd.DataFrame], list[dict], int]:
    """Load the best valid checkpoint contained in the configured query.

    Checkpoints from previous job IDs are considered so a restarted cluster job
    can continue work performed by an earlier job. A checkpoint may cover a
    smaller date range than the current query, provided its aggregation period
    matches.

    Returns:
        Initial aggregates, archive summaries, and last checkpoint index. Empty
        values are returned when no complete checkpoint pair exists.
    """

    candidates = []
    pattern = "gdelt_*_checkpoint_*.parquet"
    for aggregate_path in job_data_dir.glob(pattern):
        if aggregate_path.stem.endswith("_summary"):
            continue
        details = parse_checkpoint_filename(aggregate_path)
        if details is None:
            report_progress(
                "CHECKPOINT SKIP: filename does not match the expected format: "
                f"{aggregate_path.name}."
            )
            continue
        if details["aggregation_period"] != safe_aggregation_period:
            report_progress(
                "CHECKPOINT SKIP: aggregation period does not match for "
                f"{aggregate_path.name}."
            )
            continue
        if not (
            start_date <= details["start_date"]
            and details["end_date"] <= end_date
        ):
            report_progress(
                "CHECKPOINT SKIP: date range is not fully contained in the "
                f"configured range for {aggregate_path.name}."
            )
            continue
        summary_path = aggregate_path.with_name(
            f"{aggregate_path.stem}_summary.parquet"
        )
        if not summary_path.exists():
            report_progress(
                "CHECKPOINT SKIP: summary file is missing for "
                f"{aggregate_path.name}."
            )
            continue
        try:
            aggregate = pd.read_parquet(aggregate_path)
            summary = pd.read_parquet(summary_path)
        except (OSError, TypeError, ValueError) as error:
            report_progress(
                "CHECKPOINT SKIP: could not read checkpoint pair "
                f"{aggregate_path.name}: {error}."
            )
            continue
        if "url" not in summary.columns:
            report_progress(
                "CHECKPOINT SKIP: summary has no URL column for "
                f"{aggregate_path.name}."
            )
            continue
        if "timestamp" in summary.columns:
            latest_timestamp = pd.to_datetime(
                summary["timestamp"], errors="coerce"
            ).max()
        else:
            latest_timestamp = pd.NaT
        if pd.isna(latest_timestamp):
            latest_timestamp = pd.Timestamp.min
        candidates.append(
            (
                latest_timestamp,
                aggregate_path.stat().st_mtime,
                details["checkpoint_index"],
                aggregate_path,
                aggregate,
                summary,
                details,
            )
        )

    if candidates:
        (
            _,
            _,
            checkpoint_index,
            aggregate_path,
            aggregate,
            summary,
            details,
        ) = max(candidates, key=lambda candidate: candidate[:3])
        exact_query = (
            details["start_date"] == start_date
            and details["end_date"] == end_date
        )
        next_checkpoint_index = checkpoint_index if exact_query else 0
        report_progress(
            "RESUME: loaded contained checkpoint "
            f"{aggregate_path.name} with {len(summary):,} completed archives; "
            f"current checkpoint index starts at {next_checkpoint_index:,}."
        )
        return (
            [aggregate],
            summary.to_dict("records"),
            next_checkpoint_index,
        )

    return [], [], 0


# %%
def run_gdelt_aggregation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download GDELT archives and create location/theme aggregates.

    Returns:
        Final aggregate dataframe and one-row-per-archive summary dataframe.
    """

    signature = query_signature(start_date, end_date, aggregation_period)
    completed_paths = completed_output_paths(signature)
    if completed_paths is not None:
        completed_output_path, completed_metadata_path = completed_paths
        try:
            aggregate = pd.read_parquet(completed_output_path)
        except (OSError, ValueError) as error:
            report_progress(
                "COMPLETE OUTPUT SKIP: aggregate is unreadable for "
                f"{completed_output_path}: {error}. "
                "Starting or resuming the download instead."
            )
        else:
            summary_path = completed_summary_path(completed_metadata_path)
            try:
                summary = pd.read_parquet(summary_path)
            except (OSError, ValueError):
                summary = empty_archive_summary()
                report_progress(
                    "SUMMARY MISSING: reusing the complete aggregate without "
                    f"archive-level summary data because {summary_path} is "
                    "missing or unreadable."
                )
            report_progress(
                "SKIP: aggregate output already complete at "
                f"{completed_output_path}"
            )
            delete_temporary_archives()
            return aggregate, summary

    next_checkpoint_timestamp = next_save_timestamp(
        pd.Timestamp(start_date).normalize(),
        save_interval,
    )
    lock_file, lock_path = acquire_output_lock(aggregate_output_path)
    run_start_time = time.perf_counter()
    try:
        aggregates, archive_summaries, checkpoint_index = load_latest_checkpoint()
    except Exception:
        try:
            release_output_lock(lock_file, lock_path)
        finally:
            delete_temporary_archives()
        raise
    daily_counts = Counter()
    completed_urls = {summary["url"] for summary in archive_summaries}
    latest_archive_timestamp = max(
        (pd.Timestamp(summary["timestamp"]) for summary in archive_summaries),
        default=pd.Timestamp(start_date).normalize(),
    )
    while latest_archive_timestamp >= next_checkpoint_timestamp:
        next_checkpoint_timestamp = next_save_timestamp(
            next_checkpoint_timestamp,
            save_interval,
        )

    try:
        archive_index = load_archive_index(start_date, end_date)
        report_progress(
            "DOWNLOAD START: "
            f"{len(archive_index):,} archives from "
            f"{start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}."
        )

        archive_records = [
            archive
            for _, archive in archive_index.iterrows()
            if archive["url"] not in completed_urls
        ]
        worker_count = configured_worker_count(len(archive_records))
        report_progress(f"WORKERS: using {worker_count:,} parallel workers.")
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(process_archive, archive, time_windows): archive
                for archive in archive_records
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing GDELT archives",
            ):
                result = future.result()
                aggregate = result.pop("aggregate")
                aggregates.append(aggregate)
                archive_summaries.append(result)

                archive_date = result["archive_date"]
                archive_timestamp = pd.Timestamp(result["timestamp"])
                latest_archive_timestamp = max(
                    latest_archive_timestamp,
                    archive_timestamp,
                )
                daily_counts[archive_date] += 1
                if result["status"] == "download_failed":
                    report_progress(
                        "ARCHIVE FAILED: "
                        f"date={archive_date}, "
                        f"url={result['url']}, "
                        f"attempts={DOWNLOAD_MAX_ATTEMPTS:,}, "
                        f"elapsed={format_duration(result['elapsed_seconds'])}, "
                        f"reason={result['failure_reason']}. "
                        "Continuing with the next archive."
                    )
                if (
                    daily_counts[archive_date] == 1
                    or daily_counts[archive_date] % 25 == 0
                ):
                    report_progress(
                        "ARCHIVE PROGRESS: "
                        f"date={archive_date}, "
                        f"completed_for_date={daily_counts[archive_date]:,}, "
                        f"records={result['article_count']:,}, "
                        f"interactions={result['interaction_count']:,}, "
                        f"downloaded={format_file_size(result['downloaded_bytes'])}, "
                        f"deleted={format_file_size(result['deleted_bytes'])}."
                    )

                if latest_archive_timestamp >= next_checkpoint_timestamp:
                    checkpoint_index += 1
                    save_mid_process_results(
                        aggregates,
                        archive_summaries,
                        checkpoint_index,
                        next_checkpoint_timestamp,
                        latest_archive_timestamp,
                        time.perf_counter() - run_start_time,
                    )
                    while latest_archive_timestamp >= next_checkpoint_timestamp:
                        next_checkpoint_timestamp = next_save_timestamp(
                            next_checkpoint_timestamp,
                            save_interval,
                        )

        final_aggregate = combine_aggregates(aggregates)
        summary = pd.DataFrame(archive_summaries).sort_values(
            ["timestamp", "stream"]
        )
        summary_output_path = job_data_dir / f"{output_file_stem}_summary.parquet"
        write_parquet_atomically(summary, summary_output_path)
        write_parquet_atomically(final_aggregate, aggregate_output_path)

        metadata = {
            "complete": True,
            "schema_version": AGGREGATE_SCHEMA_VERSION,
            "query_signature": signature,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "aggregation_period": aggregation_period,
            "save_interval": save_interval,
            "archive_count": len(summary),
            "failed_archive_count": int(
                summary["status"].eq("download_failed").sum()
            ),
            "article_count": int(summary["article_count"].sum()),
            "interaction_count": int(summary["interaction_count"].sum()),
            "compressed_bytes_downloaded": int(
                summary["downloaded_bytes"].sum()
            ),
            "deleted_zip_bytes": int(summary["deleted_bytes"].sum()),
            "output_path": str(aggregate_output_path),
            "summary_output_path": str(summary_output_path),
        }
        write_json_atomically(metadata, aggregate_metadata_path)

        elapsed = format_duration(time.perf_counter() - run_start_time)
        report_progress(
            "DOWNLOAD FINISH: "
            f"archives={len(summary):,}, "
            f"failed_archives={metadata['failed_archive_count']:,}, "
            f"records={metadata['article_count']:,}, "
            f"interactions={metadata['interaction_count']:,}, "
            f"downloaded={format_file_size(metadata['compressed_bytes_downloaded'])}, "
            f"deleted={format_file_size(metadata['deleted_zip_bytes'])}, "
            f"output={aggregate_output_path}, "
            f"time={elapsed}."
        )
    finally:
        try:
            release_output_lock(lock_file, lock_path)
        finally:
            delete_temporary_archives()

    return final_aggregate, summary


# %%
aggregate_data, aggregate_summary = run_gdelt_aggregation()
aggregate_data

# %% [markdown]
# # Summary

# %%
aggregate_summary

# %%
summary_by_date = (
    aggregate_summary.groupby("archive_date")
    .agg(
        archive_count=("url", "count"),
        article_count=("article_count", "sum"),
        interaction_count=("interaction_count", "sum"),
        downloaded_bytes=("downloaded_bytes", "sum"),
        deleted_bytes=("deleted_bytes", "sum"),
    )
    .reset_index()
)
summary_by_date

# %% [markdown]
# # Plot

# %%
def plot_article_counts(
    summary: pd.DataFrame,
    windows: pd.DataFrame,
    period: str,
    output_path: Path,
) -> None:
    """Save and display matching-record counts by aggregation period.

    Args:
        summary: One-row-per-archive processing summary.
        windows: Configured aggregation windows.
        period: Pandas-compatible aggregation period used for the windows.
        output_path: Destination PNG path.
    """

    if summary.empty:
        return

    article_counts = summary.copy()
    article_counts["archive_date"] = pd.to_datetime(
        article_counts["archive_date"]
    )
    window_boundaries = pd.to_datetime(windows["start_timestamp"])
    article_counts["window_index"] = window_boundaries.searchsorted(
        article_counts["archive_date"], side="right"
    ) - 1
    counts_by_window = article_counts.groupby("window_index")[
        "article_count"
    ].sum()

    plot_data = windows[["start_date", "end_date"]].copy()
    plot_data["article_count"] = (
        plot_data.index.to_series().map(counts_by_window).fillna(0).astype(int)
    )
    plot_data["period_label"] = plot_data.apply(
        lambda row: (
            f"{row['start_date']:%Y-%m-%d}\n{row['end_date']:%Y-%m-%d}"
        ),
        axis=1,
    )

    plt.figure(figsize=(14, 6))
    sns.lineplot(
        data=plot_data,
        x="period_label",
        y="article_count",
        marker="o",
    )
    plt.title(
        f"GDELT Labor-Market Articles per {period} Period",
        fontsize=18,
    )
    plt.xlabel("Aggregation period", fontsize=14)
    plt.ylabel("Article records", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:,.0f}")
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()


# %%
plot_article_counts(
    aggregate_summary,
    time_windows,
    aggregation_period,
    aggregate_output_path.with_suffix(".png"),
)
