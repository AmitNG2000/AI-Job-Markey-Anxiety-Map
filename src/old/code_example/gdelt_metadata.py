"""Stage 0: collect and clean GDELT GKG metadata slices."""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import random
import time as time_module
import sys
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, time, timezone
from calendar import monthrange
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    TQDM_AVAILABLE = False

    def tqdm(iterable=None, **_: Any):  # type: ignore[no-redef]
        return iterable if iterable is not None else []

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


GKG_COLUMNS = [
    "GKGRECORDID",
    "DATE",
    "SourceCollectionIdentifier",
    "SourceCommonName",
    "DocumentIdentifier",
    "Counts",
    "V2Counts",
    "Themes",
    "V2Themes",
    "Locations",
    "V2Locations",
    "Persons",
    "V2Persons",
    "Organizations",
    "V2Organizations",
    "V2Tone",
    "Dates",
    "GCAM",
    "SharingImage",
    "RelatedImages",
    "SocialImageEmbeds",
    "SocialVideoEmbeds",
    "Quotations",
    "AllNames",
    "Amounts",
    "TranslationInfo",
    "Extras",
]

OUTPUT_COLUMNS = [
    "gdelt_record_id",
    "timestamp",
    "source",
    "source_collection_identifier",
    "url",
    "themes",
    "v2themes",
    "legacy_themes",
    "locations",
    "persons",
    "organizations",
    "sentiment_v2tone",
    "gcam_features",
    "counts",
    "v2counts",
    "dates",
    "all_names",
    "amounts",
    "translation_info",
    "slice_timestamp",
    "slice_url",
    "collected_at",
]

DEFAULT_BASE_URL = "http://data.gdeltproject.org/gdeltv2"
SLICE_FREQUENCY = "15min"


@dataclass(frozen=True)
class MetadataConfig:
    slice_dir: Path
    period_dir: Path
    monthly_output_dir: Path
    monthly_work_dir: Path
    status_path: Path
    failures_path: Path
    start_date: str
    end_date: str
    base_url: str
    timeout_seconds: int
    overwrite_slices: bool
    delete_slice_checkpoints: bool
    slice_workers: int
    slice_sampling_mode: str
    slice_block_hours: int
    slices_per_block: int
    row_sample_rate: float
    max_records_per_period: int | None
    random_seed: int


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def log(message: str) -> None:
    line = f"[{utc_now()}] {message}"
    if TQDM_AVAILABLE and hasattr(tqdm, "write"):
        tqdm.write(line)
    else:
        print(line)


def progress(iterable: Any, **kwargs: Any) -> Any:
    if not TQDM_AVAILABLE:
        return iterable
    return tqdm(iterable, file=sys.stdout, **kwargs)


def load_config(path: Path) -> MetadataConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    paths = raw.get("paths", {})
    collection = raw.get("collection", {})
    slice_sampling = raw.get("slice_sampling", {})
    sampling = raw.get("sampling", {})

    start_date = collection.get("start_date")
    end_date = collection.get("end_date")
    if not start_date or not end_date:
        raise ValueError("collection.start_date and collection.end_date are required")

    max_records = sampling.get("max_records_per_slice")
    if max_records is None:
        max_records = sampling.get("max_records_per_period")
    return MetadataConfig(
        slice_dir=Path(paths.get("slice_dir", "data/interim/gdelt_gkg_slices")),
        period_dir=Path(paths.get("period_dir", "data/interim/gdelt_gkg_days")),
        monthly_output_dir=Path(paths.get("monthly_output_dir", "data/interim/metadata_monthly")),
        monthly_work_dir=Path(paths.get("monthly_work_dir", "data/interim/metadata_monthly_work")),
        status_path=Path(paths.get("status_path", "data/logs/gdelt_gkg_slice_status.parquet")),
        failures_path=Path(paths.get("failures_path", "data/logs/gdelt_gkg_slice_failures.parquet")),
        start_date=str(start_date),
        end_date=str(end_date),
        base_url=str(collection.get("base_url", DEFAULT_BASE_URL)).rstrip("/"),
        timeout_seconds=int(collection.get("timeout_seconds", 60)),
        overwrite_slices=bool(collection.get("overwrite_slices", False)),
        delete_slice_checkpoints=bool(collection.get("delete_slice_checkpoints", True)),
        slice_workers=int(collection.get("slice_workers", 1)),
        slice_sampling_mode=str(slice_sampling.get("mode", "full")),
        slice_block_hours=int(slice_sampling.get("block_hours", 3)),
        slices_per_block=int(slice_sampling.get("slices_per_block", 2)),
        row_sample_rate=float(sampling.get("row_sample_rate", 1.0)),
        max_records_per_period=int(max_records) if max_records is not None else None,
        random_seed=int(sampling.get("random_seed", 42)),
    )


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Atomically write parquet so interrupted files are never treated as checkpoints."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    write_parquet(df, path)


def read_parquet_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def normalize_date_bound(value: str, is_end: bool) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    is_date_only = " " not in value and "T" not in value and len(value) <= 10
    if is_date_only and ts.time() == time.min and is_end:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    return ts.floor(SLICE_FREQUENCY)


def month_start_end(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        year, month_number = [int(part) for part in month.split("-", maxsplit=1)]
    except ValueError as exc:
        raise ValueError(f"month must be formatted as YYYY-MM, got {month!r}") from exc
    if month_number < 1 or month_number > 12:
        raise ValueError(f"month must be formatted as YYYY-MM, got {month!r}")
    last_day = monthrange(year, month_number)[1]
    return pd.Timestamp(year=year, month=month_number, day=1), pd.Timestamp(
        year=year, month=month_number, day=last_day, hour=23, minute=45
    )


def month_date_range(month: str, cfg: MetadataConfig) -> tuple[str, str]:
    month_start, month_end = month_start_end(month)
    cfg_start = normalize_date_bound(cfg.start_date, is_end=False)
    cfg_end = normalize_date_bound(cfg.end_date, is_end=True)
    start = max(month_start, cfg_start)
    end = min(month_end, cfg_end)
    if end < start:
        raise ValueError(f"month {month} is outside configured study period {cfg.start_date} through {cfg.end_date}")
    return start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")


def iter_months(start_date: str, end_date: str) -> list[str]:
    start = normalize_date_bound(start_date, is_end=False).to_period("M")
    end = normalize_date_bound(end_date, is_end=True).to_period("M")
    return [str(month) for month in pd.period_range(start=start, end=end, freq="M")]


def month_scoped_config(cfg: MetadataConfig, month: str) -> MetadataConfig:
    start_date, end_date = month_date_range(month, cfg)
    work_dir = cfg.monthly_work_dir / month
    return MetadataConfig(
        slice_dir=work_dir / "slices",
        period_dir=work_dir / "days",
        monthly_output_dir=cfg.monthly_output_dir,
        monthly_work_dir=cfg.monthly_work_dir,
        status_path=work_dir / "logs" / "slice_status.parquet",
        failures_path=work_dir / "logs" / "slice_failures.parquet",
        start_date=start_date,
        end_date=end_date,
        base_url=cfg.base_url,
        timeout_seconds=cfg.timeout_seconds,
        overwrite_slices=cfg.overwrite_slices,
        delete_slice_checkpoints=cfg.delete_slice_checkpoints,
        slice_workers=cfg.slice_workers,
        slice_sampling_mode=cfg.slice_sampling_mode,
        slice_block_hours=cfg.slice_block_hours,
        slices_per_block=cfg.slices_per_block,
        row_sample_rate=cfg.row_sample_rate,
        max_records_per_period=cfg.max_records_per_period,
        random_seed=cfg.random_seed,
    )


def iter_slice_timestamps(start_date: str, end_date: str) -> pd.DatetimeIndex:
    start = normalize_date_bound(start_date, is_end=False)
    end = normalize_date_bound(end_date, is_end=True)
    if end < start:
        raise ValueError(f"end_date {end_date!r} is before start_date {start_date!r}")
    return pd.date_range(start=start, end=end, freq=SLICE_FREQUENCY)


def slice_id(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%d%H%M%S")


def slice_url(base_url: str, ts: pd.Timestamp) -> str:
    return f"{base_url}/{slice_id(ts)}.gkg.csv.zip"


def slice_path(slice_dir: Path, ts: pd.Timestamp) -> Path:
    return slice_dir / f"{slice_id(ts)}.parquet"


def period_id(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%d")


def period_path(period_dir: Path, ts: pd.Timestamp) -> Path:
    return period_dir / f"{period_id(ts)}.parquet"


def stable_slice_seed(random_seed: int, slice_timestamp: str) -> int:
    digest = hashlib.sha256(f"{random_seed}:{slice_timestamp}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def download_slice(url: str, timeout_seconds: int) -> bytes:
    request = Request(url, headers={"User-Agent": "ai-changed-news-stage0/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


def read_gkg_zip(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if not names:
            raise ValueError("GDELT zip archive did not contain a data file")
        with archive.open(names[0]) as handle:
            return pd.read_csv(
                handle,
                sep="\t",
                header=None,
                names=GKG_COLUMNS,
                dtype="string",
                on_bad_lines="skip",
            )


def clean_metadata(raw: pd.DataFrame, ts: pd.Timestamp, url: str, cfg: MetadataConfig) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "gdelt_record_id": raw["GKGRECORDID"],
            "timestamp": pd.to_datetime(raw["DATE"], format="%Y%m%d%H%M%S", errors="coerce"),
            "source": raw["SourceCommonName"],
            "source_collection_identifier": raw["SourceCollectionIdentifier"],
            "url": raw["DocumentIdentifier"],
            "themes": raw["V2Themes"],
            "v2themes": raw["V2Themes"],
            "legacy_themes": raw["Themes"],
            "locations": raw["V2Locations"],
            "persons": raw["V2Persons"],
            "organizations": raw["V2Organizations"],
            "sentiment_v2tone": raw["V2Tone"],
            "gcam_features": raw["GCAM"],
            "counts": raw["Counts"],
            "v2counts": raw["V2Counts"],
            "dates": raw["Dates"],
            "all_names": raw["AllNames"],
            "amounts": raw["Amounts"],
            "translation_info": raw["TranslationInfo"],
        }
    )
    df["url"] = df["url"].astype("string").str.strip()
    df["source"] = df["source"].astype("string").str.strip()
    df = df.dropna(subset=["timestamp", "url"])
    df = df[df["url"].str.startswith(("http://", "https://"), na=False)]
    df = df.drop_duplicates(subset=["url"], keep="first")

    df["slice_timestamp"] = slice_id(ts)
    df["slice_url"] = url
    df["collected_at"] = utc_now()
    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def validate_sampling(cfg: MetadataConfig) -> None:
    if not 0 < cfg.row_sample_rate <= 1:
        raise ValueError("sampling.row_sample_rate must be greater than 0 and less than or equal to 1")
    if cfg.max_records_per_period is not None and cfg.max_records_per_period < 1:
        raise ValueError("sampling.max_records_per_period must be at least 1 when set")
    if cfg.slice_workers < 1:
        raise ValueError("collection.slice_workers must be at least 1")
    if cfg.slice_sampling_mode not in {"full", "stratified_blocks"}:
        raise ValueError("slice_sampling.mode must be 'full' or 'stratified_blocks'")
    if cfg.slice_block_hours < 1 or 24 % cfg.slice_block_hours != 0:
        raise ValueError("slice_sampling.block_hours must evenly divide 24")
    if cfg.slices_per_block < 1:
        raise ValueError("slice_sampling.slices_per_block must be at least 1")
    if cfg.slices_per_block > cfg.slice_block_hours * 4:
        raise ValueError("slice_sampling.slices_per_block cannot exceed slices available per block")


def block_index(ts: pd.Timestamp, block_hours: int) -> int:
    return int(ts.hour // block_hours)


def block_slice_groups(timestamps: pd.DatetimeIndex, cfg: MetadataConfig) -> dict[int, list[pd.Timestamp]]:
    groups: dict[int, list[pd.Timestamp]] = {}
    for ts in timestamps:
        timestamp = pd.Timestamp(ts)
        groups.setdefault(block_index(timestamp, cfg.slice_block_hours), []).append(timestamp)
    return {block: sorted(values) for block, values in sorted(groups.items())}


def deterministic_shuffle(items: list[pd.Timestamp], seed_key: str, random_seed: int) -> list[pd.Timestamp]:
    shuffled = list(items)
    rng = random.Random(stable_slice_seed(random_seed, seed_key))
    rng.shuffle(shuffled)
    return shuffled


def initial_stratified_plan(timestamps: pd.DatetimeIndex, cfg: MetadataConfig) -> list[pd.Timestamp]:
    planned: list[pd.Timestamp] = []
    day = period_id(pd.Timestamp(timestamps[0])) if len(timestamps) else "empty"
    for block, block_timestamps in block_slice_groups(timestamps, cfg).items():
        shuffled = deterministic_shuffle(block_timestamps, f"{day}:block:{block}", cfg.random_seed)
        planned.extend(sorted(shuffled[: min(cfg.slices_per_block, len(shuffled))]))
    return sorted(planned)


def latest_status_by_slice(cfg: MetadataConfig) -> pd.DataFrame:
    status = read_parquet_if_exists(cfg.status_path)
    if status.empty:
        return status
    return status.sort_values(["slice_timestamp", "updated_at"]).drop_duplicates(
        subset=["slice_timestamp"], keep="last"
    )


def failed_slice_ids(cfg: MetadataConfig) -> set[str]:
    if cfg.overwrite_slices:
        return set()
    status = latest_status_by_slice(cfg)
    if status.empty or "status" not in status.columns:
        return set()
    failed = status[status["status"] != "success"]
    return set(failed["slice_timestamp"].dropna().astype(str))


def existing_slice_ids(timestamps: list[pd.Timestamp] | pd.DatetimeIndex, cfg: MetadataConfig) -> set[str]:
    return {slice_id(pd.Timestamp(ts)) for ts in timestamps if slice_path(cfg.slice_dir, pd.Timestamp(ts)).exists()}


def elapsed(start: float) -> float:
    return time_module.perf_counter() - start


def fetch_clean_slice(ts: pd.Timestamp, cfg: MetadataConfig) -> dict[str, Any]:
    sid = slice_id(ts)
    url = slice_url(cfg.base_url, ts)
    total_start = time_module.perf_counter()
    timings: dict[str, float] = {}

    try:
        started = time_module.perf_counter()
        payload = download_slice(url, cfg.timeout_seconds)
        timings["download_seconds"] = elapsed(started)

        started = time_module.perf_counter()
        raw = read_gkg_zip(payload)
        timings["parse_seconds"] = elapsed(started)

        started = time_module.perf_counter()
        cleaned = clean_metadata(raw, ts, url, cfg)
        timings["clean_seconds"] = elapsed(started)
        timings["worker_total_seconds"] = elapsed(total_start)

        return {
            "slice_timestamp": sid,
            "slice_url": url,
            "status": "success",
            "raw_rows": len(raw),
            "clean_rows": len(cleaned),
            "data": cleaned,
            "timings": timings,
            "error_type": None,
            "error_message": None,
        }
    except Exception as exc:  # noqa: BLE001
        timings["worker_total_seconds"] = elapsed(total_start)
        return {
            "slice_timestamp": sid,
            "slice_url": url,
            "status": "failed",
            "raw_rows": None,
            "clean_rows": None,
            "data": None,
            "timings": timings,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def format_timings(timings: dict[str, float]) -> str:
    order = [
        "download_seconds",
        "parse_seconds",
        "clean_seconds",
        "slice_parquet_write_seconds",
        "failure_log_write_seconds",
        "status_write_seconds",
        "read_slice_parquets_seconds",
        "sample_period_seconds",
        "daily_parquet_write_seconds",
        "delete_slice_checkpoints_seconds",
        "worker_total_seconds",
        "total_seconds",
    ]
    parts = []
    for key in order:
        if key in timings:
            parts.append(f"{key.removesuffix('_seconds')}={timings[key]:.2f}s")
    return " ".join(parts)


def append_or_update_status(path: Path, row: dict[str, Any]) -> None:
    existing = read_parquet_if_exists(path)
    new_row = pd.DataFrame([row])
    updated = new_row if existing.empty else pd.concat([existing, new_row], ignore_index=True)
    updated = updated.sort_values(["slice_timestamp", "updated_at"]).drop_duplicates(
        subset=["slice_timestamp"], keep="last"
    )
    write_parquet(updated, path)


def append_failure(path: Path, row: dict[str, Any]) -> None:
    existing = read_parquet_if_exists(path)
    new_row = pd.DataFrame([row])
    updated = new_row if existing.empty else pd.concat([existing, new_row], ignore_index=True)
    write_parquet(updated, path)


def write_slice_result(ts: pd.Timestamp, cfg: MetadataConfig, result: dict[str, Any]) -> pd.DataFrame | None:
    sid = slice_id(ts)
    out_path = slice_path(cfg.slice_dir, ts)
    timings = dict(result.get("timings") or {})
    total_start = time_module.perf_counter()

    if result["status"] == "success":
        cleaned = result["data"]
        started = time_module.perf_counter()
        write_parquet(cleaned, out_path)
        timings["slice_parquet_write_seconds"] = elapsed(started)

        started = time_module.perf_counter()
        append_or_update_status(
            cfg.status_path,
            {
                "slice_timestamp": sid,
                "slice_url": result["slice_url"],
                "slice_path": str(out_path),
                "status": "success",
                "raw_rows": result["raw_rows"],
                "clean_rows": result["clean_rows"],
                "updated_at": utc_now(),
                "error_type": None,
                "error_message": None,
            },
        )
        timings["status_write_seconds"] = elapsed(started)
        timings["total_seconds"] = timings.get("worker_total_seconds", 0.0) + elapsed(total_start)
        log(
            f"{sid}: success raw={result['raw_rows']:,} clean={result['clean_rows']:,} "
            f"{format_timings(timings)}"
        )
        return cleaned

    failure = {
        "slice_timestamp": sid,
        "slice_url": result["slice_url"],
        "status": "failed",
        "failed_at": utc_now(),
        "error_type": result["error_type"],
        "error_message": result["error_message"],
    }
    started = time_module.perf_counter()
    append_failure(cfg.failures_path, failure)
    timings["failure_log_write_seconds"] = elapsed(started)

    started = time_module.perf_counter()
    append_or_update_status(
        cfg.status_path,
        {
            "slice_timestamp": sid,
            "slice_url": result["slice_url"],
            "slice_path": str(out_path),
            "status": "failed",
            "raw_rows": None,
            "clean_rows": None,
            "updated_at": failure["failed_at"],
            "error_type": failure["error_type"],
            "error_message": failure["error_message"],
        },
    )
    timings["status_write_seconds"] = elapsed(started)
    timings["total_seconds"] = timings.get("worker_total_seconds", 0.0) + elapsed(total_start)
    log(f"{sid}: failed ({failure['error_type']}: {failure['error_message']}) {format_timings(timings)}")
    return None


def collect_slice(ts: pd.Timestamp, cfg: MetadataConfig) -> pd.DataFrame | None:
    out_path = slice_path(cfg.slice_dir, ts)
    if out_path.exists() and not cfg.overwrite_slices:
        return None
    return write_slice_result(ts, cfg, fetch_clean_slice(ts, cfg))


def collect_slice_batch(timestamps: list[pd.Timestamp], cfg: MetadataConfig, day_label: str) -> dict[str, str]:
    pending = [
        pd.Timestamp(ts)
        for ts in timestamps
        if cfg.overwrite_slices or not slice_path(cfg.slice_dir, pd.Timestamp(ts)).exists()
    ]
    results = {
        slice_id(pd.Timestamp(ts)): "success"
        for ts in timestamps
        if not cfg.overwrite_slices and slice_path(cfg.slice_dir, pd.Timestamp(ts)).exists()
    }
    if not pending:
        return results

    if cfg.slice_workers == 1:
        for ts in progress(pending, desc=day_label, unit="slice", leave=False, dynamic_ncols=True):
            collected = collect_slice(ts, cfg)
            results[slice_id(ts)] = "success" if collected is not None else "failed"
        return results

    max_workers = min(cfg.slice_workers, len(pending))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_clean_slice, ts, cfg): ts for ts in pending}
        completed = as_completed(futures)
        for future in progress(completed, total=len(futures), desc=day_label, unit="slice", leave=False, dynamic_ncols=True):
            ts = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {
                    "slice_timestamp": slice_id(ts),
                    "slice_url": slice_url(cfg.base_url, ts),
                    "status": "failed",
                    "raw_rows": None,
                    "clean_rows": None,
                    "data": None,
                    "timings": {},
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            write_slice_result(ts, cfg, result)
            results[slice_id(ts)] = result["status"]
    return results


def collect_full_slices_for_period(period_timestamps: pd.DatetimeIndex, cfg: MetadataConfig, day_label: str) -> None:
    collect_slice_batch([pd.Timestamp(ts) for ts in period_timestamps], cfg, day_label)


def next_viable(candidates: list[pd.Timestamp], success_ids: set[str], failed_ids: set[str]) -> pd.Timestamp | None:
    for ts in candidates:
        sid = slice_id(ts)
        if sid not in success_ids and sid not in failed_ids:
            return ts
    return None


def collect_stratified_slices_for_period(
    period: pd.Timestamp, period_timestamps: pd.DatetimeIndex, cfg: MetadataConfig, day_label: str
) -> None:
    groups = block_slice_groups(period_timestamps, cfg)
    day = period_id(period)
    ordered_candidates = {
        block: deterministic_shuffle(timestamps, f"{day}:block:{block}", cfg.random_seed)
        for block, timestamps in groups.items()
    }
    quotas = {block: min(cfg.slices_per_block, len(timestamps)) for block, timestamps in groups.items()}
    success_ids = existing_slice_ids([ts for timestamps in groups.values() for ts in timestamps], cfg)
    failed_ids = failed_slice_ids(cfg)

    while True:
        batch: list[pd.Timestamp] = []
        for block, quota in quotas.items():
            block_successes = sum(1 for ts in groups[block] if slice_id(ts) in success_ids)
            if block_successes >= quota:
                continue
            candidate = next_viable(ordered_candidates[block], success_ids, failed_ids)
            if candidate is not None:
                batch.append(candidate)
        if not batch:
            break
        results = collect_slice_batch(batch, cfg, day_label)
        for sid, status in results.items():
            if status == "success":
                success_ids.add(sid)
            else:
                failed_ids.add(sid)

    target_total = sum(quotas.values())
    current_total = sum(1 for timestamps in groups.values() for ts in timestamps if slice_id(ts) in success_ids)
    deficit = max(0, target_total - current_total)
    if deficit == 0:
        log(f"{day}: stratified slices selected {current_total}/{target_total}")
        return

    extra_pools = {
        block: [
            ts
            for ts in deterministic_shuffle(timestamps, f"{day}:reallocate:{block}", cfg.random_seed)
            if slice_id(ts) not in success_ids and slice_id(ts) not in failed_ids
        ]
        for block, timestamps in groups.items()
    }
    rng = random.Random(stable_slice_seed(cfg.random_seed, f"{day}:reallocate"))
    while deficit > 0 and any(extra_pools.values()):
        batch = []
        while len(batch) < cfg.slice_workers and deficit - len(batch) > 0 and any(extra_pools.values()):
            available_blocks = [block for block, pool in extra_pools.items() if pool]
            block = rng.choice(available_blocks)
            batch.append(extra_pools[block].pop(0))
        results = collect_slice_batch(batch, cfg, day_label)
        for sid, status in results.items():
            if status == "success":
                success_ids.add(sid)
                deficit -= 1
            else:
                failed_ids.add(sid)

    current_total = sum(1 for timestamps in groups.values() for ts in timestamps if slice_id(ts) in success_ids)
    if current_total < target_total:
        log(f"{day}: stratified slices selected {current_total}/{target_total}; no more viable same-day replacements")
    else:
        log(f"{day}: stratified slices selected {current_total}/{target_total}")


def collect_slices_for_period(period: pd.Timestamp, period_timestamps: pd.DatetimeIndex, cfg: MetadataConfig, day_label: str) -> None:
    if cfg.slice_sampling_mode == "full":
        collect_full_slices_for_period(period_timestamps, cfg, day_label)
    else:
        collect_stratified_slices_for_period(period, period_timestamps, cfg, day_label)


def sample_period(period: pd.Timestamp, df: pd.DataFrame, cfg: MetadataConfig) -> pd.DataFrame:
    sampled = df.drop_duplicates(subset=["url"], keep="first")
    seed = stable_slice_seed(cfg.random_seed, period_id(period))
    if cfg.row_sample_rate < 1.0:
        sampled = sampled.sample(frac=cfg.row_sample_rate, random_state=seed)
    if cfg.max_records_per_period is not None and len(sampled) > cfg.max_records_per_period:
        sampled = sampled.sample(n=cfg.max_records_per_period, random_state=seed)
    return sampled.sort_values(["timestamp", "url"]).reset_index(drop=True)


def compact_period(period: pd.Timestamp, slice_paths: list[Path], cfg: MetadataConfig) -> pd.DataFrame | None:
    total_start = time_module.perf_counter()
    timings: dict[str, float] = {}
    out_path = period_path(cfg.period_dir, period)
    if out_path.exists() and not cfg.overwrite_slices:
        log(f"{period_id(period)}: already compacted")
        return None

    existing_paths = [path for path in slice_paths if path.exists()]
    if not existing_paths:
        log(f"{period_id(period)}: no successful slice files to compact")
        return None

    started = time_module.perf_counter()
    frames = [pd.read_parquet(path) for path in existing_paths]
    timings["read_slice_parquets_seconds"] = elapsed(started)

    started = time_module.perf_counter()
    sampled = sample_period(period, pd.concat(frames, ignore_index=True), cfg)
    timings["sample_period_seconds"] = elapsed(started)

    started = time_module.perf_counter()
    write_parquet(sampled, out_path)
    timings["daily_parquet_write_seconds"] = elapsed(started)

    if cfg.delete_slice_checkpoints:
        started = time_module.perf_counter()
        for path in existing_paths:
            path.unlink()
        timings["delete_slice_checkpoints_seconds"] = elapsed(started)
    timings["total_seconds"] = elapsed(total_start)
    log(
        f"{period_id(period)}: compacted {len(sampled):,} rows from {len(existing_paths):,} slices "
        f"to {out_path} {format_timings(timings)}"
    )
    return sampled


def expected_slice_ids(timestamps: pd.DatetimeIndex) -> set[str]:
    return {slice_id(pd.Timestamp(ts)) for ts in timestamps}


def attempted_slice_ids(cfg: MetadataConfig) -> set[str]:
    status = read_parquet_if_exists(cfg.status_path)
    if status.empty or "slice_timestamp" not in status.columns:
        return set()
    return set(status["slice_timestamp"].dropna().astype(str))


def period_is_complete(period_file: Path, timestamps: pd.DatetimeIndex, cfg: MetadataConfig) -> bool:
    if not period_file.exists():
        return False
    if cfg.slice_sampling_mode == "stratified_blocks":
        return True
    return expected_slice_ids(timestamps).issubset(attempted_slice_ids(cfg))


def collect_metadata(cfg: MetadataConfig) -> None:
    validate_sampling(cfg)
    timestamps = iter_slice_timestamps(cfg.start_date, cfg.end_date)
    log(f"collecting {len(timestamps):,} GDELT GKG slices")
    if not TQDM_AVAILABLE:
        log("tqdm is not installed; install requirements to show progress bars")
    timestamps_by_period = {
        period: pd.DatetimeIndex(group)
        for period, group in pd.Series(timestamps, index=timestamps).groupby(lambda ts: pd.Timestamp(ts).normalize())
    }
    day_items = list(timestamps_by_period.items())
    for period, period_timestamps in progress(day_items, desc="Days", unit="day", dynamic_ncols=True):
        daily_path = period_path(cfg.period_dir, period)
        if not cfg.overwrite_slices and period_is_complete(daily_path, period_timestamps, cfg):
            log(f"{period_id(period)}: already compacted")
            continue
        if daily_path.exists() and not cfg.overwrite_slices:
            log(f"{period_id(period)}: removing incomplete daily checkpoint")
            daily_path.unlink()
        slice_paths = []
        day_label = f"Slices {period_id(period)}"
        collect_slices_for_period(period, period_timestamps, cfg, day_label)
        for ts in period_timestamps:
            slice_paths.append(slice_path(cfg.slice_dir, ts))
        compact_period(period, slice_paths, cfg)


def collect_month(cfg: MetadataConfig, month: str) -> None:
    month_cfg = month_scoped_config(cfg, month)
    log(f"{month}: collecting month into isolated workspace {month_cfg.monthly_work_dir / month}")
    collect_metadata(month_cfg)


def parse_v2themes(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    themes = []
    for item in str(value).split(";"):
        item = item.strip()
        if not item:
            continue
        themes.append(item.split(",", maxsplit=1)[0])
    return themes


def shannon_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def sampling_metrics(df: pd.DataFrame) -> dict[str, Any]:
    source_counts = Counter(df["source"].dropna().astype(str))
    theme_counts: Counter[str] = Counter()
    theme_column = "v2themes" if "v2themes" in df.columns else "themes"
    for value in df[theme_column].dropna():
        theme_counts.update(parse_v2themes(value))
    return {
        "rows": len(df),
        "unique_sources": len(source_counts),
        "unique_v2themes": len(theme_counts),
        "source_entropy": shannon_entropy(source_counts),
        "theme_entropy": shannon_entropy(theme_counts),
    }


def simulate_daily_sampling(df: pd.DataFrame, rates: list[int], random_seed: int) -> pd.DataFrame:
    if df.empty:
        raise ValueError("No metadata rows found for the selected month")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "url"])
    df["day"] = df["timestamp"].dt.strftime("%Y%m%d")

    rows = []
    for rate in rates:
        samples = []
        for day, day_df in df.groupby("day", sort=True):
            day_df = day_df.drop_duplicates(subset=["url"], keep="first")
            n = min(rate, len(day_df))
            seed = stable_slice_seed(random_seed, f"{day}:{rate}")
            samples.append(day_df.sample(n=n, random_state=seed) if n < len(day_df) else day_df)
        sampled = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
        metrics = sampling_metrics(sampled)
        metrics["records_per_day"] = rate
        rows.append(metrics)
    return pd.DataFrame(rows)[
        ["records_per_day", "rows", "unique_sources", "unique_v2themes", "source_entropy", "theme_entropy"]
    ]


def load_month_metadata(cfg: MetadataConfig, month: str) -> pd.DataFrame:
    month_path = cfg.monthly_output_dir / f"{month}.parquet"
    if month_path.exists():
        return pd.read_parquet(month_path)
    period_paths = sorted(cfg.period_dir.glob(f"{month.replace('-', '')}*.parquet"))
    if period_paths:
        return pd.concat([pd.read_parquet(path) for path in period_paths], ignore_index=True)
    raise ValueError(f"No month metadata found in {cfg.monthly_output_dir} or {cfg.period_dir}")


def sampling_diagnostics(cfg: MetadataConfig, month: str, rates: list[int]) -> None:
    metadata = load_month_metadata(cfg, month)
    result = simulate_daily_sampling(metadata, rates, cfg.random_seed)
    log(f"sampling diagnostics for {month} from {len(metadata):,} available rows")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def planned_slices_for_day(timestamps: pd.DatetimeIndex, cfg: MetadataConfig) -> list[pd.Timestamp]:
    if cfg.slice_sampling_mode == "full":
        return [pd.Timestamp(ts) for ts in timestamps]
    return initial_stratified_plan(timestamps, cfg)


def slice_plan(cfg: MetadataConfig, month: str | None = None) -> None:
    validate_sampling(cfg)
    if month:
        start_date, end_date = month_date_range(month, cfg)
    else:
        start_date, end_date = cfg.start_date, cfg.end_date

    timestamps = iter_slice_timestamps(start_date, end_date)
    timestamps_by_period = {
        period: pd.DatetimeIndex(group)
        for period, group in pd.Series(timestamps, index=timestamps).groupby(lambda ts: pd.Timestamp(ts).normalize())
    }

    full_count = len(timestamps)
    planned_by_day = {
        period_id(period): planned_slices_for_day(day_timestamps, cfg)
        for period, day_timestamps in timestamps_by_period.items()
    }
    selected_count = sum(len(values) for values in planned_by_day.values())
    reduction = 0.0 if full_count == 0 else 100 * (1 - selected_count / full_count)

    print(f"mode: {cfg.slice_sampling_mode}")
    if cfg.slice_sampling_mode == "stratified_blocks":
        print(f"block_hours: {cfg.slice_block_hours}")
        print(f"slices_per_block: {cfg.slices_per_block}")
    print(f"days: {len(planned_by_day):,}")
    print(f"full_slices: {full_count:,}")
    print(f"selected_slices: {selected_count:,}")
    print(f"expected_download_reduction: {reduction:.2f}%")
    print()
    for day, planned in planned_by_day.items():
        timestamps_text = ", ".join(slice_id(ts) for ts in planned)
        print(f"{day}\tselected={len(planned)}\t{timestamps_text}")


def parquet_row_count(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def configured_periods(cfg: MetadataConfig) -> pd.DataFrame:
    timestamps = iter_slice_timestamps(cfg.start_date, cfg.end_date)
    rows = []
    for period, group in pd.Series(timestamps, index=timestamps).groupby(lambda ts: pd.Timestamp(ts).normalize()):
        period_ts = pd.DatetimeIndex(group)
        planned = planned_slices_for_day(period_ts, cfg)
        rows.append(
            {
                "day": period_id(period),
                "year_month": pd.Timestamp(period).strftime("%Y-%m"),
                "expected_full_slices": len(period_ts),
                "expected_configured_slices": len(planned),
            }
        )
    return pd.DataFrame(rows)


def compacted_period_counts(cfg: MetadataConfig, expected_days: set[str]) -> pd.DataFrame:
    rows = []
    for path in sorted(cfg.period_dir.glob("*.parquet")):
        if path.stem not in expected_days:
            continue
        try:
            rows.append({"day": path.stem, "rows": parquet_row_count(path), "path": str(path)})
        except Exception as exc:  # noqa: BLE001
            rows.append({"day": path.stem, "rows": None, "path": str(path), "read_error": str(exc)})
    return pd.DataFrame(rows)


def latest_status_for_config(cfg: MetadataConfig) -> pd.DataFrame:
    status = latest_status_by_slice(cfg)
    if status.empty:
        return status
    status = status.copy()
    status["_slice_dt"] = pd.to_datetime(status["slice_timestamp"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
    start = normalize_date_bound(cfg.start_date, is_end=False)
    end = normalize_date_bound(cfg.end_date, is_end=True)
    return status[(status["_slice_dt"] >= start) & (status["_slice_dt"] <= end)].reset_index(drop=True)


def monthly_row_summary(expected: pd.DataFrame, compacted: pd.DataFrame, cfg: MetadataConfig) -> pd.DataFrame:
    if expected.empty:
        return pd.DataFrame()
    daily = expected[["day", "year_month"]].merge(compacted[["day", "rows"]] if not compacted.empty else pd.DataFrame(columns=["day", "rows"]), on="day", how="left")
    daily["collected"] = daily["rows"].notna()
    rows = []
    for year_month, month_df in daily.groupby("year_month", sort=True):
        collected = month_df[month_df["collected"]]
        summary = {
            "month": year_month,
            "days_collected": int(collected["day"].nunique()),
            "days_expected": int(month_df["day"].nunique()),
            "rows": int(collected["rows"].sum()) if not collected.empty else 0,
            "avg_rows_per_collected_day": float(collected["rows"].mean()) if not collected.empty else 0.0,
            "min_day_rows": int(collected["rows"].min()) if not collected.empty else 0,
            "max_day_rows": int(collected["rows"].max()) if not collected.empty else 0,
        }
        if cfg.max_records_per_period:
            summary["avg_pct_daily_cap"] = 100 * summary["avg_rows_per_collected_day"] / cfg.max_records_per_period
        rows.append(summary)
    return pd.DataFrame(rows)


def fully_failed_404_days(expected: pd.DataFrame, status: pd.DataFrame) -> list[str]:
    if expected.empty or status.empty:
        return []
    failures = status.copy()
    failures["day"] = failures["slice_timestamp"].astype(str).str[:8]
    failures = failures[
        (failures["status"] == "failed")
        & (failures["error_type"] == "HTTPError")
        & (failures["error_message"].astype(str).str.contains("404", na=False))
    ]
    if failures.empty:
        return []
    failed_counts = failures.groupby("day").size()
    expected_counts = expected.set_index("day")["expected_full_slices"]
    return sorted(day for day, expected_count in expected_counts.items() if failed_counts.get(day, 0) >= expected_count)


def stratified_fallback_summary(cfg: MetadataConfig, status: pd.DataFrame) -> pd.DataFrame:
    if cfg.slice_sampling_mode != "stratified_blocks":
        return pd.DataFrame()

    successes = status[status["status"] == "success"].copy() if not status.empty else pd.DataFrame()
    if successes.empty:
        success_ids: set[str] = set()
    else:
        success_ids = set(successes["slice_timestamp"].dropna().astype(str))

    timestamps = iter_slice_timestamps(cfg.start_date, cfg.end_date)
    rows = []
    for period, group in pd.Series(timestamps, index=timestamps).groupby(lambda ts: pd.Timestamp(ts).normalize()):
        day_ts = pd.DatetimeIndex(group)
        groups = block_slice_groups(day_ts, cfg)
        proper_blocks = 0
        deficit_blocks = 0
        fallback_slices = 0
        proper_slices = 0
        target_slices = 0
        collected_slices = 0
        for block, block_timestamps in groups.items():
            quota = min(cfg.slices_per_block, len(block_timestamps))
            actual = sum(1 for ts in block_timestamps if slice_id(ts) in success_ids)
            target_slices += quota
            collected_slices += actual
            proper_slices += min(actual, quota)
            fallback_slices += max(0, actual - quota)
            if actual >= quota:
                proper_blocks += 1
            else:
                deficit_blocks += 1
        rows.append(
            {
                "day": period_id(period),
                "year_month": pd.Timestamp(period).strftime("%Y-%m"),
                "proper_blocks": proper_blocks,
                "deficit_blocks": deficit_blocks,
                "fallback_slices": fallback_slices,
                "proper_slices": proper_slices,
                "target_slices": target_slices,
                "collected_slices": collected_slices,
            }
        )

    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily
    daily = daily[daily["collected_slices"] > 0].reset_index(drop=True)
    if daily.empty:
        return daily
    monthly = (
        daily.groupby("year_month", sort=True)
        .agg(
            days=("day", "nunique"),
            proper_blocks=("proper_blocks", "sum"),
            deficit_blocks=("deficit_blocks", "sum"),
            fallback_slices=("fallback_slices", "sum"),
            proper_slices=("proper_slices", "sum"),
            target_slices=("target_slices", "sum"),
            collected_slices=("collected_slices", "sum"),
        )
        .reset_index()
        .rename(columns={"year_month": "month"})
    )
    monthly["proper_block_pct"] = 100 * monthly["proper_blocks"] / (monthly["proper_blocks"] + monthly["deficit_blocks"])
    monthly["fallback_slice_pct"] = monthly.apply(
        lambda row: 0.0 if row["collected_slices"] == 0 else 100 * row["fallback_slices"] / row["collected_slices"],
        axis=1,
    )
    return monthly


def summarize_status(cfg: MetadataConfig) -> None:
    validate_sampling(cfg)
    expected = configured_periods(cfg)
    expected_days = set(expected["day"]) if not expected.empty else set()
    compacted = compacted_period_counts(cfg, expected_days)
    status = latest_status_for_config(cfg)

    unavailable_404_days = fully_failed_404_days(expected, status)
    unavailable_404_set = set(unavailable_404_days)
    expected_day_count = len(expected)
    available_expected_day_count = max(0, expected_day_count - len(unavailable_404_set))
    compacted_day_count = int(compacted["day"].nunique()) if not compacted.empty else 0
    rows_collected = int(compacted["rows"].dropna().sum()) if not compacted.empty else 0
    expected_configured_slices = int(expected["expected_configured_slices"].sum()) if not expected.empty else 0
    expected_full_slices = int(expected["expected_full_slices"].sum()) if not expected.empty else 0
    available_expected = expected[~expected["day"].isin(unavailable_404_set)] if not expected.empty else expected
    available_expected_configured_slices = (
        int(available_expected["expected_configured_slices"].sum()) if not available_expected.empty else 0
    )
    available_expected_full_slices = int(available_expected["expected_full_slices"].sum()) if not available_expected.empty else 0
    successful_slices = int((status["status"] == "success").sum()) if not status.empty else 0

    print("Metadata collection status")
    print(f"config: {cfg.start_date} through {cfg.end_date}")
    print(f"slice_sampling: {cfg.slice_sampling_mode}")
    if cfg.slice_sampling_mode == "stratified_blocks":
        print(f"stratified_blocks: block_hours={cfg.slice_block_hours}, slices_per_block={cfg.slices_per_block}")
    print(f"daily_row_sampling: row_sample_rate={cfg.row_sample_rate}, max_records_per_period={cfg.max_records_per_period}")
    print()

    print("Coverage")
    print(f"compacted_days: {compacted_day_count:,}/{expected_day_count:,} ({100 * compacted_day_count / expected_day_count:.2f}%)")
    if unavailable_404_days:
        available_day_pct = 100.0 if available_expected_day_count == 0 else 100 * compacted_day_count / available_expected_day_count
        print(
            "available_source_days: "
            f"{compacted_day_count:,}/{available_expected_day_count:,} ({available_day_pct:.2f}%)"
        )
    if expected_configured_slices:
        print(
            "configured_slice_coverage: "
            f"{successful_slices:,}/{expected_configured_slices:,} ({100 * successful_slices / expected_configured_slices:.2f}%)"
        )
    if unavailable_404_days and available_expected_configured_slices:
        print(
            "available_source_configured_slice_coverage: "
            f"{successful_slices:,}/{available_expected_configured_slices:,} "
            f"({100 * successful_slices / available_expected_configured_slices:.2f}%)"
        )
    if expected_full_slices:
        print(
            "full_15min_slice_equivalent: "
            f"{successful_slices:,}/{expected_full_slices:,} ({100 * successful_slices / expected_full_slices:.2f}%)"
        )
    if unavailable_404_days and available_expected_full_slices:
        print(
            "available_source_full_15min_slice_equivalent: "
            f"{successful_slices:,}/{available_expected_full_slices:,} "
            f"({100 * successful_slices / available_expected_full_slices:.2f}%)"
        )
    print(f"compacted_daily_rows: {rows_collected:,}")
    if cfg.max_records_per_period and expected_day_count:
        configured_row_capacity = expected_day_count * cfg.max_records_per_period
        print(
            "configured_daily_cap_fill: "
            f"{rows_collected:,}/{configured_row_capacity:,} ({100 * rows_collected / configured_row_capacity:.2f}%)"
        )
    if not compacted.empty and compacted["rows"].notna().any():
        daily_rows = compacted["rows"].dropna()
        print(
            "rows_per_collected_day: "
            f"avg={daily_rows.mean():,.1f}, median={daily_rows.median():,.1f}, "
            f"min={int(daily_rows.min()):,}, max={int(daily_rows.max()):,}"
        )
    missing_days = sorted((expected_days - set(compacted["day"])) - unavailable_404_set) if not compacted.empty else sorted(expected_days - unavailable_404_set)
    if missing_days:
        preview = ", ".join(missing_days[:12])
        suffix = " ..." if len(missing_days) > 12 else ""
        print(f"missing_days: {len(missing_days):,} ({preview}{suffix})")
    else:
        print("missing_days: 0")
    if unavailable_404_days:
        preview = ", ".join(unavailable_404_days[:12])
        suffix = " ..." if len(unavailable_404_days) > 12 else ""
        print(f"upstream_404_days: {len(unavailable_404_days):,} ({preview}{suffix})")
    available_complete = compacted_day_count >= available_expected_day_count and not missing_days
    print(f"collection_complete_for_available_source: {'yes' if available_complete else 'no'}")

    print()

    monthly = monthly_row_summary(expected, compacted, cfg)
    if not monthly.empty:
        print("Monthly row summary")
        print(monthly.to_string(index=False, float_format=lambda value: f"{value:,.1f}"))
        print()

    fallback = stratified_fallback_summary(cfg, status)
    if not fallback.empty:
        print("Stratified block/fallback summary")
        print(fallback.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GDELT GKG metadata collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Stage 0 commands:
  collect                Collect configured date range into compacted daily parquet.
  collect-month          Collect one YYYY-MM month into an isolated monthly workspace.
  status                 Summarize collection coverage, daily row yield, and stratified fallback use.
  sampling-diagnostics   Simulate daily row caps for one month from existing metadata.
  slice-plan             Show the deterministic initial slice plan for the configured sampling mode.
  list-months            Print configured study months.

Important options:
  --config PATH          TOML config path. Defaults to configs/metadata_collection.toml.
  --month YYYY-MM        Required for collect-month and sampling-diagnostics; optional for slice-plan.
  --rates N ...          Daily row sample sizes for sampling-diagnostics.

Config fields:
  collection.start_date/end_date bound the study window. Date-only end_date includes the full day.
  collection.slice_workers controls parallel slice downloads.
  slice_sampling.mode is "full" or "stratified_blocks".
  slice_sampling.block_hours and slices_per_block control stratified daily coverage.
  sampling.row_sample_rate and max_records_per_period control daily compacted row volume.
""",
    )
    parser.add_argument("command", choices=("collect", "collect-month", "status", "sampling-diagnostics", "slice-plan", "list-months"))
    parser.add_argument("--config", default=Path("configs/metadata_collection.toml"), type=Path)
    parser.add_argument("--month", help="Month for sampling diagnostics, formatted as YYYY-MM")
    parser.add_argument(
        "--rates",
        nargs="+",
        type=int,
        default=[250, 500, 1000, 2000, 5000],
        help="Daily sample sizes to simulate for sampling diagnostics",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    cfg = load_config(args.config)

    if args.command == "collect":
        collect_metadata(cfg)
    elif args.command == "collect-month":
        if not args.month:
            raise ValueError("--month is required for collect-month")
        collect_month(cfg, args.month)
    elif args.command == "status":
        summarize_status(cfg)
    elif args.command == "sampling-diagnostics":
        if not args.month:
            raise ValueError("--month is required for sampling-diagnostics")
        sampling_diagnostics(cfg, args.month, args.rates)
    elif args.command == "slice-plan":
        slice_plan(cfg, args.month)
    elif args.command == "list-months":
        print("\n".join(iter_months(cfg.start_date, cfg.end_date)))
    else:  # pragma: no cover
        raise ValueError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
