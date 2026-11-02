"""Optional collector for daily GDELT GKG 1.0-compatible files."""

from __future__ import annotations

import argparse
import io
import sys
import time as time_module
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd

from ai_changed_news.gdelt_metadata import GKG_COLUMNS, OUTPUT_COLUMNS, write_parquet_atomic

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


DEFAULT_BASE_URL = "http://data.gdeltproject.org/gkg"
GKG1_COLUMNS = [
    "DATE",
    "NUMARTS",
    "COUNTS",
    "THEMES",
    "LOCATIONS",
    "PERSONS",
    "ORGANIZATIONS",
    "TONE",
    "CAMEOEVENTIDS",
    "SOURCES",
    "SOURCEURLS",
]


@dataclass(frozen=True)
class Gkg1Config:
    period_dir: Path
    status_path: Path
    failures_path: Path
    start_date: str
    end_date: str
    base_url: str
    timeout_seconds: int
    overwrite_days: bool
    day_workers: int
    row_sample_rate: float
    max_records_per_day: int | None
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


def load_config(path: Path) -> Gkg1Config:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    paths = raw.get("paths", {})
    collection = raw.get("collection", {})
    sampling = raw.get("sampling", {})
    start_date = collection.get("start_date")
    end_date = collection.get("end_date")
    if not start_date or not end_date:
        raise ValueError("collection.start_date and collection.end_date are required")
    return Gkg1Config(
        period_dir=Path(paths.get("period_dir", "data/interim/gdelt_gkg1_days")),
        status_path=Path(paths.get("status_path", "data/logs/gdelt_gkg1_day_status.parquet")),
        failures_path=Path(paths.get("failures_path", "data/logs/gdelt_gkg1_day_failures.parquet")),
        start_date=str(start_date),
        end_date=str(end_date),
        base_url=str(collection.get("base_url", DEFAULT_BASE_URL)).rstrip("/"),
        timeout_seconds=int(collection.get("timeout_seconds", 120)),
        overwrite_days=bool(collection.get("overwrite_days", False)),
        day_workers=int(collection.get("day_workers", 1)),
        row_sample_rate=float(sampling.get("row_sample_rate", 1.0)),
        max_records_per_day=(
            int(sampling["max_records_per_day"]) if sampling.get("max_records_per_day") is not None else None
        ),
        random_seed=int(sampling.get("random_seed", 42)),
    )


def iter_days(start_date: str, end_date: str) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    return [pd.Timestamp(day) for day in pd.date_range(start=start, end=end, freq="D")]


def day_id(day: pd.Timestamp) -> str:
    return pd.Timestamp(day).strftime("%Y%m%d")


def day_url(cfg: Gkg1Config, day: pd.Timestamp) -> str:
    return f"{cfg.base_url}/{day_id(day)}.gkg.csv.zip"


def day_path(cfg: Gkg1Config, day: pd.Timestamp) -> Path:
    return cfg.period_dir / f"{day_id(day)}.parquet"


def download(url: str, timeout_seconds: int) -> bytes:
    request = Request(url, headers={"User-Agent": "ai-changed-news-gkg1-metadata/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


def read_gkg_zip(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if not names:
            raise ValueError("GDELT zip archive did not contain a data file")
        with archive.open(names[0]) as handle:
            first_line = handle.readline().decode("utf-8", errors="replace").rstrip("\n")
            first_columns = first_line.split("\t")
            handle.seek(0)
            if first_columns == GKG1_COLUMNS:
                return pd.read_csv(
                    handle,
                    sep="\t",
                    header=0,
                    dtype="string",
                    on_bad_lines="skip",
                )
            if len(first_columns) == len(GKG1_COLUMNS):
                return pd.read_csv(
                    handle,
                    sep="\t",
                    header=None,
                    names=GKG1_COLUMNS,
                    dtype="string",
                    on_bad_lines="skip",
                )
            return pd.read_csv(
                handle,
                sep="\t",
                header=None,
                names=GKG_COLUMNS,
                dtype="string",
                on_bad_lines="skip",
            )


def parse_gkg_date(value: pd.Series) -> pd.Series:
    text = value.astype("string").str.replace(r"[^0-9]", "", regex=True)
    parsed = pd.to_datetime(text, format="%Y%m%d%H%M%S", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], format="%Y%m%d", errors="coerce")
    return parsed


def choose_column(raw: pd.DataFrame, preferred: str, fallback: str) -> pd.Series:
    preferred_values = raw[preferred] if preferred in raw.columns else pd.Series(pd.NA, index=raw.index, dtype="string")
    fallback_values = raw[fallback] if fallback in raw.columns else pd.Series(pd.NA, index=raw.index, dtype="string")
    return preferred_values.fillna(fallback_values)


def clean(raw: pd.DataFrame, day: pd.Timestamp, url: str) -> pd.DataFrame:
    if set(GKG1_COLUMNS).issubset(raw.columns):
        return clean_gkg1(raw, day, url)
    return clean_gkg2_compatible(raw, day, url)


def clean_gkg1(raw: pd.DataFrame, day: pd.Timestamp, url: str) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "gdelt_record_id": raw["DATE"].astype("string") + "#" + raw.index.astype("string"),
            "timestamp": parse_gkg_date(raw["DATE"]),
            "source": raw["SOURCES"],
            "source_collection_identifier": pd.NA,
            "url": raw["SOURCEURLS"],
            "themes": raw["THEMES"],
            "v2themes": pd.NA,
            "legacy_themes": raw["THEMES"],
            "locations": raw["LOCATIONS"],
            "persons": raw["PERSONS"],
            "organizations": raw["ORGANIZATIONS"],
            "sentiment_v2tone": raw["TONE"],
            "gcam_features": pd.NA,
            "counts": raw["COUNTS"],
            "v2counts": pd.NA,
            "dates": pd.NA,
            "all_names": pd.NA,
            "amounts": pd.NA,
            "translation_info": pd.NA,
        }
    )
    return finish_clean(df, day, url)


def clean_gkg2_compatible(raw: pd.DataFrame, day: pd.Timestamp, url: str) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "gdelt_record_id": raw["GKGRECORDID"],
            "timestamp": parse_gkg_date(raw["DATE"]),
            "source": raw["SourceCommonName"],
            "source_collection_identifier": raw["SourceCollectionIdentifier"],
            "url": raw["DocumentIdentifier"],
            "themes": choose_column(raw, "V2Themes", "Themes"),
            "v2themes": raw["V2Themes"],
            "legacy_themes": raw["Themes"],
            "locations": choose_column(raw, "V2Locations", "Locations"),
            "persons": choose_column(raw, "V2Persons", "Persons"),
            "organizations": choose_column(raw, "V2Organizations", "Organizations"),
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
    return finish_clean(df, day, url)


def finish_clean(df: pd.DataFrame, day: pd.Timestamp, url: str) -> pd.DataFrame:
    df["url"] = df["url"].astype("string").str.strip()
    df["source"] = df["source"].astype("string").str.strip()
    df = df.dropna(subset=["timestamp", "url"])
    df = df[df["url"].str.startswith(("http://", "https://"), na=False)]
    df = df.drop_duplicates(subset=["url"], keep="first")
    df["slice_timestamp"] = day_id(day)
    df["slice_url"] = url
    df["collected_at"] = utc_now()
    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def sample_day(df: pd.DataFrame, cfg: Gkg1Config, day: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    sampled = df
    if cfg.row_sample_rate < 1.0:
        sampled = sampled.sample(frac=cfg.row_sample_rate, random_state=cfg.random_seed)
    if cfg.max_records_per_day is not None and len(sampled) > cfg.max_records_per_day:
        sampled = sampled.sample(n=cfg.max_records_per_day, random_state=cfg.random_seed + int(day_id(day)))
    return sampled.sort_values(["timestamp", "url"]).reset_index(drop=True)


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    if "day" in updated.columns and "updated_at" in updated.columns:
        updated = updated.sort_values(["day", "updated_at"]).drop_duplicates(subset=["day"], keep="last")
    write_parquet_atomic(updated, path)


def day_already_collected(cfg: Gkg1Config, day: pd.Timestamp) -> bool:
    out_path = day_path(cfg, day)
    return out_path.exists() and not cfg.overwrite_days


def fetch_clean_day(day: pd.Timestamp, cfg: Gkg1Config) -> dict[str, Any]:
    url = day_url(cfg, day)
    started = time_module.perf_counter()
    try:
        payload = download(url, cfg.timeout_seconds)
        raw = read_gkg_zip(payload)
        cleaned = sample_day(clean(raw, day, url), cfg, day)
        return {
            "day": day_id(day),
            "url": url,
            "status": "success",
            "raw_rows": len(raw),
            "clean_rows": len(cleaned),
            "seconds": time_module.perf_counter() - started,
            "updated_at": utc_now(),
            "error_type": None,
            "error_message": None,
            "data": cleaned,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "day": day_id(day),
            "url": url,
            "status": "failed",
            "raw_rows": None,
            "clean_rows": None,
            "seconds": time_module.perf_counter() - started,
            "updated_at": utc_now(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "data": None,
        }


def write_day_result(cfg: Gkg1Config, day: pd.Timestamp, result: dict[str, Any]) -> None:
    data = result.pop("data", None)
    append_row(cfg.status_path, result)
    if result["status"] == "success":
        write_parquet_atomic(data, day_path(cfg, day))
        log(f"{day_id(day)}: wrote {len(data):,} rows")
    else:
        append_row(cfg.failures_path, result)
        log(f"{day_id(day)}: failed: {result['error_type']}: {result['error_message']}")


def collect(cfg: Gkg1Config) -> None:
    if not 0 < cfg.row_sample_rate <= 1:
        raise ValueError("sampling.row_sample_rate must be greater than 0 and less than or equal to 1")
    if cfg.day_workers < 1:
        raise ValueError("collection.day_workers must be at least 1")
    days = iter_days(cfg.start_date, cfg.end_date)
    pending_days = []
    for day in days:
        if day_already_collected(cfg, day):
            log(f"{day_id(day)}: already collected")
        else:
            pending_days.append(day)
    if not pending_days:
        return

    max_workers = min(cfg.day_workers, len(pending_days))
    if max_workers == 1:
        for day in progress(pending_days, desc="GKG 1.0 days", unit="day", dynamic_ncols=True):
            write_day_result(cfg, day, fetch_clean_day(day, cfg))
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_clean_day, day, cfg): day for day in pending_days}
        for future in progress(as_completed(futures), total=len(futures), desc="GKG 1.0 days", unit="day", dynamic_ncols=True):
            day = futures[future]
            write_day_result(cfg, day, future.result())


def status(cfg: Gkg1Config) -> None:
    df = pd.read_parquet(cfg.status_path) if cfg.status_path.exists() else pd.DataFrame()
    expected = len(iter_days(cfg.start_date, cfg.end_date))
    if df.empty:
        print(f"collected_days: 0/{expected}")
        return
    latest = df.sort_values(["day", "updated_at"]).drop_duplicates(subset=["day"], keep="last")
    success = int((latest["status"] == "success").sum())
    failed = int((latest["status"] != "success").sum())
    rows = int(latest.loc[latest["status"] == "success", "clean_rows"].fillna(0).sum())
    print(f"collected_days: {success}/{expected}")
    print(f"failed_days: {failed}")
    print(f"clean_rows: {rows:,}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional daily GDELT GKG 1.0 collector.")
    parser.add_argument("command", choices=("collect", "status"))
    parser.add_argument("--config", type=Path, default=Path("configs/gkg1_metadata.example.toml"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    cfg = load_config(args.config)
    if args.command == "collect":
        collect(cfg)
    elif args.command == "status":
        status(cfg)
    else:  # pragma: no cover
        raise ValueError(args.command)
    return 0
