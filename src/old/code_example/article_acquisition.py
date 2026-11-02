"""Article acquisition pipeline: metadata sampling, scraping, language filtering."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import socket
import ssl
import sys
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from configparser import ConfigParser
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

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


REQUIRED_COLUMNS = ("timestamp", "source", "url")
TERMINAL_STATUSES = {"success", "non_english", "too_short"}
DEFAULT_CONFIG_PATH = Path("configs/article_acquisition.toml")


@dataclass(frozen=True)
class AcquisitionConfig:
    metadata_path: Path
    metadata_period_dir: Path | None
    candidates_path: Path
    articles_path: Path
    status_path: Path
    attempts_path: Path
    target_usable_per_month: int
    candidate_multiplier: float
    candidate_mode: str
    random_seed: int
    save_every: int
    min_text_chars: int
    max_attempts: int
    workers: int
    download_timeout_seconds: int
    min_free_disk_gb: float
    stop_at_target: bool
    start_date: str | None
    end_date: str | None


class ArticleAcquisitionError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


class DownloadDiagnosticError(ArticleAcquisitionError):
    def __init__(
        self,
        reason: str,
        message: str,
        *,
        http_status: int | None = None,
        final_url: str | None = None,
    ) -> None:
        super().__init__("download", message)
        self.failure_reason = reason
        self.http_status = http_status
        self.final_url = final_url


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def progress(iterable: Any = None, **kwargs: Any) -> Any:
    if not TQDM_AVAILABLE:
        if iterable is None:
            class _NoProgress:
                def update(self, _: int = 1) -> None:
                    return None

            return nullcontext(_NoProgress())
        return iterable
    return tqdm(iterable, file=sys.stdout, **kwargs)


def default_config() -> AcquisitionConfig:
    return AcquisitionConfig(
        metadata_path=Path("data/interim/gdelt_clean_metadata.parquet"),
        metadata_period_dir=Path("data/interim/gdelt_gkg_days"),
        candidates_path=Path("data/interim/article_candidates.parquet"),
        articles_path=Path("data/processed/articles_english.parquet"),
        status_path=Path("data/logs/article_url_status.parquet"),
        attempts_path=Path("data/logs/article_attempt_history.parquet"),
        target_usable_per_month=10000,
        candidate_multiplier=3.0,
        candidate_mode="sample",
        random_seed=42,
        save_every=100,
        min_text_chars=500,
        max_attempts=2,
        workers=8,
        download_timeout_seconds=10,
        min_free_disk_gb=0.0,
        stop_at_target=True,
        start_date="2018-01-01",
        end_date=None,
    )


def load_config(path: Path) -> AcquisitionConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    paths = raw.get("paths", {})
    sampling = raw.get("sampling", {})
    scraping = raw.get("scraping", {})
    filtering = raw.get("filtering", {})

    return AcquisitionConfig(
        metadata_path=Path(paths["metadata_path"]),
        metadata_period_dir=Path(paths["metadata_period_dir"]) if paths.get("metadata_period_dir") else None,
        candidates_path=Path(paths["candidates_path"]),
        articles_path=Path(paths["articles_path"]),
        status_path=Path(paths["status_path"]),
        attempts_path=Path(paths["attempts_path"]),
        target_usable_per_month=int(sampling.get("target_usable_per_month", 10000)),
        candidate_multiplier=float(sampling.get("candidate_multiplier", 3.0)),
        candidate_mode=str(sampling.get("candidate_mode", "sample")),
        random_seed=int(sampling.get("random_seed", 42)),
        save_every=int(scraping.get("save_every", 100)),
        min_text_chars=int(filtering.get("min_text_chars", 500)),
        max_attempts=int(scraping.get("max_attempts", 2)),
        workers=int(scraping.get("workers", 8)),
        download_timeout_seconds=int(scraping.get("download_timeout_seconds", 10)),
        min_free_disk_gb=float(scraping.get("min_free_disk_gb", 0.0)),
        stop_at_target=bool(scraping.get("stop_at_target", True)),
        start_date=sampling.get("start_date"),
        end_date=sampling.get("end_date"),
    )


def read_parquet_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Atomically write parquet so interrupted writes are never treated as complete."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    df.to_parquet(temp_path, index=False)
    temp_path.replace(path)


def resolve_month_path(path: Path, month: str | None) -> Path:
    if month is None:
        return path
    return Path(str(path).format(month=month))


def month_config(cfg: AcquisitionConfig, month: str | None) -> AcquisitionConfig:
    if month is None:
        return cfg
    return replace(
        cfg,
        candidates_path=resolve_month_path(cfg.candidates_path, month),
        articles_path=resolve_month_path(cfg.articles_path, month),
        status_path=resolve_month_path(cfg.status_path, month),
        attempts_path=resolve_month_path(cfg.attempts_path, month),
    )


def stable_url_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def stable_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def desired_candidate_count(cfg: AcquisitionConfig) -> int:
    return max(1, int(cfg.target_usable_per_month * cfg.candidate_multiplier))


def validate_acquisition_config(cfg: AcquisitionConfig) -> None:
    if cfg.candidate_mode not in {"sample", "all"}:
        raise ValueError("sampling.candidate_mode must be either 'sample' or 'all'")
    if cfg.min_free_disk_gb < 0:
        raise ValueError("scraping.min_free_disk_gb must be at least 0")


def free_disk_gb(path: Path) -> float:
    """Return free disk space for a path whose output directory may not exist yet."""

    existing = path
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    return shutil.disk_usage(existing).free / (1024**3)


def require_free_disk(cfg: AcquisitionConfig) -> None:
    if cfg.min_free_disk_gb <= 0:
        return
    output_paths = (cfg.articles_path, cfg.status_path, cfg.attempts_path)
    free_by_path = [(path, free_disk_gb(path)) for path in output_paths]
    lowest_path, lowest_free = min(free_by_path, key=lambda item: item[1])
    if lowest_free < cfg.min_free_disk_gb:
        raise RuntimeError(
            f"Stopping acquisition: {lowest_free:.1f} GiB free at {lowest_path}, "
            f"below scraping.min_free_disk_gb={cfg.min_free_disk_gb:.1f}. "
            "Saved checkpoints can be resumed after freeing or expanding disk space."
        )


def normalize_metadata(metadata: pd.DataFrame, cfg: AcquisitionConfig) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in metadata.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    df = metadata.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    df = df.dropna(subset=["timestamp", "url"])
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].str.startswith(("http://", "https://"), na=False)]

    if cfg.start_date:
        df = df[df["timestamp"] >= pd.Timestamp(cfg.start_date)]
    if cfg.end_date:
        end = pd.Timestamp(cfg.end_date)
        if " " not in cfg.end_date and "T" not in cfg.end_date and len(cfg.end_date) <= 10:
            end = end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        df = df[df["timestamp"] <= end]

    df = df.drop_duplicates(subset=["url"])
    df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["url_id"] = df["url"].map(stable_url_id)
    return df.reset_index(drop=True)


def month_period_paths(period_dir: Path, month: str) -> list[Path]:
    prefix = month.replace("-", "")
    return sorted(period_dir.glob(f"{prefix}*.parquet"))


def available_period_months(period_dir: Path) -> list[str]:
    months = {f"{path.stem[:4]}-{path.stem[4:6]}" for path in period_dir.glob("*.parquet") if len(path.stem) >= 6}
    return sorted(months)


def month_from_date(value: str, *, is_end: bool = False) -> str:
    ts = pd.Timestamp(value)
    if is_end and " " not in value and "T" not in value and len(value) <= 10:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return str(ts.to_period("M"))


def configured_months(cfg: AcquisitionConfig) -> list[str]:
    if cfg.metadata_period_dir and cfg.metadata_period_dir.exists():
        months = available_period_months(cfg.metadata_period_dir)
    elif cfg.start_date and cfg.end_date:
        months = [
            str(month)
            for month in pd.period_range(
                month_from_date(cfg.start_date),
                month_from_date(cfg.end_date, is_end=True),
                freq="M",
            )
        ]
    else:
        raise ValueError("Cannot infer months without paths.metadata_period_dir or sampling.start_date/sampling.end_date")

    if cfg.start_date:
        months = [month for month in months if month >= month_from_date(cfg.start_date)]
    if cfg.end_date:
        months = [month for month in months if month <= month_from_date(cfg.end_date, is_end=True)]
    if not months:
        raise ValueError("No months found for the configured metadata source and date range")
    return months


def select_months(cfg: AcquisitionConfig, start_month: str | None = None, end_month: str | None = None) -> list[str]:
    months = configured_months(cfg)
    if start_month:
        months = [month for month in months if month >= start_month]
    if end_month:
        months = [month for month in months if month <= end_month]
    if not months:
        raise ValueError("No months selected")
    return months


def has_month_placeholder(path: Path) -> bool:
    return "{month}" in str(path)


def require_month_placeholders(cfg: AcquisitionConfig) -> None:
    paths = {
        "paths.candidates_path": cfg.candidates_path,
        "paths.articles_path": cfg.articles_path,
        "paths.status_path": cfg.status_path,
        "paths.attempts_path": cfg.attempts_path,
    }
    missing = [name for name, path in paths.items() if not has_month_placeholder(path)]
    if missing:
        raise ValueError("All-month commands require {month} placeholders in: " + ", ".join(missing))


def read_metadata_source(cfg: AcquisitionConfig, month: str | None) -> pd.DataFrame:
    if month and cfg.metadata_period_dir:
        paths = month_period_paths(cfg.metadata_period_dir, month)
        if paths:
            return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)

    if cfg.metadata_path.exists():
        metadata = pd.read_parquet(cfg.metadata_path)
        if month:
            timestamps = pd.to_datetime(metadata["timestamp"], errors="coerce")
            return metadata[timestamps.dt.strftime("%Y-%m") == month].reset_index(drop=True)
        return metadata

    if cfg.metadata_period_dir and cfg.metadata_period_dir.exists():
        if month:
            raise ValueError(f"No daily metadata parquet files found for {month} in {cfg.metadata_period_dir}")
        months = ", ".join(available_period_months(cfg.metadata_period_dir)[:6])
        suffix = " ..." if len(available_period_months(cfg.metadata_period_dir)) > 6 else ""
        raise ValueError(
            "No metadata file found. Run with --month to sample directly from daily parquet files "
            f"in {cfg.metadata_period_dir}. Available months include: {months}{suffix}"
        )

    raise ValueError(f"No metadata found at {cfg.metadata_path}")


def prepare_candidates(cfg: AcquisitionConfig, month: str | None = None, *, overwrite: bool = False) -> pd.DataFrame:
    cfg = month_config(cfg, month)
    validate_acquisition_config(cfg)
    if cfg.candidates_path.exists() and not overwrite:
        existing = read_parquet_if_exists(cfg.candidates_path)
        if candidate_checkpoint_complete(existing, cfg):
            print(f"Candidate checkpoint already exists at {cfg.candidates_path}; skipping")
            return existing
    metadata = read_metadata_source(cfg, month)
    normalized = normalize_metadata(metadata, cfg)
    per_month = desired_candidate_count(cfg)

    sampled = []
    for year_month, month_df in normalized.groupby("year_month", sort=True):
        if cfg.candidate_mode == "all":
            n = len(month_df)
            month_sample = month_df.copy()
            mode_label = "selected all"
        else:
            n = min(per_month, len(month_df))
            month_sample = month_df.sample(n=n, random_state=cfg.random_seed)
            mode_label = "sampled"
        month_sample = month_sample.copy()
        month_sample["candidate_sample_size"] = n
        month_sample["candidate_pool_size"] = len(month_df)
        month_sample["candidate_mode"] = cfg.candidate_mode
        month_sample["sampled_at"] = utc_now()
        sampled.append(month_sample)
        print(f"{year_month}: {mode_label} {n:,} of {len(month_df):,} candidate URLs")

    if not sampled:
        raise ValueError("No candidate URLs found after metadata normalization")

    candidates = pd.concat(sampled, ignore_index=True)
    write_parquet(candidates, cfg.candidates_path)
    return candidates


def exclude_existing_urls(candidates: pd.DataFrame, status: pd.DataFrame) -> set[str]:
    urls: set[str] = set()
    if not candidates.empty and "url" in candidates.columns:
        urls.update(candidates["url"].dropna().astype(str))
    if not status.empty and "url" in status.columns:
        urls.update(status["url"].dropna().astype(str))
    return urls


def top_up_candidates(cfg: AcquisitionConfig, month: str, extra_count: int | None = None) -> pd.DataFrame:
    validate_acquisition_config(cfg)
    if extra_count is not None and extra_count < 1:
        raise ValueError("--extra-candidates must be at least 1 when provided")
    cfg = month_config(cfg, month)
    existing = read_parquet_if_exists(cfg.candidates_path)
    status = read_parquet_if_exists(cfg.status_path)
    if cfg.candidate_mode == "all":
        print(f"{month}: candidate_mode=all; use prepare-candidates to rebuild the full candidate file")
        return existing
    if cfg.stop_at_target and success_count(status) >= cfg.target_usable_per_month:
        print(f"{month}: English target already met; no candidate top-up needed")
        return existing

    current_count = len(existing)
    desired_count = current_count + extra_count if extra_count is not None else desired_candidate_count(cfg)
    if current_count >= desired_count:
        print(f"{month}: candidate pool already has {current_count:,}/{desired_count:,} URLs")
        return existing

    metadata = read_metadata_source(cfg, month)
    normalized = normalize_metadata(metadata, cfg)
    normalized = normalized[normalized["year_month"] == month].reset_index(drop=True)
    excluded_urls = exclude_existing_urls(existing, status)
    pool = normalized[~normalized["url"].astype(str).isin(excluded_urls)].reset_index(drop=True)
    needed = desired_count - current_count
    n = min(needed, len(pool))
    if n <= 0:
        print(
            f"{month}: no unused candidate URLs available "
            f"(existing={current_count:,}, excluded={len(excluded_urls):,}, desired={desired_count:,})"
        )
        return existing

    seed = stable_int(f"{cfg.random_seed}:{month}:top-up:{current_count}:{desired_count}:{len(excluded_urls)}")
    top_up = pool.sample(n=n, random_state=seed).copy()
    top_up["candidate_sample_size"] = n
    top_up["candidate_pool_size"] = len(pool)
    top_up["sampled_at"] = utc_now()
    top_up["candidate_batch"] = "top_up"

    if existing.empty:
        updated = top_up.reset_index(drop=True)
    else:
        updated = pd.concat([existing, top_up], ignore_index=True)
        updated = updated.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    write_parquet(updated, cfg.candidates_path)
    print(
        f"{month}: added {len(updated) - current_count:,} candidate URLs "
        f"({current_count:,} -> {len(updated):,}; desired {desired_count:,})"
    )
    if n < needed:
        print(f"{month}: unused metadata pool exhausted before desired candidate count")
    return updated


def top_up_candidates_for_months(cfg: AcquisitionConfig, months: list[str], extra_count: int | None = None) -> None:
    require_month_placeholders(cfg)
    for index, month in enumerate(months, start=1):
        print(f"\n[{index}/{len(months)}] top-up-candidates {month}")
        top_up_candidates(cfg, month, extra_count=extra_count)


def prepare_candidates_for_months(cfg: AcquisitionConfig, months: list[str], *, overwrite: bool = False) -> None:
    require_month_placeholders(cfg)
    for index, month in enumerate(months, start=1):
        print(f"\n[{index}/{len(months)}] prepare-candidates {month}")
        prepare_candidates(cfg, month, overwrite=overwrite)


def candidates_match_mode(candidates: pd.DataFrame, cfg: AcquisitionConfig) -> bool:
    if cfg.candidate_mode != "all":
        return True
    return not candidates.empty and "candidate_mode" in candidates.columns and set(candidates["candidate_mode"].dropna()) == {"all"}


def candidate_checkpoint_complete(candidates: pd.DataFrame, cfg: AcquisitionConfig) -> bool:
    """Return whether an existing candidate file satisfies current preparation settings."""

    if not candidates_match_mode(candidates, cfg):
        return False
    if cfg.candidate_mode == "all":
        return True
    return len(candidates) >= desired_candidate_count(cfg)


def ensure_candidates(cfg: AcquisitionConfig, month: str) -> None:
    validate_acquisition_config(cfg)
    month_cfg = month_config(cfg, month)
    if month_cfg.candidates_path.exists():
        candidates = read_parquet_if_exists(month_cfg.candidates_path)
        if not candidates_match_mode(candidates, month_cfg):
            print(f"{month}: rebuilding candidates for candidate_mode={month_cfg.candidate_mode}")
            prepare_candidates(cfg, month)
            return
        status = read_parquet_if_exists(month_cfg.status_path)
        if month_cfg.candidate_mode == "sample" and (
            not month_cfg.stop_at_target or success_count(status) < month_cfg.target_usable_per_month
        ):
            top_up_candidates(cfg, month)
        else:
            print(f"{month}: using existing candidates at {month_cfg.candidates_path}")
        return
    prepare_candidates(cfg, month)


def latest_status(status: pd.DataFrame) -> pd.DataFrame:
    if status.empty:
        return status
    ordered = status.sort_values(["url", "last_attempt_at"])
    return ordered.drop_duplicates(subset=["url"], keep="last")


def success_count(status: pd.DataFrame) -> int:
    if status.empty or "status" not in status.columns:
        return 0
    return int((latest_status(status)["status"] == "success").sum())


def acquisition_status_counts(status: pd.DataFrame) -> dict[str, int]:
    latest = latest_status(status)
    if latest.empty or "status" not in latest.columns:
        return {"success": 0, "non_english": 0, "too_short": 0, "failed": 0}
    counts = latest["status"].value_counts()
    return {
        "success": int(counts.get("success", 0)),
        "non_english": int(counts.get("non_english", 0)),
        "too_short": int(counts.get("too_short", 0)),
        "failed": int(counts.get("download_failed", 0) + counts.get("extract_failed", 0) + counts.get("failed", 0)),
    }


def urls_to_process(candidates: pd.DataFrame, status: pd.DataFrame, cfg: AcquisitionConfig) -> pd.DataFrame:
    if status.empty:
        candidates = candidates.copy()
        candidates["_attempt_count"] = 0
        return candidates

    candidates = candidates.copy()
    candidates["_candidate_order"] = range(len(candidates))
    latest = latest_status(status)[["url", "status", "attempt_count"]]
    merged = candidates.merge(latest, on="url", how="left")
    merged["attempt_count"] = merged["attempt_count"].fillna(0).astype(int)
    terminal = merged["status"].isin(TERMINAL_STATUSES)
    under_attempt_limit = merged["attempt_count"] < cfg.max_attempts
    queued = merged[~terminal & under_attempt_limit].copy()
    queued = queued.sort_values(["attempt_count", "_candidate_order"], kind="stable")
    return queued.drop(columns=["status", "_candidate_order"]).rename(columns={"attempt_count": "_attempt_count"})


def make_detector() -> Any:
    from lingua import LanguageDetectorBuilder

    return LanguageDetectorBuilder.from_all_languages().build()


def detect_language(text: str, detector: Any) -> str | None:
    from lingua import Language

    detected = detector.detect_language_of(text)
    if detected is None:
        return None
    if detected == Language.ENGLISH:
        return "en"
    try:
        return detected.iso_code_639_1.name.lower()
    except AttributeError:
        return detected.name.lower()


def trafilatura_config(download_timeout_seconds: int) -> ConfigParser:
    from trafilatura.settings import DEFAULT_CONFIG

    config = ConfigParser()
    config["DEFAULT"] = dict(DEFAULT_CONFIG.defaults())
    config["DEFAULT"]["DOWNLOAD_TIMEOUT"] = str(download_timeout_seconds)
    return config


def classify_url_error(reason: Any) -> str:
    if isinstance(reason, socket.gaierror):
        return "dns_error"
    if isinstance(reason, socket.timeout | TimeoutError):
        return "timeout"
    if isinstance(reason, ssl.SSLError):
        return "ssl_error"
    reason_text = str(reason).lower()
    if "timed out" in reason_text or "timeout" in reason_text:
        return "timeout"
    if "name or service not known" in reason_text or "temporary failure in name resolution" in reason_text:
        return "dns_error"
    if "certificate" in reason_text or "ssl" in reason_text:
        return "ssl_error"
    if "connection refused" in reason_text:
        return "connection_refused"
    if "connection reset" in reason_text:
        return "connection_reset"
    return "url_error"


def diagnose_download_failure(url: str, timeout_seconds: int) -> DownloadDiagnosticError:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ai-changed-news-article-acquisition/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response.read(1024)
            status = getattr(response, "status", None)
            final_url = response.geturl()
        return DownloadDiagnosticError(
            "trafilatura_rejected_response",
            "trafilatura.fetch_url returned no usable HTML after an HTTP response",
            http_status=int(status) if status is not None else None,
            final_url=final_url,
        )
    except urllib.error.HTTPError as exc:
        return DownloadDiagnosticError(
            f"http_{exc.code}",
            str(exc),
            http_status=int(exc.code),
            final_url=exc.geturl(),
        )
    except urllib.error.URLError as exc:
        reason = classify_url_error(exc.reason)
        return DownloadDiagnosticError(reason, str(exc), final_url=url)
    except socket.timeout as exc:
        return DownloadDiagnosticError("timeout", str(exc), final_url=url)
    except ssl.SSLError as exc:
        return DownloadDiagnosticError("ssl_error", str(exc), final_url=url)
    except Exception as exc:  # noqa: BLE001
        return DownloadDiagnosticError("download_exception", f"{type(exc).__name__}: {exc}", final_url=url)


def download_html(url: str, timeout_seconds: int, max_bytes: int = 20_000_000) -> tuple[str, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ai-changed-news-article-acquisition/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            data = bytearray()
            while True:
                chunk = response.read(2**17)
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > max_bytes:
                    raise DownloadDiagnosticError(
                        "max_file_size_exceeded",
                        f"download exceeded {max_bytes} bytes",
                        http_status=getattr(response, "status", None),
                        final_url=response.geturl(),
                    )
            final_url = response.geturl()
            encoding = response.headers.get_content_charset() or "utf-8"
        return bytes(data).decode(encoding, errors="replace"), final_url
    except DownloadDiagnosticError:
        raise
    except urllib.error.HTTPError as exc:
        raise DownloadDiagnosticError(
            f"http_{exc.code}",
            str(exc),
            http_status=int(exc.code),
            final_url=exc.geturl(),
        ) from exc
    except urllib.error.URLError as exc:
        raise DownloadDiagnosticError(classify_url_error(exc.reason), str(exc), final_url=url) from exc
    except socket.timeout as exc:
        raise DownloadDiagnosticError("timeout", str(exc), final_url=url) from exc
    except ssl.SSLError as exc:
        raise DownloadDiagnosticError("ssl_error", str(exc), final_url=url) from exc
    except Exception as exc:
        raise DownloadDiagnosticError("download_exception", f"{type(exc).__name__}: {exc}", final_url=url) from exc


def extract_article_text_from_html(url: str, downloaded: str) -> str:
    import trafilatura

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        url=url,
    )
    if not text:
        raise ArticleAcquisitionError("extract", "extract_failed")
    return text.strip()


def build_attempt(row: pd.Series, status: str, attempt_count: int, error: Exception | None = None) -> dict[str, Any]:
    return {
        "url": row["url"],
        "url_id": row.get("url_id", stable_url_id(row["url"])),
        "timestamp": row["timestamp"],
        "source": row.get("source"),
        "year_month": row.get("year_month"),
        "status": status,
        "stage": getattr(error, "stage", status),
        "attempt_count": attempt_count,
        "attempted_at": utc_now(),
        "error_type": type(error).__name__ if error else None,
        "error_message": str(error) if error else None,
        "failure_reason": getattr(error, "failure_reason", None),
        "http_status": getattr(error, "http_status", None),
        "final_url": getattr(error, "final_url", None),
    }


def build_status(attempt: dict[str, Any], language: str | None, text_chars: int | None) -> dict[str, Any]:
    return {
        "url": attempt["url"],
        "url_id": attempt["url_id"],
        "timestamp": attempt["timestamp"],
        "source": attempt["source"],
        "year_month": attempt["year_month"],
        "status": attempt["status"],
        "stage": attempt["stage"],
        "language": language,
        "text_chars": text_chars,
        "attempt_count": attempt["attempt_count"],
        "last_attempt_at": attempt["attempted_at"],
        "last_error_type": attempt["error_type"],
        "last_error_message": attempt["error_message"],
        "failure_reason": attempt.get("failure_reason"),
        "http_status": attempt.get("http_status"),
        "final_url": attempt.get("final_url"),
    }


def download_candidate(row: pd.Series, cfg: AcquisitionConfig) -> tuple[pd.Series, str | None, Exception | None]:
    try:
        downloaded, _ = download_html(row["url"], cfg.download_timeout_seconds)
        return row, downloaded, None
    except ArticleAcquisitionError as exc:
        return row, None, exc
    except Exception as exc:  # noqa: BLE001
        return row, None, exc


def process_candidate_result(
    row: pd.Series,
    downloaded: str | None,
    download_error: Exception | None,
    cfg: AcquisitionConfig,
    detector: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    attempt_count = int(row.get("_attempt_count", 0)) + 1
    language = None
    text_chars = None
    article = None

    try:
        if download_error is not None:
            raise download_error
        if downloaded is None:
            raise ArticleAcquisitionError("download", "download_failed")

        article_text = extract_article_text_from_html(row["url"], downloaded)
        text_chars = len(article_text)

        if text_chars < cfg.min_text_chars:
            attempt = build_attempt(row, "too_short", attempt_count)
        else:
            language = detect_language(article_text, detector)
            if language == "en":
                attempt = build_attempt(row, "success", attempt_count)
                article = row.drop(labels=["_attempt_count"], errors="ignore").to_dict()
                article["article_text"] = article_text
                article["language"] = language
                article["text_chars"] = text_chars
                article["extracted_at"] = attempt["attempted_at"]
            else:
                attempt = build_attempt(row, "non_english", attempt_count)
    except ArticleAcquisitionError as exc:
        attempt = build_attempt(row, exc.stage + "_failed", attempt_count, exc)
    except Exception as exc:  # noqa: BLE001
        attempt = build_attempt(row, "failed", attempt_count, exc)

    return article, attempt, build_status(attempt, language, text_chars)


def process_download_failure_diagnostic(row: pd.Series, cfg: AcquisitionConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    attempt_count = int(row.get("attempt_count", 1) or 1)
    error = diagnose_download_failure(str(row["url"]), cfg.download_timeout_seconds)
    attempt = build_attempt(row, "download_failed", attempt_count, error)
    attempt["stage"] = "download_diagnosis"
    status = build_status(attempt, None, None)
    status["stage"] = "download_diagnosis"
    return attempt, status


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    existing = read_parquet_if_exists(path)
    updated = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    write_parquet(updated, path)


def append_articles(path: Path, rows: list[dict[str, Any]], batch_id: str | None = None) -> None:
    if not rows:
        return
    if path.suffix == ".parquet":
        append_rows(path, rows)
        return
    path.mkdir(parents=True, exist_ok=True)
    batch = batch_id or utc_now().replace(":", "").replace("+", "")
    write_parquet(pd.DataFrame(rows), path / f"articles_{batch}.parquet")


def update_current_status(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    existing = read_parquet_if_exists(path)
    updated = latest_status(pd.concat([existing, pd.DataFrame(rows)], ignore_index=True))
    write_parquet(updated, path)


def acquire_articles(cfg: AcquisitionConfig, month: str | None = None) -> None:
    cfg = month_config(cfg, month)
    validate_acquisition_config(cfg)
    require_free_disk(cfg)
    candidates = pd.read_parquet(cfg.candidates_path)
    if month and "year_month" in candidates.columns:
        candidates = candidates[candidates["year_month"] == month].reset_index(drop=True)
    status = read_parquet_if_exists(cfg.status_path)
    live_counts = acquisition_status_counts(status)
    successful_urls = live_counts["success"]
    if cfg.stop_at_target and successful_urls >= cfg.target_usable_per_month:
        print(
            f"English target already met: {successful_urls:,}/{cfg.target_usable_per_month:,}; "
            "skipping remaining candidates"
        )
        return

    pending = urls_to_process(candidates, status, cfg)
    detector = make_detector()

    print(f"Pending URLs to process: {len(pending):,}")
    if cfg.stop_at_target:
        print(f"English successes: {successful_urls:,}/{cfg.target_usable_per_month:,}")
    else:
        print(f"English successes: {successful_urls:,}; stop_at_target=false")
    print(
        f"workers={cfg.workers:,}, download_timeout_seconds={cfg.download_timeout_seconds:,}, "
        f"candidate_mode={cfg.candidate_mode}, stop_at_target={cfg.stop_at_target}"
    )
    socket.setdefaulttimeout(cfg.download_timeout_seconds)
    article_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    processed_count = 0

    def save_progress() -> None:
        nonlocal article_rows, attempt_rows, status_rows
        if not attempt_rows and not status_rows and not article_rows:
            return
        batch_id = f"{utc_now().replace(':', '').replace('+', '')}_{processed_count:08d}"
        append_articles(cfg.articles_path, article_rows, batch_id=batch_id)
        append_rows(cfg.attempts_path, attempt_rows)
        update_current_status(cfg.status_path, status_rows)
        print(f"Saved progress after {processed_count:,} URLs")
        article_rows.clear()
        attempt_rows.clear()
        status_rows.clear()
        require_free_disk(cfg)

    pending_iter = (row for _, row in pending.iterrows())
    max_workers = max(1, cfg.workers)

    def desired_in_flight() -> int:
        if not cfg.stop_at_target:
            return max_workers
        remaining_successes = max(0, cfg.target_usable_per_month - successful_urls)
        if remaining_successes == 0:
            return 0
        return min(max_workers, max(1, remaining_successes * 2))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: set[Future[tuple[pd.Series, str | None, Exception | None]]] = set()
        for _ in range(min(desired_in_flight(), len(pending))):
            try:
                futures.add(executor.submit(download_candidate, next(pending_iter), cfg))
            except StopIteration:
                break

        with progress(total=len(pending), desc=f"URLs {month or 'all'}", unit="url", dynamic_ncols=True) as bar:
            bar.set_postfix(
                ok=live_counts["success"],
                non_en=live_counts["non_english"],
                short=live_counts["too_short"],
                failed=live_counts["failed"],
            )
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    row, downloaded, download_error = future.result()
                    article, attempt, current_status = process_candidate_result(
                        row,
                        downloaded,
                        download_error,
                        cfg,
                        detector,
                    )
                    if article:
                        article_rows.append(article)
                    status_name = attempt["status"]
                    if status_name == "success":
                        successful_urls += 1
                        live_counts["success"] += 1
                    elif status_name == "non_english":
                        live_counts["non_english"] += 1
                    elif status_name == "too_short":
                        live_counts["too_short"] += 1
                    elif status_name in {"download_failed", "extract_failed", "failed"}:
                        live_counts["failed"] += 1
                    attempt_rows.append(attempt)
                    status_rows.append(current_status)
                    processed_count += 1
                    bar.update(1)
                    bar.set_postfix(
                        ok=live_counts["success"],
                        non_en=live_counts["non_english"],
                        short=live_counts["too_short"],
                        failed=live_counts["failed"],
                    )

                    while len(futures) < desired_in_flight():
                        try:
                            futures.add(executor.submit(download_candidate, next(pending_iter), cfg))
                        except StopIteration:
                            break

                    if processed_count % cfg.save_every == 0:
                        save_progress()

    save_progress()
    if cfg.stop_at_target and successful_urls >= cfg.target_usable_per_month:
        print(f"English target met: {successful_urls:,}/{cfg.target_usable_per_month:,}")


def acquire_articles_for_months(cfg: AcquisitionConfig, months: list[str]) -> None:
    require_month_placeholders(cfg)
    for index, month in enumerate(months, start=1):
        month_cfg = month_config(cfg, month)
        if not month_cfg.candidates_path.exists():
            raise FileNotFoundError(f"No candidates found for {month}: {month_cfg.candidates_path}")
        print(f"\n[{index}/{len(months)}] acquire {month}")
        acquire_articles(cfg, month)


def download_failures_to_diagnose(status: pd.DataFrame, rediagnose: bool) -> pd.DataFrame:
    latest = latest_status(status)
    if latest.empty or "status" not in latest.columns:
        return pd.DataFrame()
    failures = latest[latest["status"] == "download_failed"].copy()
    if failures.empty or rediagnose:
        return failures.reset_index(drop=True)
    if "failure_reason" not in failures.columns:
        return failures.reset_index(drop=True)
    return failures[failures["failure_reason"].isna()].reset_index(drop=True)


def diagnose_download_failures(cfg: AcquisitionConfig, month: str | None = None, *, rediagnose: bool = False) -> None:
    cfg = month_config(cfg, month)
    status = read_parquet_if_exists(cfg.status_path)
    failures = download_failures_to_diagnose(status, rediagnose)
    if failures.empty:
        print(f"No download failures need diagnosis at {cfg.status_path}.")
        return

    print(f"Download failures to diagnose: {len(failures):,}")
    print(f"workers={cfg.workers:,}, download_timeout_seconds={cfg.download_timeout_seconds:,}")
    attempt_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    processed_count = 0

    def save_progress() -> None:
        nonlocal attempt_rows, status_rows
        if not attempt_rows and not status_rows:
            return
        append_rows(cfg.attempts_path, attempt_rows)
        update_current_status(cfg.status_path, status_rows)
        print(f"Saved diagnostic progress after {processed_count:,} URLs")
        attempt_rows.clear()
        status_rows.clear()

    pending_iter = (row for _, row in failures.iterrows())
    max_workers = max(1, cfg.workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: set[Future[tuple[dict[str, Any], dict[str, Any]]]] = set()
        for _ in range(min(max_workers, len(failures))):
            try:
                futures.add(executor.submit(process_download_failure_diagnostic, next(pending_iter), cfg))
            except StopIteration:
                break

        with progress(total=len(failures), desc=f"Diagnose {month or 'all'}", unit="url", dynamic_ncols=True) as bar:
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    attempt, current_status = future.result()
                    attempt_rows.append(attempt)
                    status_rows.append(current_status)
                    processed_count += 1
                    bar.update(1)

                    try:
                        futures.add(executor.submit(process_download_failure_diagnostic, next(pending_iter), cfg))
                    except StopIteration:
                        pass

                    if processed_count % cfg.save_every == 0:
                        save_progress()

    save_progress()
    refreshed = latest_status(read_parquet_if_exists(cfg.status_path))
    diagnosed = refreshed[refreshed["status"] == "download_failed"] if not refreshed.empty else pd.DataFrame()
    if "failure_reason" in diagnosed.columns and not diagnosed.empty:
        print("Download failure reason counts")
        print(diagnosed["failure_reason"].fillna("unknown").value_counts().head(20).to_string())


def diagnose_download_failures_for_months(cfg: AcquisitionConfig, months: list[str], *, rediagnose: bool = False) -> None:
    require_month_placeholders(cfg)
    for index, month in enumerate(months, start=1):
        print(f"\n[{index}/{len(months)}] diagnose-failures {month}")
        diagnose_download_failures(cfg, month, rediagnose=rediagnose)


def summarize_status(cfg: AcquisitionConfig, month: str | None = None) -> None:
    cfg = month_config(cfg, month)
    status = latest_status(read_parquet_if_exists(cfg.status_path))
    if status.empty:
        print(f"No status file found yet at {cfg.status_path}.")
        return

    candidates = read_parquet_if_exists(cfg.candidates_path)
    if month and "year_month" in candidates.columns:
        candidates = candidates[candidates["year_month"] == month].reset_index(drop=True)
    if month and "year_month" in status.columns:
        status = status[status["year_month"] == month].reset_index(drop=True)
    candidate_counts = (
        candidates.groupby("year_month", dropna=False).size().rename("candidate_urls").reset_index()
        if not candidates.empty and "year_month" in candidates.columns
        else pd.DataFrame(columns=["year_month", "candidate_urls"])
    )

    counts = status.pivot_table(index="year_month", columns="status", values="url", aggfunc="count", fill_value=0)
    counts = counts.reset_index().rename_axis(None, axis=1)
    for column in ["success", "non_english", "too_short", "download_failed", "extract_failed", "failed"]:
        if column not in counts.columns:
            counts[column] = 0
    counts["attempted_urls"] = counts.drop(columns=["year_month"]).sum(axis=1)
    counts = counts.merge(candidate_counts, on="year_month", how="outer").fillna(0)
    counts["target_usable"] = cfg.target_usable_per_month
    counts["english_target_pct"] = 100 * counts["success"] / counts["target_usable"]

    pending_rows = []
    if not candidates.empty and "year_month" in candidates.columns:
        for year_month, month_candidates in candidates.groupby("year_month", sort=True):
            month_status = status[status["year_month"] == year_month] if "year_month" in status.columns else status
            if cfg.stop_at_target and success_count(month_status) >= cfg.target_usable_per_month:
                pending_or_retryable = 0
            else:
                pending_or_retryable = len(urls_to_process(month_candidates, month_status, cfg))
            pending_rows.append({"year_month": year_month, "pending_or_retryable": pending_or_retryable})
    pending_counts = pd.DataFrame(pending_rows) if pending_rows else pd.DataFrame(columns=["year_month", "pending_or_retryable"])
    counts = counts.drop(columns=["pending_or_retryable"], errors="ignore").merge(
        pending_counts,
        on="year_month",
        how="left",
    )
    counts["pending_or_retryable"] = counts["pending_or_retryable"].fillna(0)

    columns = [
        "year_month",
        "candidate_urls",
        "attempted_urls",
        "success",
        "english_target_pct",
        "non_english",
        "too_short",
        "download_failed",
        "extract_failed",
        "failed",
        "pending_or_retryable",
    ]
    print(counts[columns].sort_values("year_month").to_string(index=False, float_format=lambda value: f"{value:.1f}"))
    if "failure_reason" in status.columns:
        reasons = status[status["status"] == "download_failed"]["failure_reason"].dropna()
        if not reasons.empty:
            print()
            print("Download failure reasons")
            print(reasons.value_counts().head(12).to_string())


def summarize_status_for_months(cfg: AcquisitionConfig, months: list[str]) -> None:
    require_month_placeholders(cfg)
    for index, month in enumerate(months, start=1):
        print(f"\n[{index}/{len(months)}] status {month}")
        summarize_status(cfg, month)


def run_months(cfg: AcquisitionConfig, months: list[str]) -> None:
    require_month_placeholders(cfg)
    for index, month in enumerate(months, start=1):
        print(f"\n[{index}/{len(months)}] run-month {month}")
        ensure_candidates(cfg, month)
        acquire_articles(cfg, month)
        summarize_status(cfg, month)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Article acquisition pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Commands:
  list-months             Print months available for article acquisition.
  prepare-candidates      Sample candidate article URLs from the configured metadata source.
  prepare-candidates-all  Sample candidate article URLs for every selected month.
  top-up-candidates       Append unused candidate URLs to one existing monthly candidate file.
  top-up-candidates-all   Append unused candidate URLs for every selected month.
  acquire                 Download/extract/filter articles for one month.
  acquire-all             Download/extract/filter articles for every selected month.
  diagnose-failures       Recheck existing download_failed URLs and bucket failure reasons.
  diagnose-failures-all   Diagnose download failures for every selected month.
  status                  Summarize English yield for one month.
  status-all              Summarize English yield for every selected month.
  run-months              For every selected month: prepare candidates, acquire articles, print status.

Important options:
  --config PATH       TOML config path. Defaults to configs/article_acquisition.toml.
  --month YYYY-MM     Process one month. Recommended for local VM runs.
  --start-month YYYY-MM and --end-month YYYY-MM narrow all-month commands.
  --extra-candidates N  Add N extra URLs beyond the current candidate count for top-up commands.
  --rediagnose        Recheck download failures even when a failure_reason already exists.
  --overwrite-candidates  Rebuild existing candidate checkpoints.

Config fields:
  paths.metadata_period_dir allows candidate sampling directly from daily parquet files.
  paths with {month} are resolved when --month is provided.
  sampling.target_usable_per_month is the final English-article target, currently 10000.
  sampling.candidate_multiplier controls oversampling before scraping/language filtering.
  sampling.candidate_mode is "sample" for target-based candidates or "all" for every normalized URL.
  scraping.save_every controls resumable checkpoint frequency.
  scraping.max_attempts limits retries for retryable URL failures.
  scraping.workers controls concurrent URL downloads/extractions.
  scraping.download_timeout_seconds caps each trafilatura download.
  scraping.min_free_disk_gb stops safely at a saved checkpoint before storage is exhausted.
  scraping.stop_at_target controls whether acquisition stops after the monthly English target.
  filtering.min_text_chars drops extracted pages that are too short to treat as articles.
""",
    )
    parser.add_argument(
        "command",
        choices=(
            "list-months",
            "prepare-candidates",
            "prepare-candidates-all",
            "top-up-candidates",
            "top-up-candidates-all",
            "acquire",
            "acquire-all",
            "diagnose-failures",
            "diagnose-failures-all",
            "status",
            "status-all",
            "run-months",
        ),
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, type=Path)
    parser.add_argument("--month", help="Process one YYYY-MM month. Supports {month} placeholders in output paths.")
    parser.add_argument("--start-month", help="First YYYY-MM month for all-month commands.")
    parser.add_argument("--end-month", help="Last YYYY-MM month for all-month commands.")
    parser.add_argument("--extra-candidates", type=int, help="Extra URLs to add beyond current candidate count.")
    parser.add_argument("--rediagnose", action="store_true", help="Recheck download failures that already have a failure_reason.")
    parser.add_argument("--overwrite-candidates", action="store_true", help="Rebuild existing candidate checkpoint files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.command == "status" and not args.config.exists():
        cfg = default_config()
    else:
        cfg = load_config(args.config)

    if args.command == "list-months":
        print("\n".join(select_months(cfg, args.start_month, args.end_month)))
    elif args.command == "prepare-candidates":
        prepare_candidates(cfg, args.month, overwrite=args.overwrite_candidates)
    elif args.command == "prepare-candidates-all":
        prepare_candidates_for_months(cfg, select_months(cfg, args.start_month, args.end_month), overwrite=args.overwrite_candidates)
    elif args.command == "top-up-candidates":
        if not args.month:
            raise ValueError("--month is required for top-up-candidates")
        top_up_candidates(cfg, args.month, extra_count=args.extra_candidates)
    elif args.command == "top-up-candidates-all":
        top_up_candidates_for_months(
            cfg,
            select_months(cfg, args.start_month, args.end_month),
            extra_count=args.extra_candidates,
        )
    elif args.command == "acquire":
        acquire_articles(cfg, args.month)
    elif args.command == "acquire-all":
        acquire_articles_for_months(cfg, select_months(cfg, args.start_month, args.end_month))
    elif args.command == "diagnose-failures":
        diagnose_download_failures(cfg, args.month, rediagnose=args.rediagnose)
    elif args.command == "diagnose-failures-all":
        diagnose_download_failures_for_months(
            cfg,
            select_months(cfg, args.start_month, args.end_month),
            rediagnose=args.rediagnose,
        )
    elif args.command == "status":
        summarize_status(cfg, args.month)
    elif args.command == "status-all":
        summarize_status_for_months(cfg, select_months(cfg, args.start_month, args.end_month))
    elif args.command == "run-months":
        run_months(cfg, select_months(cfg, args.start_month, args.end_month))
    else:  # pragma: no cover
        raise ValueError(args.command)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
