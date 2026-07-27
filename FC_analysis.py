from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import importlib.util
import json
import os
import re
import struct
import sys
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.color import gray2rgb
from skimage.exposure import rescale_intensity
from skimage.transform import resize

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


def progress(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(f"[FC] {message}", flush=True)


def format_elapsed(start: float) -> str:
    return f"{time.time() - start:.1f}s"


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "?:??"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


class JobProgress:
    """Small dependency-free progress display for frame/TAR jobs.

    In an interactive terminal, this renders a single updating line. When stdout is
    redirected to a log file, it falls back to low-frequency progress messages so
    the log remains readable.
    """

    def __init__(
        self,
        total: int,
        *,
        quiet: bool = False,
        style: str = "auto",
        log_every: int = 100,
        min_interval_s: float = 0.25,
    ) -> None:
        self.total = max(int(total), 0)
        self.quiet = quiet
        self.style = style
        self.log_every = max(int(log_every), 1)
        self.min_interval_s = float(min_interval_s)
        self.start = time.time()
        self.last_draw = 0.0
        self.last_logged = 0
        self.rendered_bar = False
        if self.style == "auto":
            self.effective_style = "bar" if sys.stdout.isatty() else "log"
        else:
            self.effective_style = self.style

    def update(self, current: int, *, ok: int = 0, failed: int = 0, force: bool = False) -> None:
        if self.quiet or self.effective_style == "none":
            return
        current = max(0, min(int(current), self.total if self.total else int(current)))
        now = time.time()
        if not force and now - self.last_draw < self.min_interval_s and current < self.total:
            return
        elapsed = now - self.start
        rate = current / elapsed if elapsed > 0 and current > 0 else 0.0
        eta = (self.total - current) / rate if rate > 0 and self.total else None
        pct = (100.0 * current / self.total) if self.total else 100.0

        if self.effective_style == "bar":
            width = 32
            filled = int(width * current / self.total) if self.total else width
            bar = "#" * filled + "-" * (width - filled)
            msg = (
                f"[FC] Processing jobs |{bar}| {current}/{self.total} "
                f"{pct:5.1f}% ok={ok} failed={failed} "
                f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
            )
            # Pad to clear remnants from a previous longer line.
            sys.stdout.write("\r" + msg + " " * 8)
            sys.stdout.flush()
            self.rendered_bar = True
        else:
            should_log = force or current == self.total or current - self.last_logged >= self.log_every
            if should_log:
                print(
                    f"[FC] Progress: {current}/{self.total} jobs "
                    f"({pct:.1f}%, ok={ok}, failed={failed}, "
                    f"elapsed={format_duration(elapsed)}, eta={format_duration(eta)})",
                    flush=True,
                )
                self.last_logged = current
        self.last_draw = now

    def finish(self, *, ok: int = 0, failed: int = 0) -> None:
        self.update(self.total, ok=ok, failed=failed, force=True)
        if self.rendered_bar:
            sys.stdout.write("\n")
            sys.stdout.flush()


class UserConfigError(RuntimeError):
    pass


def parquet_writer_available() -> bool:
    """Return True when pandas can write Parquet through pyarrow or fastparquet."""
    return importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None


def require_parquet_writer() -> None:
    """Fail early when default pixel Parquet export cannot be written."""
    if not parquet_writer_available():
        raise UserConfigError(
            "Pixel-level output is enabled by default and is written only as Parquet. "
            "Install a Parquet engine in this environment, for example: pip install pyarrow. "
            "If this run intentionally should not write pixel-level files, use --pixel-export none."
        )


PIXEL_PARQUET_COLUMNS = [
    "root_type",
    "fc_id",
    "frame_id",
    "plant_id",
    "filter_type",
    "dumm_layer",
    "dumm_member",
    "region",
    "y",
    "x",
    "fluorescence",
]


ALIGNMENT_CONFIG: dict[str, dict[str, dict[str, Any]]] = {
    "ROOT1": {
        "F635": dict(target_size=(3006, 4202), resized_shape=(3004, 4199), offset_x=3, offset_y=-12),
        "F483": dict(target_size=(3006, 4202), resized_shape=(3005, 4201), offset_x=-3, offset_y=-11),
        "F513": dict(target_size=(3006, 4202), resized_shape=(3005, 4202), offset_x=0, offset_y=-24),
    },
    "ROOT2": {
        "F635": dict(target_size=(3006, 4202), resized_shape=(3006, 4196), offset_x=5, offset_y=5),
        "F513": dict(target_size=(3006, 4202), resized_shape=(3005, 4202), offset_x=0, offset_y=-8),
        "F593": dict(target_size=(3006, 4202), resized_shape=(3005, 4202), offset_x=0, offset_y=-6),
    },
}


def active_root_types(root_set: str) -> list[str]:
    return ["ROOT1", "ROOT2"] if root_set == "both" else [root_set]


def available_alignment_filters(root_type: str) -> list[str]:
    return sorted(ALIGNMENT_CONFIG[root_type].keys())


def normalize_alignment_filter(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip().upper()
    return value or None


def prompt_for_alignment_filter(root_type: str) -> str:
    fc_label = "FC1" if root_type == "ROOT1" else "FC2"
    choices = available_alignment_filters(root_type)
    print(
        f"[FC] Alignment filter is required for {root_type}/{fc_label}. "
        "No default is used because the wrong filter causes wrong alignment.",
        flush=True,
    )
    print(f"[FC] Available filters for {root_type}: {', '.join(choices)}", flush=True)
    while True:
        value = input(f"[FC] Enter filter for {root_type}/{fc_label} ({', '.join(choices)}): ").strip().upper()
        if not value:
            print("[FC] No default is available. Please enter one of: " + ", ".join(choices), flush=True)
            continue
        if value in ALIGNMENT_CONFIG[root_type]:
            return value
        print(f"[FC] Unknown filter {value!r} for {root_type}. Available: {', '.join(choices)}", flush=True)


def resolve_required_alignment_filters(args: argparse.Namespace, root_set: str) -> dict[str, str]:
    """Resolve and validate per-root alignment filters.

    The filter selects the FC/ROOT alignment configuration. There is intentionally
    no default because a wrong filter can systematically misalign all outputs.
    In an interactive run, missing values are requested from the user. In a
    non-interactive/batch run, missing values are a hard error.
    """
    requested = {
        "ROOT1": normalize_alignment_filter(getattr(args, "filter_fc1", None)),
        "ROOT2": normalize_alignment_filter(getattr(args, "filter_fc2", None)),
    }
    resolved: dict[str, str] = {}
    missing: list[str] = []

    for root_type in active_root_types(root_set):
        value = requested[root_type]
        if value is None:
            if sys.stdin.isatty():
                value = prompt_for_alignment_filter(root_type)
            else:
                missing.append(root_type)
                continue

        if value not in ALIGNMENT_CONFIG[root_type]:
            choices = ", ".join(available_alignment_filters(root_type))
            flag = "--filter-root1/--filter-fc1" if root_type == "ROOT1" else "--filter-root2/--filter-fc2"
            raise UserConfigError(f"Unknown alignment filter {value!r} for {root_type}. Use {flag} with one of: {choices}")
        resolved[root_type] = value

    if missing:
        lines = [
            "Missing required alignment filter(s).",
            "There is no default because the wrong filter causes wrong FC/ROOT alignment.",
        ]
        for root_type in missing:
            choices = ", ".join(available_alignment_filters(root_type))
            flag = "--filter-root1/--filter-fc1" if root_type == "ROOT1" else "--filter-root2/--filter-fc2"
            lines.append(f"  {root_type}: provide {flag} with one of: {choices}")
        raise UserConfigError("\n".join(lines))

    args.filter_fc1 = resolved.get("ROOT1")
    args.filter_fc2 = resolved.get("ROOT2")
    return resolved

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
ROOT_DIR_NAMES = {"ROOT1", "ROOT2"}
ROOT_ANALYSIS_NAMES = {"ROOT1_ANALYSIS", "ROOT2_ANALYSIS"}


@dataclass(frozen=True)
class PlantMaskSet:
    plant_id: str
    plant_dir: str
    root_mask: str


@dataclass(frozen=True)
class FrameJob:
    root_type: str
    filter_type: str
    frame_id: str
    fc_id: str
    frame_dir: str
    tar_path: str
    output_root: str
    plants: tuple[PlantMaskSet, ...]
    pixel_export: str
    overlay: bool
    fc_preview: str
    preview_max_dim: int


@dataclass
class JobResult:
    frame_id: str
    root_type: str
    tar_path: str
    ok: bool
    elapsed_s: float
    plants_processed: int = 0
    dumm_layers_processed: int = 0
    summary_csv_exported: int = 0
    pixel_files_exported: int = 0
    overlay_images_created: int = 0
    fc_preview_images_created: int = 0
    alignment_qc_files_created: int = 0
    error: str | None = None


def normalize_string(value: str) -> str:
    return re.sub(r"[_\-\s]", "", str(value).lower())


def natural_sort_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def clean_tar_filename(filename: str) -> str:
    return re.sub(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_?", "", filename)


def generate_fc_name(frame_id: str) -> str:
    fc_id = re.sub(r"ROOT1.*", "FC1_FcTar", frame_id)
    fc_id = re.sub(r"ROOT2.*", "FC2_FcTar", fc_id)
    return clean_tar_filename(fc_id)


def infer_experiment_context(path: str | Path, root_set: str = "auto") -> tuple[Path, str]:
    p = Path(path).expanduser().resolve()
    name = p.name.upper()
    inferred_root: str | None = None

    if name in ROOT_DIR_NAMES:
        inferred_root = name
        experiment_root = p.parent
    elif name == "MEASUREMENT" and p.parent.name.upper() in ROOT_DIR_NAMES:
        inferred_root = p.parent.name.upper()
        experiment_root = p.parent.parent
    elif name in ROOT_ANALYSIS_NAMES:
        inferred_root = name.replace("_ANALYSIS", "")
        experiment_root = p.parent
    else:
        experiment_root = p

    if root_set.lower() == "auto":
        resolved_root_set = inferred_root or "both"
    elif root_set.lower() == "both":
        resolved_root_set = "both"
    else:
        resolved_root_set = root_set.upper()

    if resolved_root_set not in {"ROOT1", "ROOT2", "both"}:
        raise ValueError("root_set must be auto, ROOT1, ROOT2, or both")
    return experiment_root, resolved_root_set


def fc_input_candidates(experiment_root: Path, fc_name: str) -> list[Path]:
    # Hades data have appeared in both FC2/Measurement/TARs and
    # FC2_TAR/Measurement/TARs. Keep broader fallbacks for copied/test data.
    return [
        experiment_root / fc_name / "Measurement" / "TARs",
        experiment_root / f"{fc_name}_TAR" / "Measurement" / "TARs",
        experiment_root / fc_name / "Measurement",
        experiment_root / f"{fc_name}_TAR" / "Measurement",
        experiment_root / f"{fc_name}_TAR",
        experiment_root / fc_name,
    ]


def first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_root_analysis_dirs(experiment_root: Path, root_set: str, explicit_root_analysis: Path | None = None) -> dict[str, Path]:
    if explicit_root_analysis is not None:
        p = explicit_root_analysis.expanduser().resolve()
        name = p.name.upper()
        if name in ROOT_ANALYSIS_NAMES:
            return {name.replace("_ANALYSIS", ""): p}
        raise ValueError("--root-analysis must point to ROOT1_analysis or ROOT2_analysis")

    out: dict[str, Path] = {}
    roots = ["ROOT1", "ROOT2"] if root_set == "both" else [root_set]
    for root in roots:
        out[root] = experiment_root / f"{root}_analysis"
    return out


def resolve_fc_dirs(experiment_root: Path, root_set: str, explicit_fc_dir: Path | None = None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    roots = ["ROOT1", "ROOT2"] if root_set == "both" else [root_set]
    for root in roots:
        fc_name = "FC1" if root == "ROOT1" else "FC2"
        if explicit_fc_dir is not None and len(roots) == 1:
            out[root] = explicit_fc_dir.expanduser().resolve()
        else:
            found = first_existing(fc_input_candidates(experiment_root, fc_name))
            out[root] = found or fc_input_candidates(experiment_root, fc_name)[0]
    return out


def resolve_output_dirs(experiment_root: Path, root_set: str, explicit_output: Path | None = None) -> dict[str, Path]:
    roots = ["ROOT1", "ROOT2"] if root_set == "both" else [root_set]
    out: dict[str, Path] = {}
    if explicit_output is not None and len(roots) == 1:
        out[roots[0]] = explicit_output.expanduser().resolve()
    else:
        for root in roots:
            fc_name = "FC1" if root == "ROOT1" else "FC2"
            out[root] = experiment_root / f"{fc_name}_analysis"
    return out


def list_tar_files(tar_dir: str | Path, *, quiet: bool = False) -> list[Path]:
    root = Path(tar_dir)
    if not root.exists():
        return []
    start = time.time()
    out: list[Path] = []
    # os.walk lets us emit progress during large recursive TAR directories.
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(".tar"):
                out.append(Path(dirpath) / filename)
                if len(out) % 1000 == 0:
                    progress(f"  indexed {len(out)} TAR files under {root} ({format_elapsed(start)})", quiet=quiet)
    out.sort(key=lambda p: natural_sort_key(str(p)))
    return out


def build_tar_index(tar_files: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in tar_files:
        cleaned_stem = re.sub(r"\.tar$", "", clean_tar_filename(path.name), flags=re.IGNORECASE)
        norm = normalize_string(cleaned_stem)
        index.setdefault(norm, []).append(path)
    return index


def find_matching_tar(fc_id: str, tar_index: dict[str, list[Path]]) -> Path | None:
    target = normalize_string(re.sub(r"\.tar$", "", clean_tar_filename(fc_id), flags=re.IGNORECASE))
    exact = tar_index.get(target)
    if exact and len(exact) == 1:
        return exact[0]
    if exact and len(exact) > 1:
        raise RuntimeError(f"Multiple TAR files exactly matched {fc_id}: {[str(p) for p in exact]}")

    matches: list[Path] = []
    for key, paths in tar_index.items():
        if target in key:
            matches.extend(paths)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple TAR files matched {fc_id}: {[str(p) for p in matches]}")
    return None


def discover_frame_jobs(
    root_analysis_dirs: dict[str, Path],
    fc_dirs: dict[str, Path],
    output_dirs: dict[str, Path],
    filter_fc1: str,
    filter_fc2: str,
    pixel_export: str,
    overlay: bool,
    fc_preview: str,
    preview_max_dim: int,
    quiet: bool = False,
) -> tuple[list[FrameJob], dict[str, Any]]:
    jobs: list[FrameJob] = []
    missing_frames: list[dict[str, str]] = []
    discovered: dict[str, Any] = {"roots": {}}

    for root_type, analysis_dir in root_analysis_dirs.items():
        root_start = time.time()
        fc_dir = fc_dirs[root_type]
        progress(f"Discovering {root_type}: ROOT analysis={analysis_dir}", quiet=quiet)
        progress(f"Indexing FC TAR files for {root_type}: {fc_dir}", quiet=quiet)
        tar_files = list_tar_files(fc_dir, quiet=quiet)
        progress(f"Found {len(tar_files)} TAR file(s) for {root_type} ({format_elapsed(root_start)})", quiet=quiet)
        tar_index = build_tar_index(tar_files)
        filter_type = filter_fc1 if root_type == "ROOT1" else filter_fc2
        discovered["roots"][root_type] = {
            "analysis_dir": str(analysis_dir),
            "fc_dir": str(fc_dir),
            "tar_files": len(tar_files),
            "frame_dirs_scanned": 0,
            "frame_jobs": 0,
            "plant_masks": 0,
            "missing_tar_frames": 0,
        }

        if not analysis_dir.exists():
            progress(f"Skipping {root_type}: analysis directory does not exist", quiet=quiet)
            continue

        progress(f"Scanning frame folders for {root_type}...", quiet=quiet)
        scan_start = time.time()
        frame_dirs: list[Path] = []
        tray_dirs = sorted((p for p in analysis_dir.iterdir() if p.is_dir()), key=lambda p: natural_sort_key(p.name))
        progress(f"  found {len(tray_dirs)} tray folder(s) under {analysis_dir}", quiet=quiet)
        for tray_i, tray_dir in enumerate(tray_dirs, start=1):
            if tray_i == 1 or tray_i % 50 == 0 or tray_i == len(tray_dirs):
                progress(f"  scanning tray {tray_i}/{len(tray_dirs)}: {tray_dir.name}", quiet=quiet)
            for frame_dir in sorted((p for p in tray_dir.iterdir() if p.is_dir()), key=lambda p: natural_sort_key(p.name)):
                if any(child.is_dir() and re.fullmatch(r"plant_\d+", child.name, flags=re.IGNORECASE) for child in frame_dir.iterdir()):
                    frame_dirs.append(frame_dir)

        discovered["roots"][root_type]["frame_dirs_scanned"] = len(frame_dirs)
        progress(f"Found {len(frame_dirs)} frame folder(s) with plant masks for {root_type} ({format_elapsed(scan_start)})", quiet=quiet)
        progress(f"Matching frame IDs to TAR files for {root_type}...", quiet=quiet)
        match_start = time.time()

        for idx, frame_dir in enumerate(frame_dirs, start=1):
            if idx % 1000 == 0:
                progress(
                    f"  matched {idx}/{len(frame_dirs)} frame folders for {root_type}; "
                    f"jobs={discovered['roots'][root_type]['frame_jobs']}, "
                    f"missing={discovered['roots'][root_type]['missing_tar_frames']} ({format_elapsed(match_start)})",
                    quiet=quiet,
                )
            plants: list[PlantMaskSet] = []
            for plant_dir in sorted((p for p in frame_dir.iterdir() if p.is_dir() and re.fullmatch(r"plant_\d+", p.name, flags=re.IGNORECASE)), key=lambda p: natural_sort_key(p.name)):
                root_mask = plant_dir / "root_mask.png"
                if root_mask.exists():
                    plants.append(PlantMaskSet(plant_id=plant_dir.name, plant_dir=str(plant_dir), root_mask=str(root_mask)))
            if not plants:
                continue

            frame_id = frame_dir.name
            fc_id = generate_fc_name(frame_id)
            try:
                tar_path = find_matching_tar(fc_id, tar_index)
            except RuntimeError as exc:
                missing_frames.append({"root_type": root_type, "frame_id": frame_id, "fc_id": fc_id, "error": str(exc)})
                discovered["roots"][root_type]["missing_tar_frames"] += 1
                continue
            if tar_path is None:
                missing_frames.append({"root_type": root_type, "frame_id": frame_id, "fc_id": fc_id, "error": "No matching TAR"})
                discovered["roots"][root_type]["missing_tar_frames"] += 1
                continue

            jobs.append(FrameJob(
                root_type=root_type,
                filter_type=filter_type,
                frame_id=frame_id,
                fc_id=fc_id,
                frame_dir=str(frame_dir),
                tar_path=str(tar_path),
                output_root=str(output_dirs[root_type]),
                plants=tuple(plants),
                pixel_export=pixel_export,
                overlay=overlay,
                fc_preview=fc_preview,
                preview_max_dim=preview_max_dim,
            ))
            discovered["roots"][root_type]["frame_jobs"] += 1
            discovered["roots"][root_type]["plant_masks"] += len(plants)

        progress(
            f"Finished {root_type}: jobs={discovered['roots'][root_type]['frame_jobs']}, "
            f"plant masks={discovered['roots'][root_type]['plant_masks']}, "
            f"missing TAR frames={discovered['roots'][root_type]['missing_tar_frames']} ({format_elapsed(root_start)})",
            quiet=quiet,
        )

    discovered["missing_frames"] = missing_frames[:100]
    discovered["missing_frames_total"] = len(missing_frames)
    return jobs, discovered


def choose_workers(workers: str, profile: str, job_count: int, gb_per_worker: float, reserve_gb: float, pixel_export: str) -> dict[str, Any]:
    logical = os.cpu_count() or 1
    physical = None
    available_gb = None
    total_gb = None
    if psutil is not None:
        try:
            physical = psutil.cpu_count(logical=False)
            logical = psutil.cpu_count(logical=True) or logical
            vm = psutil.virtual_memory()
            available_gb = vm.available / (1024 ** 3)
            total_gb = vm.total / (1024 ** 3)
        except Exception:
            pass
    physical = physical or max(1, logical // 2)

    if pixel_export != "none":
        gb_per_worker = max(gb_per_worker, 1.5)

    ram_limit = None
    if available_gb is not None:
        ram_limit = max(1, int((available_gb - reserve_gb) // gb_per_worker))

    if profile == "conservative":
        cpu_limit = max(1, min(physical // 2, 12))
        io_limit = 8
    elif profile == "aggressive":
        cpu_limit = max(1, physical - 1)
        io_limit = min(physical, 24)
    else:
        cpu_limit = max(1, min(int(physical * 0.70), 16))
        io_limit = 16

    if workers != "auto":
        selected = max(1, int(workers))
        selected = min(selected, max(1, job_count)) if job_count > 0 else selected
        mode = "manual"
    else:
        caps = [cpu_limit, io_limit, max(1, job_count)]
        if ram_limit is not None:
            caps.append(ram_limit)
        selected = max(1, min(caps)) if job_count > 0 else 1
        mode = "auto"

    return {
        "mode": mode,
        "profile": profile,
        "selected_workers": selected,
        "job_count": job_count,
        "logical_cores": logical,
        "physical_cores_estimate": physical,
        "total_ram_gb": total_gb,
        "available_ram_gb": available_gb,
        "gb_per_worker": gb_per_worker,
        "reserve_gb": reserve_gb,
        "cpu_limit": cpu_limit,
        "ram_limit": ram_limit,
        "io_limit": io_limit,
        "pixel_export": pixel_export,
    }


def load_dumm_image_from_fileobj(fileobj: Any, pad_width_right: int = 90) -> tuple[np.ndarray, int, int, int]:
    raw = fileobj.read()
    if len(raw) < 16:
        raise ValueError("DUMM file is shorter than its 16-byte header")
    width, height, bits_per_pixel, bytes_per_pixel = struct.unpack("iiii", raw[:16])
    if bytes_per_pixel == 2:
        dtype = np.uint16
    elif bytes_per_pixel == 1:
        dtype = np.uint8
    elif bytes_per_pixel == 4:
        dtype = np.float32
    else:
        raise ValueError(f"Unsupported bytes per pixel: {bytes_per_pixel}")
    data = np.frombuffer(raw, dtype=dtype, offset=16)
    expected_pixels = width * height
    if data.size != expected_pixels:
        raise ValueError(f"Mismatch: Expected {expected_pixels}, got {data.size}")
    image = data.reshape((height, width)).copy()
    if pad_width_right > 0:
        padding = np.zeros((height, pad_width_right), dtype=image.dtype)
        image = np.concatenate((image, padding), axis=1)
    return image, width + pad_width_right, height, bits_per_pixel


def iter_dumm_images_from_tar(tar_path: str | Path) -> Iterable[tuple[str, str, np.ndarray]]:
    with tarfile.open(tar_path, "r:") as tar:
        members = sorted(
            [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(".dumm")],
            key=lambda member: natural_sort_key(member.name),
        )
        if not members:
            raise FileNotFoundError(f"No .dumm member found in {tar_path}")
        for idx, member in enumerate(members, start=1):
            fh = tar.extractfile(member)
            if fh is None:
                raise IOError(f"Could not open {member.name} from {tar_path}")
            with fh:
                fc_image, _, _, _ = load_dumm_image_from_fileobj(fh)
            yield f"Ft_{idx}", member.name, fc_image


def align_image(
    input_image: np.ndarray,
    target_size: tuple[int, int] = (3006, 4202),
    resized_shape: tuple[int, int] = (3035, 4223),
    offset_x: int = -89,
    offset_y: int = -61,
) -> np.ndarray:
    resized_img = resize(
        input_image.astype(float),
        resized_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(input_image.dtype)

    canvas = np.zeros(target_size, dtype=input_image.dtype)
    h_resized, w_resized = resized_shape
    h_canvas, w_canvas = target_size

    start_y = max(0, offset_y)
    start_x = max(0, offset_x)
    end_y = min(offset_y + h_resized, h_canvas)
    end_x = min(offset_x + w_resized, w_canvas)

    src_y = 0 if offset_y >= 0 else -offset_y
    src_x = 0 if offset_x >= 0 else -offset_x
    cropped_h = end_y - start_y
    cropped_w = end_x - start_x
    if cropped_h > 0 and cropped_w > 0:
        canvas[start_y:end_y, start_x:end_x] = resized_img[src_y:src_y + cropped_h, src_x:src_x + cropped_w]
    return canvas


def resolve_mask_path(mask_path: str | Path) -> Path | None:
    path = Path(mask_path)
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    return path if path.exists() else None


def load_binary_mask(mask_path: str | Path, fallback_shape: tuple[int, int] | None = None) -> np.ndarray:
    resolved = resolve_mask_path(mask_path)
    if resolved is not None:
        arr = np.array(Image.open(resolved).convert("L"))
        return (arr > 0).astype(np.uint8)
    if fallback_shape is None:
        raise FileNotFoundError(mask_path)
    return np.zeros(fallback_shape, dtype=np.uint8)


def pad_mask_if_needed(mask: np.ndarray, target_shape: tuple[int, int], pad_left: int = 90) -> np.ndarray:
    if mask.shape == target_shape:
        return mask
    h, w = target_shape
    mh, mw = mask.shape
    if mh == h and mw + pad_left == w:
        return np.pad(mask, ((0, 0), (pad_left, 0)), mode="constant", constant_values=0)
    if mh <= h and mw <= w:
        out = np.zeros(target_shape, dtype=mask.dtype)
        out[:mh, :mw] = mask
        return out
    return np.array(Image.fromarray(mask).resize((w, h), resample=Image.Resampling.NEAREST)) > 0


def mask_exists(mask_dir: str | Path, name: str) -> bool:
    return resolve_mask_path(Path(mask_dir) / name) is not None


def summarize_mask(fc_image: np.ndarray, mask: np.ndarray) -> tuple[float, int, float]:
    idx = mask > 0
    n = int(idx.sum())
    if n == 0:
        return float("nan"), 0, float("nan")
    values = fc_image[idx]
    return float(values.mean()), n, float(values.sum())


def create_overlay(fc_image: np.ndarray, region_layers: dict[str, tuple[np.ndarray, tuple[int, int, int]]]) -> np.ndarray:
    fc_image_rescaled = rescale_intensity(fc_image, in_range="image", out_range=(0, 255)).astype(np.uint8)
    overlay_rgb = gray2rgb(fc_image_rescaled)
    for mask, color in region_layers.values():
        overlay_rgb[mask > 0] = list(color)
    return overlay_rgb


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.nanmin(arr[finite]))
    hi = float(np.nanmax(arr[finite]))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr.astype(np.float32) - lo) / (hi - lo)
    scaled = np.clip(scaled * 255.0, 0, 255)
    return scaled.astype(np.uint8)


def resize_preview_uint8(image_u8: np.ndarray, max_dim: int) -> np.ndarray:
    if max_dim <= 0:
        return image_u8
    h, w = image_u8.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image_u8
    scale = max_dim / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    pil = Image.fromarray(image_u8)
    return np.array(pil.resize(new_size, resample=Image.Resampling.BILINEAR))


def write_fc_preview(path: Path, image: np.ndarray, max_dim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = resize_preview_uint8(image_to_uint8(image), max_dim=max_dim)
    Image.fromarray(preview).save(path)


def write_alignment_qc(path: Path, qc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(qc, fh, indent=2, sort_keys=True, default=str)


def write_summary_csv(path: Path, rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fc_file", "dumm_layer", "dumm_member", "plant_id", "parameter", "value"])
        writer.writerows(rows)


def write_pixels_parquet(
    path: Path,
    fc_image: np.ndarray,
    masks: dict[str, np.ndarray],
    job: FrameJob,
    plant: PlantMaskSet,
    dumm_layer: str,
    dumm_member: str,
) -> None:
    """Write long-format per-pixel fluorescence values as Parquet.

    One row is one pixel membership in one ROI region. A physical pixel may appear
    more than once if it belongs to multiple semantic regions, for example
    `root_area` and `dilated_root`. That is intentional and mirrors the summary
    CSV contract where regions are independent measurements.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for Parquet pixel export") from exc

    frames = []
    for region, mask in masks.items():
        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            continue
        values = fc_image[coords[:, 0], coords[:, 1]]
        n = int(coords.shape[0])
        frames.append(pd.DataFrame({
            "root_type": np.repeat(job.root_type, n),
            "fc_id": np.repeat(job.fc_id, n),
            "frame_id": np.repeat(job.frame_id, n),
            "plant_id": np.repeat(plant.plant_id, n),
            "filter_type": np.repeat(job.filter_type, n),
            "dumm_layer": np.repeat(dumm_layer, n),
            "dumm_member": np.repeat(dumm_member, n),
            "region": np.repeat(region, n),
            "y": coords[:, 0].astype(np.int32, copy=False),
            "x": coords[:, 1].astype(np.int32, copy=False),
            "fluorescence": values.astype(np.float32, copy=False),
        }))

    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df[PIXEL_PARQUET_COLUMNS]
    else:
        df = pd.DataFrame(columns=PIXEL_PARQUET_COLUMNS)

    try:
        df.to_parquet(path, index=False, compression="snappy")
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Parquet pixel export requires pyarrow or fastparquet; install pyarrow") from exc



def process_plant(
    job: FrameJob,
    plant: PlantMaskSet,
    aligned_fc: np.ndarray,
    dumm_layer: str,
    dumm_member_name: str,
) -> tuple[int, int, int]:
    mask_dir = Path(plant.plant_dir)
    binary_mask = load_binary_mask(plant.root_mask)
    binary_mask = pad_mask_if_needed(binary_mask, aligned_fc.shape)

    main_root_mask = pad_mask_if_needed(load_binary_mask(mask_dir / "main_root_mask", binary_mask.shape), aligned_fc.shape)
    lateral_root_mask = pad_mask_if_needed(load_binary_mask(mask_dir / "lateral_root_mask", binary_mask.shape), aligned_fc.shape)
    tip_mask = pad_mask_if_needed(load_binary_mask(mask_dir / "tip_mask", binary_mask.shape), aligned_fc.shape)
    node_mask = pad_mask_if_needed(load_binary_mask(mask_dir / "node_mask", binary_mask.shape), aligned_fc.shape)
    node_mask = binary_dilation(node_mask, iterations=5).astype(np.uint8)
    shoot_mask = pad_mask_if_needed(load_binary_mask(mask_dir / "shoot_mask", binary_mask.shape), aligned_fc.shape)

    root_area_mask: np.ndarray | None = None
    if mask_exists(mask_dir, "root_area_mask"):
        root_area_mask = pad_mask_if_needed(load_binary_mask(mask_dir / "root_area_mask", binary_mask.shape), aligned_fc.shape)

    dilated_root_mask = binary_dilation(binary_mask, iterations=50).astype(np.uint8)
    shoot_coords = np.argwhere(shoot_mask > 0)
    if shoot_coords.size > 0:
        shoot_bottom_y = int(shoot_coords[:, 0].max())
        allowed = np.ones_like(dilated_root_mask, dtype=bool)
        allowed[:shoot_bottom_y, :] = False
        dilated_root_mask = np.where(allowed, dilated_root_mask, 0).astype(np.uint8)

    dilated_border = (dilated_root_mask - binary_erosion(dilated_root_mask, iterations=1).astype(np.uint8)).astype(np.uint8)
    dilated_exclusive = ((dilated_root_mask == 1) & (binary_mask == 0)).astype(np.uint8)

    tip_in_main_root = (tip_mask > 0) & (main_root_mask > 0)
    main_tip_coords = np.argwhere(tip_in_main_root)
    if main_tip_coords.size > 0:
        main_tip_coord = main_tip_coords[np.argmax(main_tip_coords[:, 0])]
        main_tip_mask = np.zeros_like(tip_mask, dtype=np.uint8)
        main_tip_mask[main_tip_coord[0], main_tip_coord[1]] = 255
        main_tip_mask = binary_dilation(main_tip_mask, iterations=5).astype(np.uint8)
        other_tip_mask = tip_mask.copy()
        other_tip_mask[main_tip_coord[0], main_tip_coord[1]] = 0
        other_tip_mask = binary_dilation(other_tip_mask, iterations=5).astype(np.uint8)
    else:
        main_tip_mask = np.zeros_like(tip_mask, dtype=np.uint8)
        other_tip_mask = tip_mask.copy()

    masks: dict[str, np.ndarray] = {
        "main_root": main_root_mask,
        "lateral_root": lateral_root_mask,
        "main_root_tip": main_tip_mask,
        "other_tip": other_tip_mask,
        "node": node_mask,
        "dilated_root_exclusive": dilated_exclusive,
        "dilated_root": dilated_root_mask,
        "shoot": shoot_mask,
    }
    if root_area_mask is not None:
        masks["root_area"] = root_area_mask

    summary_rows: list[list[Any]] = []
    fc_base = Path(job.tar_path).name
    for region, mask in masks.items():
        mean_val, n_pixels, sum_val = summarize_mask(aligned_fc, mask)
        summary_rows.extend([
            [fc_base, dumm_layer, dumm_member_name, plant.plant_id, f"mean_fluorescence_{region}", mean_val],
            [fc_base, dumm_layer, dumm_member_name, plant.plant_id, f"n_pixels_{region}", n_pixels],
            [fc_base, dumm_layer, dumm_member_name, plant.plant_id, f"sum_fluorescence_{region}", sum_val],
        ])

    plant_suffix = plant.plant_id.replace("_", "")
    output_subfolder = Path(job.output_root) / job.fc_id / plant_suffix / dumm_layer
    output_prefix = f"{job.fc_id}_{dumm_layer}"
    write_summary_csv(output_subfolder / f"{output_prefix}_summary.csv", summary_rows)

    pixel_files = 0
    pixels_path: Path | None = None
    if job.pixel_export == "parquet":
        pixels_path = output_subfolder / f"{output_prefix}_pixels.parquet"
        write_pixels_parquet(pixels_path, aligned_fc, masks, job, plant, dumm_layer, dumm_member_name)
        pixel_files = 1

    overlays = 0
    if job.overlay:
        overlay_layers = {
            "dilated_border": (dilated_border, (255, 128, 0)),
            "main_root": (main_root_mask, (255, 0, 0)),
            "lateral_root": (lateral_root_mask, (0, 255, 0)),
            "main_tip": (main_tip_mask, (0, 0, 255)),
            "other_tip": (other_tip_mask, (0, 255, 255)),
            "node": (node_mask, (255, 255, 0)),
            "shoot": (shoot_mask, (255, 0, 255)),
        }
        if root_area_mask is not None:
            overlay_layers["root_area"] = (root_area_mask, (128, 255, 128))
        overlay_rgb = create_overlay(aligned_fc, overlay_layers)
        Image.fromarray(overlay_rgb.astype(np.uint8)).save(output_subfolder / f"{output_prefix}_overlay.png")
        overlays = 1

    params = {
        "analysis_type": "hades_fc_single_plant_fluorescence",
        "frame_id": job.frame_id,
        "root_type": job.root_type,
        "plant_id": plant.plant_id,
        "source_root_mask": plant.root_mask,
        "source_fc_tar": job.tar_path,
        "source_dumm_member": dumm_member_name,
        "dumm_layer": dumm_layer,
        "filter_type": job.filter_type,
        "alignment_parameters": ALIGNMENT_CONFIG[job.root_type][job.filter_type],
        "pixel_export": job.pixel_export,
        "overlay": job.overlay,
        "roi_definition": {
            "node_binary_dilation_iterations": 5,
            "tip_binary_dilation_iterations": 5,
            "peri_root_binary_dilation_iterations": 50,
            "exclude_peri_root_above_shoot_bottom": True,
        },
        "outputs": {
            "summary_csv": str(output_subfolder / f"{output_prefix}_summary.csv"),
            "pixels": str(pixels_path) if pixels_path is not None else None,
            "overlay_png": str(output_subfolder / f"{output_prefix}_overlay.png") if job.overlay else None,
        },
    }
    with open(output_subfolder / "run_parameters.json", "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2, sort_keys=True)

    return 1, pixel_files, overlays


def process_frame_job(job: FrameJob) -> JobResult:
    start = time.time()
    result = JobResult(frame_id=job.frame_id, root_type=job.root_type, tar_path=job.tar_path, ok=False, elapsed_s=0.0)
    try:
        if job.filter_type not in ALIGNMENT_CONFIG[job.root_type]:
            raise ValueError(f"Unknown filter {job.filter_type!r} for {job.root_type}")
        align_params = ALIGNMENT_CONFIG[job.root_type][job.filter_type]
        for dumm_layer, dumm_member_name, fc_image in iter_dumm_images_from_tar(job.tar_path):
            raw_flipped_fc = np.fliplr(fc_image)
            aligned_fc = align_image(raw_flipped_fc, **align_params)
            result.dumm_layers_processed += 1

            if job.fc_preview != "none":
                qc_dir = Path(job.output_root) / job.fc_id / "_frame_qc" / dumm_layer
                if job.fc_preview in {"raw", "both"}:
                    write_fc_preview(qc_dir / f"{job.fc_id}_{dumm_layer}_raw_fc_preview.png", raw_flipped_fc, job.preview_max_dim)
                    result.fc_preview_images_created += 1
                if job.fc_preview in {"aligned", "both"}:
                    write_fc_preview(qc_dir / f"{job.fc_id}_{dumm_layer}_aligned_fc_preview.png", aligned_fc, job.preview_max_dim)
                    result.fc_preview_images_created += 1
                write_alignment_qc(qc_dir / "alignment_qc.json", {
                    "analysis_type": "hades_fc_frame_alignment_qc",
                    "frame_id": job.frame_id,
                    "fc_id": job.fc_id,
                    "root_type": job.root_type,
                    "filter_type": job.filter_type,
                    "source_fc_tar": job.tar_path,
                    "source_dumm_member": dumm_member_name,
                    "dumm_layer": dumm_layer,
                    "raw_flipped_shape": list(raw_flipped_fc.shape),
                    "aligned_shape": list(aligned_fc.shape),
                    "alignment_parameters": align_params,
                    "plants_in_frame": [p.plant_id for p in job.plants],
                    "preview_max_dim": job.preview_max_dim,
                    "fc_preview_mode": job.fc_preview,
                })
                result.alignment_qc_files_created += 1

            for plant in job.plants:
                summaries, pixel_files, overlays = process_plant(job, plant, aligned_fc, dumm_layer, dumm_member_name)
                result.summary_csv_exported += summaries
                result.pixel_files_exported += pixel_files
                result.overlay_images_created += overlays
                result.plants_processed += 1
        result.ok = True
    except Exception as exc:
        result.error = repr(exc)
    finally:
        result.elapsed_s = time.time() - start
    return result


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True, default=str)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


MASTER_SUMMARY_COLUMNS = [
    "root_type",
    "fc_id",
    "frame_id",
    "plant_folder",
    "dumm_layer_folder",
    "summary_csv",
    "run_parameters_json",
    "fc_file",
    "dumm_layer",
    "dumm_member",
    "plant_id",
    "parameter",
    "value",
    "source_root_mask",
    "source_fc_tar",
    "filter_type",
]


def infer_summary_context(output_dir: Path, summary_csv: Path) -> tuple[str, str, str]:
    rel = summary_csv.relative_to(output_dir)
    parts = rel.parts
    fc_id = parts[0] if len(parts) > 0 else ""
    plant_folder = parts[1] if len(parts) > 1 else ""
    dumm_layer_folder = parts[2] if len(parts) > 2 else ""
    return fc_id, plant_folder, dumm_layer_folder


def read_run_parameters(summary_csv: Path) -> dict[str, Any]:
    params_path = summary_csv.parent / "run_parameters.json"
    if not params_path.exists():
        return {}
    try:
        with open(params_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def write_master_summary_csv(output_dir: Path, root_type: str, master_name: str = "master_summary.csv") -> tuple[Path, int, int]:
    summary_files = sorted(
        (p for p in output_dir.rglob("*_summary.csv") if p.name != master_name and not p.name.startswith("master_")),
        key=lambda p: natural_sort_key(str(p)),
    )
    master_path = output_dir / master_name
    master_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    files_read = 0
    with open(master_path, "w", newline="", encoding="utf-8") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=MASTER_SUMMARY_COLUMNS)
        writer.writeheader()
        for summary_csv in summary_files:
            params = read_run_parameters(summary_csv)
            fc_id, plant_folder, dumm_layer_folder = infer_summary_context(output_dir, summary_csv)
            with open(summary_csv, "r", newline="", encoding="utf-8") as in_fh:
                reader = csv.DictReader(in_fh)
                files_read += 1
                for row in reader:
                    writer.writerow({
                        "root_type": root_type,
                        "fc_id": fc_id,
                        "frame_id": params.get("frame_id", ""),
                        "plant_folder": plant_folder,
                        "dumm_layer_folder": dumm_layer_folder,
                        "summary_csv": str(summary_csv),
                        "run_parameters_json": str(summary_csv.parent / "run_parameters.json") if (summary_csv.parent / "run_parameters.json").exists() else "",
                        "fc_file": row.get("fc_file", ""),
                        "dumm_layer": row.get("dumm_layer", ""),
                        "dumm_member": row.get("dumm_member", ""),
                        "plant_id": row.get("plant_id", ""),
                        "parameter": row.get("parameter", ""),
                        "value": row.get("value", ""),
                        "source_root_mask": params.get("source_root_mask", ""),
                        "source_fc_tar": params.get("source_fc_tar", ""),
                        "filter_type": params.get("filter_type", ""),
                    })
                    rows_written += 1
    return master_path, files_read, rows_written


def write_combined_master_summary_csv(experiment_root: Path, per_root_masters: list[Path], master_name: str = "FC_master_summary.csv") -> tuple[Path, int]:
    combined_path = experiment_root / master_name
    rows_written = 0
    with open(combined_path, "w", newline="", encoding="utf-8") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=MASTER_SUMMARY_COLUMNS)
        writer.writeheader()
        for master_path in per_root_masters:
            with open(master_path, "r", newline="", encoding="utf-8") as in_fh:
                reader = csv.DictReader(in_fh)
                for row in reader:
                    writer.writerow(row)
                    rows_written += 1
    return combined_path, rows_written


def build_master_summaries(experiment_root: Path, output_dirs: dict[str, Path], master_name: str = "master_summary.csv") -> dict[str, Any]:
    result: dict[str, Any] = {"per_root": {}, "combined": None}
    masters: list[Path] = []
    for root_type, output_dir in output_dirs.items():
        master_path, files_read, rows_written = write_master_summary_csv(output_dir, root_type, master_name=master_name)
        masters.append(master_path)
        result["per_root"][root_type] = {
            "master_summary_csv": str(master_path),
            "summary_files_read": files_read,
            "rows_written": rows_written,
        }
    if len(masters) > 1:
        combined_path, combined_rows = write_combined_master_summary_csv(experiment_root, masters)
        result["combined"] = {"master_summary_csv": str(combined_path), "rows_written": combined_rows}
    return result


def run_pipeline(args: argparse.Namespace) -> None:
    pipeline_start = time.time()
    quiet = bool(args.quiet)
    progress("Starting Hades FC analysis", quiet=quiet)
    progress(f"Input: {args.input}", quiet=quiet)
    if args.pixel_export == "parquet":
        require_parquet_writer()
    progress("Resolving experiment/root/output paths...", quiet=quiet)
    experiment_root, root_set = infer_experiment_context(args.input, root_set=args.root_set)
    root_analysis_dirs = resolve_root_analysis_dirs(experiment_root, root_set, args.root_analysis)
    fc_dirs = resolve_fc_dirs(experiment_root, root_set, args.fc_dir)
    output_dirs = resolve_output_dirs(experiment_root, root_set, args.output)
    alignment_filters = resolve_required_alignment_filters(args, root_set)

    # Store run logs under the experiment root as early as possible, so users can
    # see a file appear before long discovery/indexing work starts.
    log_dir = (args.log_dir.expanduser().resolve() if args.log_dir else experiment_root / "FC_analysis_logs" / time.strftime("%Y%m%d_%H%M%S"))
    write_json(log_dir / "fc_start_context.json", {
        "analysis_type": "hades_fc_parallelized_start_context",
        "experiment_root": str(experiment_root),
        "root_set": root_set,
        "root_analysis_dirs": {k: str(v) for k, v in root_analysis_dirs.items()},
        "fc_dirs": {k: str(v) for k, v in fc_dirs.items()},
        "output_dirs": {k: str(v) for k, v in output_dirs.items()},
        "alignment_filters": alignment_filters,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    progress(f"Log directory: {log_dir}", quiet=quiet)
    progress(f"Root set: {root_set}", quiet=quiet)
    for root_type in sorted(root_analysis_dirs):
        progress(f"  {root_type}: analysis={root_analysis_dirs[root_type]}", quiet=quiet)
        progress(f"  {root_type}: FC TAR search dir={fc_dirs[root_type]}", quiet=quiet)
        progress(f"  {root_type}: output={output_dirs[root_type]}", quiet=quiet)
        progress(f"  {root_type}: alignment filter={alignment_filters[root_type]}", quiet=quiet)

    missing = [str(p) for p in list(root_analysis_dirs.values()) + list(fc_dirs.values()) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required directories:\n" + "\n".join(missing))

    progress("Discovering frame/TAR jobs. This scans ROOT*_analysis and indexes FC TAR files.", quiet=quiet)
    jobs, discovered = discover_frame_jobs(
        root_analysis_dirs=root_analysis_dirs,
        fc_dirs=fc_dirs,
        output_dirs=output_dirs,
        filter_fc1=alignment_filters.get("ROOT1", ""),
        filter_fc2=alignment_filters.get("ROOT2", ""),
        pixel_export=args.pixel_export,
        overlay=not args.no_overlay,
        fc_preview=args.fc_preview,
        preview_max_dim=args.preview_max_dim,
        quiet=quiet,
    )
    progress(f"Discovery complete: {len(jobs)} matched frame/TAR job(s); missing TAR frames={discovered.get('missing_frames_total', 0)}", quiet=quiet)
    progress("Estimating worker plan...", quiet=quiet)
    plan = choose_workers(args.workers, args.worker_profile, len(jobs), args.gb_per_worker, args.reserve_gb, args.pixel_export)
    progress(
        f"Worker plan: {plan['selected_workers']} worker(s) "
        f"(mode={plan['mode']}, profile={plan['profile']}, jobs={plan['job_count']}, "
        f"physical_cores={plan['physical_cores_estimate']}, available_ram_gb={plan['available_ram_gb']})",
        quiet=quiet,
    )

    run_info = {
        "analysis_type": "hades_fc_parallelized",
        "experiment_root": str(experiment_root),
        "root_set": root_set,
        "root_analysis_dirs": {k: str(v) for k, v in root_analysis_dirs.items()},
        "fc_dirs": {k: str(v) for k, v in fc_dirs.items()},
        "output_dirs": {k: str(v) for k, v in output_dirs.items()},
        "alignment_filters": alignment_filters,
        "filter_fc1": alignment_filters.get("ROOT1"),
        "filter_fc2": alignment_filters.get("ROOT2"),
        "pixel_export": args.pixel_export,
        "overlay": not args.no_overlay,
        "fc_preview": args.fc_preview,
        "preview_max_dim": args.preview_max_dim,
        "master_summary": not args.no_master_summary,
        "master_summary_name": args.master_summary_name,
        "discovered": discovered,
        "worker_plan": plan,
        "progress_style": args.progress_style,
        "progress_log_every": args.progress_log_every,
    }

    write_json(log_dir / "fc_run_plan.json", run_info)
    if not quiet:
        print(json.dumps(run_info, indent=2, sort_keys=True, default=str), flush=True)

    if args.dry_run_plan:
        progress(f"Dry run only. Plan written to: {log_dir / 'fc_run_plan.json'}", quiet=quiet)
        return

    if not jobs:
        raise RuntimeError("No matched frame/TAR jobs were discovered. See fc_run_plan.json for missing frames.")

    selected_workers = int(plan["selected_workers"])
    progress(f"Starting processing with {selected_workers} worker(s). Manifest: {log_dir / 'fc_job_manifest.jsonl'}", quiet=quiet)
    manifest_path = log_dir / "fc_job_manifest.jsonl"
    failed: list[dict[str, Any]] = []
    totals = {
        "jobs_total": len(jobs),
        "jobs_ok": 0,
        "jobs_failed": 0,
        "plants_processed": 0,
        "dumm_layers_processed": 0,
        "summary_csv_exported": 0,
        "pixel_files_exported": 0,
        "overlay_images_created": 0,
        "fc_preview_images_created": 0,
        "alignment_qc_files_created": 0,
    }
    start = time.time()
    job_progress = JobProgress(
        len(jobs),
        quiet=quiet,
        style=args.progress_style,
        log_every=args.progress_log_every,
    )
    job_progress.update(0, force=True)

    if selected_workers == 1:
        iterator = (process_frame_job(job) for job in jobs)
        for i, result in enumerate(iterator, start=1):
            row = asdict(result)
            append_jsonl(manifest_path, row)
            if result.ok:
                totals["jobs_ok"] += 1
            else:
                totals["jobs_failed"] += 1
                failed.append(row)
            totals["plants_processed"] += result.plants_processed
            totals["dumm_layers_processed"] += result.dumm_layers_processed
            totals["summary_csv_exported"] += result.summary_csv_exported
            totals["pixel_files_exported"] += result.pixel_files_exported
            totals["overlay_images_created"] += result.overlay_images_created
            totals["fc_preview_images_created"] += result.fc_preview_images_created
            totals["alignment_qc_files_created"] += result.alignment_qc_files_created
            job_progress.update(i, ok=totals["jobs_ok"], failed=totals["jobs_failed"])
    else:
        executor_kwargs = {"max_workers": selected_workers}
        try:
            import inspect
            if "max_tasks_per_child" in inspect.signature(futures.ProcessPoolExecutor).parameters:
                executor_kwargs["max_tasks_per_child"] = args.maxtasksperchild
        except Exception:
            pass
        with futures.ProcessPoolExecutor(**executor_kwargs) as pool:
            future_map = {pool.submit(process_frame_job, job): job for job in jobs}
            for i, fut in enumerate(futures.as_completed(future_map), start=1):
                result = fut.result()
                row = asdict(result)
                append_jsonl(manifest_path, row)
                if result.ok:
                    totals["jobs_ok"] += 1
                else:
                    totals["jobs_failed"] += 1
                    failed.append(row)
                totals["plants_processed"] += result.plants_processed
                totals["dumm_layers_processed"] += result.dumm_layers_processed
                totals["summary_csv_exported"] += result.summary_csv_exported
                totals["pixel_files_exported"] += result.pixel_files_exported
                totals["overlay_images_created"] += result.overlay_images_created
                totals["fc_preview_images_created"] += result.fc_preview_images_created
                totals["alignment_qc_files_created"] += result.alignment_qc_files_created
                job_progress.update(i, ok=totals["jobs_ok"], failed=totals["jobs_failed"])

    job_progress.finish(ok=totals["jobs_ok"], failed=totals["jobs_failed"])

    totals["elapsed_s"] = time.time() - start
    if not args.no_master_summary:
        progress("Building master_summary.csv file(s)...", quiet=quiet)
        totals["master_summary"] = build_master_summaries(experiment_root, output_dirs, master_name=args.master_summary_name)
        progress("Master summary complete", quiet=quiet)
    write_json(log_dir / "fc_run_summary.json", totals)
    if failed:
        write_json(log_dir / "failed_jobs.json", failed)
    progress(f"FC analysis finished in {format_elapsed(pipeline_start)}. Summary: {log_dir / 'fc_run_summary.json'}", quiet=quiet)
    if not quiet:
        print(json.dumps(totals, indent=2, sort_keys=True), flush=True)
    if failed and args.stop_on_error:
        raise RuntimeError(f"{len(failed)} FC jobs failed; see {log_dir / 'failed_jobs.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel Hades FC analysis. Give --input as the experiment root, ROOT1/ROOT2, "
            "or ROOT*_analysis. The script infers ROOT*_analysis and FC*/Measurement/TARs by default."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Experiment root, ROOT1/ROOT2 folder, ROOT*_analysis folder, or Measurement folder.")
    parser.add_argument("--root-set", default="auto", help="auto, ROOT1, ROOT2, or both. auto infers from --input when possible.")
    parser.add_argument("--root-analysis", type=Path, default=None, help="Override ROOT*_analysis path for a single root set.")
    parser.add_argument("--fc-dir", type=Path, default=None, help="Override FC TAR directory for a single root set.")
    parser.add_argument("--output", type=Path, default=None, help="Override FC analysis output directory for a single root set. Default: FC1_analysis or FC2_analysis.")
    parser.add_argument("--filter-fc1", "--filter-root1", dest="filter_fc1", default=None, help="Required alignment filter/config for FC1/ROOT1. Available ROOT1 filters: F483, F513, F635. If omitted in an interactive run, the script prompts for it.")
    parser.add_argument("--filter-fc2", "--filter-root2", dest="filter_fc2", default=None, help="Required alignment filter/config for FC2/ROOT2. Available ROOT2 filters: F513, F593, F635. If omitted in an interactive run, the script prompts for it.")
    parser.add_argument("--workers", default="auto", help="auto or an integer worker count. Jobs are grouped by frame/TAR.")
    parser.add_argument("--worker-profile", choices=["conservative", "balanced", "aggressive"], default="balanced")
    parser.add_argument("--gb-per-worker", type=float, default=1.5, help="Memory estimate for auto worker planning. Default reflects that pixel Parquet export is enabled.")
    parser.add_argument("--reserve-gb", type=float, default=8.0, help="RAM to reserve for OS/filesystem/cache/other processes.")
    parser.add_argument("--maxtasksperchild", type=int, default=20, help="Restart worker processes after this many frame jobs.")
    parser.add_argument("--pixel-export", choices=["parquet", "none"], default="parquet", help="Default is parquet. Pixel-level output is written only as Parquet; use none to disable it.")
    parser.add_argument("--no-overlay", action="store_true", help="Disable overlay PNG output. Default: overlays are always written.")
    parser.add_argument("--fc-preview", choices=["none", "aligned", "raw", "both"], default="aligned", help="Frame-level FC QC preview. Default writes one aligned fluorescence preview per TAR/DUMM layer.")
    parser.add_argument("--preview-max-dim", type=int, default=1200, help="Longest side for FC preview PNGs. Use 0 to keep full size.")
    parser.add_argument("--no-master-summary", action="store_true", help="Disable default master_summary.csv aggregation under each FC*_analysis directory.")
    parser.add_argument("--master-summary-name", default="master_summary.csv", help="Filename for the default per-FC master summary CSV.")
    parser.add_argument("--dry-run-plan", action="store_true", help="Resolve paths, jobs, TAR matches, and worker plan, then exit.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Optional directory for plan, manifest, and summary logs.")
    parser.add_argument("--progress-style", choices=["auto", "bar", "log", "none"], default="auto", help="Job progress display. auto uses a single-line progress bar in an interactive terminal and sparse log messages when redirected.")
    parser.add_argument("--progress-log-every", type=int, default=100, help="When progress style is log, print one progress line every N completed jobs.")
    parser.add_argument("--quiet", action="store_true", help="Suppress startup/discovery/progress console messages.")
    parser.add_argument("--stop-on-error", action="store_true", help="Return an error if any frame job fails.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_pipeline(args)
    except UserConfigError as exc:
        print(f"[FC] ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
