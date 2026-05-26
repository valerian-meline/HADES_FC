from __future__ import annotations

import json
import os
import re
import struct
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.color import gray2rgb
from skimage.exposure import rescale_intensity
from skimage.transform import resize


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


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_run_parameters(output_dir: str | Path, params: dict[str, Any], filename: str = "run_parameters.json") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(params), fh, indent=2, sort_keys=True)
    return out_path


def load_dumm_image(filepath: str | Path, pad_width_right: int = 90) -> tuple[np.ndarray, int, int, int]:
    with open(filepath, "rb") as f:
        header = f.read(16)
        width, height, bits_per_pixel, bytes_per_pixel = struct.unpack("iiii", header)

        if bytes_per_pixel == 2:
            dtype = np.uint16
        elif bytes_per_pixel == 1:
            dtype = np.uint8
        elif bytes_per_pixel == 4:
            dtype = np.float32
        else:
            raise ValueError(f"Unsupported bytes per pixel: {bytes_per_pixel}")

        data = np.fromfile(f, dtype=dtype)
        expected_pixels = width * height
        if data.size != expected_pixels:
            raise ValueError(f"Mismatch: Expected {expected_pixels}, got {data.size}")

        image = data.reshape((height, width))

        if pad_width_right > 0:
            padding = np.zeros((height, pad_width_right), dtype=image.dtype)
            image = np.concatenate((image, padding), axis=1)

    return image, width + pad_width_right, height, bits_per_pixel


def normalize_string(value: str) -> str:
    return re.sub(r"[_\-\s]", "", str(value).lower())


def list_root_mask_images(working_directory: str | Path) -> list[str]:
    root_mask_files: list[str] = []
    for dirpath, _, filenames in os.walk(working_directory):
        if any(f"plant_{i}" in os.path.normpath(dirpath) for i in range(1, 6)):
            for filename in filenames:
                if filename.lower() == "root_mask.png":
                    root_mask_files.append(os.path.abspath(os.path.join(dirpath, filename)))
    return root_mask_files


def list_tar_files(working_directory: str | Path) -> list[str]:
    tar_files: list[str] = []
    for dirpath, _, filenames in os.walk(working_directory):
        for filename in filenames:
            if filename.lower().endswith(".tar"):
                tar_files.append(os.path.join(dirpath, filename))
    return tar_files


def extract_file_id_from_path(file_path: str | Path) -> str:
    parts = os.path.normpath(str(file_path)).split(os.sep)
    for i, part in enumerate(parts):
        if normalize_string(part).startswith("plant") and i > 0:
            return parts[i - 1]
    return "Unknown"

def generate_fc_name(file_id: str) -> str:
    # Replace ROOT1 or ROOT2 and everything after it
    file_id = re.sub(r"ROOT1.*", "FC1_FcTar", file_id)
    file_id = re.sub(r"ROOT2.*", "FC2_FcTar", file_id)

    return file_id

def clean_tar_filename(filename: str) -> str:
    return re.sub(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_?", "", filename)


def natural_sort_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def extract_dumm_files_from_tar(tar_path: str | Path, extract_to: str | Path) -> list[tuple[str, str, str]]:
    with tarfile.open(tar_path, "r") as tar:
        dumm_members = sorted(
            [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(".dumm")],
            key=lambda member: natural_sort_key(member.name),
        )
        if not dumm_members:
            print(f"⚠️ No .dumm file found in: {os.path.basename(tar_path)}")
            return []
        if len(dumm_members) > 1:
            print(f"Found {len(dumm_members)} .dumm files in: {os.path.basename(tar_path)}")

        extracted_files: list[tuple[str, str, str]] = []
        for index, dumm_member in enumerate(dumm_members, start=1):
            dumm_layer = f"Ft_{index}"
            tar.extract(dumm_member, path=extract_to)
            extracted_files.append((dumm_layer, os.path.join(extract_to, dumm_member.name), dumm_member.name))
        return extracted_files


def align_mask(
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

    canvas[start_y:end_y, start_x:end_x] = resized_img[src_y:src_y + cropped_h, src_x:src_x + cropped_w]
    return canvas


def load_binary_mask(mask_path: str | Path, fallback_shape: tuple[int, int] | None = None) -> np.ndarray:
    if os.path.exists(mask_path):
        return (np.array(Image.open(mask_path).convert("L")) > 0).astype(np.uint8)
    if fallback_shape is None:
        raise FileNotFoundError(mask_path)
    return np.zeros(fallback_shape, dtype=np.uint8)


def create_overlay(fc_image: np.ndarray, region_layers: dict[str, tuple[np.ndarray, tuple[int, int, int]]]) -> np.ndarray:
    fc_image_rescaled = rescale_intensity(fc_image, in_range="image", out_range=(0, 255)).astype(np.uint8)
    overlay_rgb = gray2rgb(fc_image_rescaled)
    for mask, color in region_layers.values():
        overlay_rgb[mask > 0] = list(color)
    return overlay_rgb


def summarize_region(df: pd.DataFrame) -> tuple[float, int, float]:
    if df.empty:
        return np.nan, 0, np.nan
    return float(df["fluorescence"].mean()), int(len(df)), float(df["fluorescence"].sum())

def pad_mask_if_needed(mask, target_shape, file_id="", mask_name="", pad_right=90):
    """
    mask: PIL Image or numpy array
    target_shape: fc_image.shape, usually (height, width, channels)
    """

    target_size = (target_shape[1], target_shape[0])  # (width, height)

    # Convert numpy array to PIL if needed
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)

    # Already correct
    if mask.size == target_size:
        return mask

    mask_array = np.array(mask)

    padded = np.pad(
        mask_array,
        ((0, 0), (pad_right, 0)),  # pad right side only
        mode="constant",
        constant_values=0
    )

    padded_mask = Image.fromarray(padded)

    if padded_mask.size == target_size:
        return padded_mask
    return padded_mask


def process_single_root_mask(
    file_path: str | Path,
    working_directory: str | Path,
    fc_files: list[str],
    filter_type: str = "F635",
    alignment_config: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> int:

    alignment_config = alignment_config or ALIGNMENT_CONFIG
    file_path = str(file_path)
    working_directory = str(working_directory)

    file_id = extract_file_id_from_path(file_path)
    file_id_fc = generate_fc_name(file_id)
    file_id_fc = clean_tar_filename(file_id_fc)

    matching_tar_files = [f for f in fc_files if file_id_fc in re.sub(".tar", "", clean_tar_filename(os.path.basename(f)))]
    if not matching_tar_files:
        print(f"No matching TAR for: {file_id_fc}")
        return 0
    if len(matching_tar_files) > 1:
        print(f"Multiple TAR files matched for {file_id_fc}:")
        for tar in matching_tar_files:
            print(f"   - {os.path.basename(tar)}")
        return 0

    fc_path = matching_tar_files[0]
    print(f"\nFound TAR: {os.path.basename(fc_path)}")

    if "ROOT1" in file_path:
        root_type = "ROOT1"
    elif "ROOT2" in file_path:
        root_type = "ROOT2"
    else:
        print(f"Unknown ROOT type for file: {file_path}")
        return 0

    if filter_type not in alignment_config[root_type]:
        print(f"Unknown filter type '{filter_type}' for ROOT '{root_type}'")
        return 0

    align_params = alignment_config[root_type][filter_type]
    plant_id = next((f"plant_{i}" for i in range(1, 6) if f"plant_{i}" in file_path), "unknown")
    fc_type = "FC1_analysis" if "FC1" in file_id_fc else "FC2_analysis"
    plant_match = [f"_plant_{i}" for i in range(1, 6) if f"plant_{i}" in file_path]
    plant_suffix = plant_match[0].replace("_", "") if plant_match else "plant_?"
    mask_dir = os.path.dirname(file_path)

    processed_dumm_count = 0
    with tempfile.TemporaryDirectory() as tempdir:
        dumm_files = extract_dumm_files_from_tar(fc_path, tempdir)
        if not dumm_files:
            print("   No .dumm found or failed to extract.")
            return 0

        for dumm_layer, dumm_path, dumm_member_name in dumm_files:
            print(f"   Processing {dumm_layer}: {dumm_member_name}")
            fc_image, _, _, _ = load_dumm_image(dumm_path)
            fc_image = np.fliplr(fc_image)
            fc_image = align_mask(fc_image, **align_params)

            root_mask = Image.open(file_path).convert("L")
            root_mask = pad_mask_if_needed(root_mask, fc_image.shape, file_id, "root_mask")
            root_mask_array = np.array(root_mask)
            binary_mask = (root_mask_array > 0).astype(np.uint8)

            main_root_mask = load_binary_mask(
                os.path.join(mask_dir, "main_root_mask.png"),
                binary_mask.shape
            )
            main_root_mask = np.array(
                pad_mask_if_needed(main_root_mask, fc_image.shape, file_id, "main_root_mask")
            )

            lateral_root_mask = load_binary_mask(
                os.path.join(mask_dir, "lateral_root_mask.png"),
                binary_mask.shape
            )
            lateral_root_mask = np.array(
                pad_mask_if_needed(lateral_root_mask, fc_image.shape, file_id, "lateral_root_mask")
            )

            tip_mask = load_binary_mask(
                os.path.join(mask_dir, "tip_mask.png"),
                binary_mask.shape
            )
            tip_mask = np.array(
                pad_mask_if_needed(tip_mask, fc_image.shape, file_id, "tip_mask")
            )

            node_mask = load_binary_mask(
                os.path.join(mask_dir, "node_mask.png"),
                binary_mask.shape
            )
            node_mask = np.array(
                pad_mask_if_needed(node_mask, fc_image.shape, file_id, "node_mask")
            )
            node_mask = binary_dilation(node_mask, iterations=5).astype(np.uint8)

            shoot_mask = load_binary_mask(
                os.path.join(mask_dir, "shoot_mask.png"),
                binary_mask.shape
            )
            shoot_mask = np.array(
                pad_mask_if_needed(shoot_mask, fc_image.shape, file_id, "shoot_mask")
            )

            dilated_root_mask = binary_dilation(binary_mask, iterations=50).astype(np.uint8)
            shoot_coords = np.argwhere(shoot_mask > 0)
            if shoot_coords.size > 0:
                shoot_bottom_y = int(shoot_coords[:, 0].max())
                allowed = np.ones_like(dilated_root_mask, dtype=bool)
                allowed[:shoot_bottom_y, :] = False
                dilated_root_mask = np.where(allowed, dilated_root_mask, 0).astype(np.uint8)
            else:
                print("Warning: shoot_mask is empty, no cutoff applied.")

            dilated_border = dilated_root_mask - binary_erosion(dilated_root_mask, iterations=1).astype(np.uint8)
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

            def extract_mask_data(mask: np.ndarray, label: str) -> pd.DataFrame:
                coords = np.argwhere(mask > 0)
                values = fc_image[coords[:, 0], coords[:, 1]] if coords.size > 0 else np.array([], dtype=fc_image.dtype)
                return pd.DataFrame(
                    {
                        "dumm_layer": dumm_layer,
                        "dumm_member": dumm_member_name,
                        "y": coords[:, 0],
                        "x": coords[:, 1],
                        "fluorescence": values,
                        "region": label,
                    }
                )

            mask_dfs = [
                extract_mask_data(main_root_mask, "main_root"),
                extract_mask_data(lateral_root_mask, "lateral_root"),
                extract_mask_data(main_tip_mask, "main_root_tip"),
                extract_mask_data(other_tip_mask, "other_tip"),
                extract_mask_data(node_mask, "node"),
                extract_mask_data(dilated_exclusive, "dilated_root_exclusive"),
                extract_mask_data(dilated_root_mask, "dilated_root"),
                extract_mask_data(shoot_mask, "shoot"),
            ]

            summary_rows: list[list[Any]] = []
            for df in mask_dfs:
                region = df["region"].iloc[0] if not df.empty else "unknown"
                mean_val, n_pixels, sum_val = summarize_region(df)
                summary_rows.extend([
                    [os.path.basename(fc_path), dumm_layer, dumm_member_name, plant_id, f"mean_fluorescence_{region}", mean_val],
                    [os.path.basename(fc_path), dumm_layer, dumm_member_name, plant_id, f"n_pixels_{region}", n_pixels],
                    [os.path.basename(fc_path), dumm_layer, dumm_member_name, plant_id, f"sum_fluorescence_{region}", sum_val],
                ])

            df_summary = pd.DataFrame(
                summary_rows,
                columns=["fc_file", "dumm_layer", "dumm_member", "plant_id", "parameter", "value"],
            )

            output_subfolder = Path(working_directory) / fc_type / file_id_fc / plant_suffix / dumm_layer
            output_subfolder.mkdir(parents=True, exist_ok=True)
            output_prefix = f"{file_id_fc}_{dumm_layer}"

            df_pixels = pd.concat(mask_dfs, ignore_index=True)
            df_pixels.to_csv(output_subfolder / f"{output_prefix}_pixels.csv", index=False)
            df_summary.to_csv(output_subfolder / f"{output_prefix}_summary.csv", index=False)

            overlay_rgb = create_overlay(
                fc_image,
                {
                    "dilated_border": (dilated_border, (255, 128, 0)),
                    "main_root": (main_root_mask, (255, 0, 0)),
                    "lateral_root": (lateral_root_mask, (0, 255, 0)),
                    "main_tip": (main_tip_mask, (0, 0, 255)),
                    "other_tip": (other_tip_mask, (0, 255, 255)),
                    "node": (node_mask, (255, 255, 0)),
                    "shoot": (shoot_mask, (255, 0, 255)),
                },
            )
            Image.fromarray(overlay_rgb.astype(np.uint8)).save(output_subfolder / f"{output_prefix}_overlay.png")

            region_counts = {
                "main_root": int((main_root_mask > 0).sum()),
                "lateral_root": int((lateral_root_mask > 0).sum()),
                "main_root_tip": int((main_tip_mask > 0).sum()),
                "other_tip": int((other_tip_mask > 0).sum()),
                "node": int((node_mask > 0).sum()),
                "dilated_root_exclusive": int((dilated_exclusive > 0).sum()),
                "dilated_root": int((dilated_root_mask > 0).sum()),
                "shoot": int((shoot_mask > 0).sum()),
            }
            plant_params = {
                "analysis_type": "single_root_mask_fluorescence",
                "source_root_mask": file_path,
                "source_fc_tar": fc_path,
                "source_dumm_member": dumm_member_name,
                "dumm_layer": dumm_layer,
                "plant_id": plant_id,
                "root_type": root_type,
                "filter_type": filter_type,
                "alignment_parameters": align_params,
                "preprocessing": {
                    "fc_input_mode": "tar_all_dumm_files",
                    "dumm_right_padding_pixels": 90,
                    "horizontal_flip": True,
                    "fixed_canvas_alignment": True,
                },
                "roi_definition": {
                    "node_binary_dilation_iterations": 5,
                    "tip_binary_dilation_iterations": 5,
                    "peri_root_binary_dilation_iterations": 50,
                    "exclude_peri_root_above_shoot_bottom": True,
                },
                "regions_exported": region_counts,
                "outputs": {
                    "pixels_csv": output_subfolder / f"{output_prefix}_pixels.csv",
                    "summary_csv": output_subfolder / f"{output_prefix}_summary.csv",
                    "overlay_png": output_subfolder / f"{output_prefix}_overlay.png",
                },
            }
            save_run_parameters(output_subfolder, plant_params)

            processed_dumm_count += 1
            print(f"Exported {dumm_layer} results to {output_subfolder}")

    return processed_dumm_count


def main(
    working_directory: str | Path,
    filter_fc1: str = "F513",
    filter_fc2: str = "F635",
    root_set: str = "both",
) -> None:

    valid_root_sets = {"ROOT1", "ROOT2", "both"}
    if root_set not in valid_root_sets:
        raise ValueError("root_set must be 'ROOT1', 'ROOT2', or 'both'")

    working_directory = str(working_directory)
    root_mask_files = list_root_mask_images(working_directory)
    has_root1 = any("ROOT1" in f for f in root_mask_files)
    has_root2 = any("ROOT2" in f for f in root_mask_files)

    fc1_dir = os.path.join(working_directory, "FC1_TAR")
    fc2_dir = os.path.join(working_directory, "FC2_TAR")
    fc_files = list_tar_files(working_directory)

    if root_set != "both":
        root_mask_files = [f for f in root_mask_files if root_set in f]

    if (root_set in {"ROOT1", "both"}) and has_root1 and not os.path.exists(fc1_dir):
        raise FileNotFoundError(f"❌ ROOT1 root masks found but {os.path.basename(fc1_dir)} folder is missing.")
    if (root_set in {"ROOT2", "both"}) and has_root2 and not os.path.exists(fc2_dir):
        raise FileNotFoundError(f"❌ ROOT2 root masks found but {os.path.basename(fc2_dir)} folder is missing.")

    run_params = {
        "analysis_type": "rootcam_single_channel_fluorescence",
        "working_directory": working_directory,
        "root_set": root_set,
        "filter_fc1": filter_fc1,
        "filter_fc2": filter_fc2,
        "fluorescence_input_mode": "tar_all_dumm_files",
        "required_folders": ["ROOT1_analysis", "ROOT2_analysis", "FC1_TAR", "FC2_TAR"],
        "alignment_config": ALIGNMENT_CONFIG,
        "discovered_inputs": {
            "root_mask_files_total": len(root_mask_files),
            "tar_files_total": len(fc_files),
        },
    }
    save_run_parameters(working_directory, run_params, filename="fc_analysis_run_parameters.json")

    total_root_masks = len(root_mask_files)
    matched = summary_csv_count = pixel_csv_count = overlay_count = 0

    print(f"📦 Processing {total_root_masks} root mask files...\n")
    for i, file_path in enumerate(root_mask_files, 1):
        if "ROOT1" in file_path:
            filter_type = filter_fc1
        elif "ROOT2" in file_path:
            filter_type = filter_fc2
        else:
            print(f"⚠️ Unknown ROOT folder in file: {file_path}")
            continue

        output_count = process_single_root_mask(
            file_path=file_path,
            working_directory=working_directory,
            fc_files=fc_files,
            filter_type=filter_type,
            alignment_config=ALIGNMENT_CONFIG,
        )

        if output_count:
            matched += 1
            summary_csv_count += output_count
            pixel_csv_count += output_count
            overlay_count += output_count

        if total_root_masks:
            print(f"Progress: {i}/{total_root_masks} ({(i / total_root_masks) * 100:.1f}%)")

    stats = pd.DataFrame(
        [
            ["root_mask_files_total", total_root_masks],
            ["root_mask_with_fc_match", matched],
            ["dumm_layers_processed", summary_csv_count],
            ["summary_csv_exported", summary_csv_count],
            ["pixels_csv_exported", pixel_csv_count],
            ["overlay_images_created", overlay_count],
        ],
        columns=["description", "value"],
    )
    stats_output = Path(working_directory) / "root_mask_processing_summary.csv"
    stats.to_csv(stats_output, index=False)
    print(f"\n📊 Summary saved to: {stats_output}")


# ---------------------- PyCharm run configuration ----------------------
# Edit these values in PyCharm, then run this file directly.
PYCHARM_SETTINGS = {
    "working_directory": r"C:\Users\admin\OneDrive - Universiteit Utrecht\Desktop\FC_Improvement",
    "filter_fc1": "F483",
    "filter_fc2": "F635",
    "root_set": "ROOT2",  # choose from: "ROOT1", "ROOT2", "both"
}


if __name__ == "__main__":
    settings = PYCHARM_SETTINGS.copy()
    main(
        working_directory=settings["working_directory"],
        filter_fc1=settings["filter_fc1"],
        filter_fc2=settings["filter_fc2"],
        root_set=settings["root_set"],
    )
