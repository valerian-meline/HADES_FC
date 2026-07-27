# FC Improvement

Utilities for combining FluorCam `.tar` acquisitions with root mask outputs and exporting fluorescence measurements per root region.

`FC_analysis.py` reads FluorCam TAR files, reads every `.dumm` image inside each TAR, reconstructs and aligns the fluorescence image, applies plant/root masks from the root-analysis output, and exports pixel-level fluorescence values, summary statistics, overlay images, run metadata, and alignment QC. Multiple `.dumm` files in one TAR are processed as separate layers named `Ft_1`, `Ft_2`, and so on.

This branch keeps the biological ROI definitions of the previous FC workflow, but makes the script safer and faster for larger HADES experiments:

- work is grouped by frame/TAR instead of by plant, so the same FC image is not repeatedly read and aligned for every plant;
- alignment filters are required explicitly for every active root set, because the filter determines the alignment configuration;
- pixel-level output is produced by default as Parquet only, not CSV;
- per-root summaries, master summaries, overlays, alignment QC, and run logs are written automatically;
- the script can run one root set or both root sets from the command line.

Raw data and generated outputs are intentionally not tracked by Git.

## Repository contents

```text
FC_analysis.py        Main FC analysis script
parquet_to_csv.py     Optional utility for converting pixel Parquet files to CSV
requirements.txt      pip dependency list
environment.yml       conda environment definition
.gitignore            Excludes local data, outputs, IDE files, and virtualenvs
```

## Installation

Using conda:

```powershell
conda env create -f environment.yml
conda activate fc-improvement
```

Using pip:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

The default pixel-level output is Parquet, so the environment must include a Parquet writer. The recommended dependency is `pyarrow`.

Minimal package list:

```text
numpy>=1.26
pandas>=2.1
pyarrow>=14
pillow>=10.0
scipy>=1.11
scikit-image>=0.22
matplotlib>=3.8
```

## Expected input layout

Place root-analysis folders and FluorCam folders inside one experiment directory:

```text
experiment/
  ROOT1_analysis/
    <frame_id>/
      plant_1/root_mask.png
      plant_1/main_root_mask.png
      plant_1/lateral_root_mask.png
      plant_1/tip_mask.png
      plant_1/node_mask.png
      plant_1/shoot_mask.png
      plant_1/root_area_mask.png          optional
      plant_2/...
  ROOT2_analysis/
    <frame_id>/
      plant_1/root_mask.png
      ...
  FC1_TAR/
    *.tar
  FC2_TAR/
    *.tar
```

The script also searches common HADES acquisition layouts, for example:

```text
experiment/FC1/Measurement/TARs
experiment/FC1_TAR/Measurement/TARs
experiment/FC1/Measurement
experiment/FC1_TAR/Measurement
experiment/FC1_TAR
experiment/FC1
```

The same patterns are used for `FC2`.

Root mask folders are matched to FluorCam TAR files by experiment/frame naming. If a frame has multiple plants, the FC TAR is read and aligned once, then reused for all `plant_<number>` folders in that frame.

## Required alignment filters

The FC filter selects the alignment configuration. It is not just metadata. If the wrong filter is used, the FC image and root masks will be systematically misaligned.

For that reason, there is no default filter. Every active root set must have an explicit filter.

Available filters:

```text
ROOT1 / FC1: F483, F513, F635
ROOT2 / FC2: F513, F593, F635
```

Recommended command for both root systems:

```powershell
python FC_analysis.py `
  --input C:\path\to\experiment `
  --root-set both `
  --filter-root1 F513 `
  --filter-root2 F635
```

Equivalent legacy-style option names are accepted:

```powershell
python FC_analysis.py `
  --input C:\path\to\experiment `
  --root-set both `
  --filter-fc1 F513 `
  --filter-fc2 F635
```

If a required filter is missing in an interactive terminal, the script asks for it before processing starts. If a required filter is missing in a non-interactive run, such as a scheduler job or redirected shell, the script exits with an error instead of guessing.

## Minimal one-line usage

For most full experiments, the simplest command is one line. Replace the experiment path and filters with the values used for the acquisition:

```powershell
python FC_analysis.py --input C:\path\to\experiment --root-set both --filter-root1 F513 --filter-root2 F635
```

For a quick alignment/overlay QC run without pixel Parquet output:

```powershell
python FC_analysis.py --input C:\path\to\experiment --root-set both --filter-root1 F513 --filter-root2 F635 --pixel-export none
```

For a conservative run on a laptop or shared workstation:

```powershell
python FC_analysis.py --input C:\path\to\experiment --root-set both --filter-root1 F513 --filter-root2 F635 --workers 4 --worker-profile conservative
```

Always use the actual FC filter for each root set. The filter is an alignment setting, not just a label.

## Basic usage

Dry run first to check discovery, matching, filters, output folders, and worker plan without processing images:

```powershell
python FC_analysis.py `
  --input C:\path\to\experiment `
  --root-set both `
  --filter-root1 F513 `
  --filter-root2 F635 `
  --dry-run-plan
```

Run ROOT1 only:

```powershell
python FC_analysis.py `
  --input C:\path\to\experiment\ROOT1 `
  --filter-root1 F513
```

Run ROOT2 only:

```powershell
python FC_analysis.py `
  --input C:\path\to\experiment\ROOT2 `
  --filter-root2 F635
```

Run both root systems:

```powershell
python FC_analysis.py `
  --input C:\path\to\experiment `
  --root-set both `
  --filter-root1 F513 `
  --filter-root2 F635
```

Override the FC TAR directory for a copied or test dataset:

```powershell
python FC_analysis.py `
  --input C:\path\to\experiment\ROOT2 `
  --filter-root2 F635 `
  --fc-dir D:\path\to\FC2\Measurement\TARs
```

## Pixel-level output

Pixel-level output is enabled by default and is written as Parquet:

```text
--pixel-export parquet
```

Each plant and DUMM layer receives one long-format pixel file:

```text
FC1_analysis/<fc_id>/plant1/Ft_1/<fc_id>_Ft_1_pixels.parquet
```

Pixel Parquet columns:

```text
root_type
fc_id
frame_id
plant_id
filter_type
dumm_layer
dumm_member
region
y
x
fluorescence
```

One row represents one pixel in one ROI region. The same physical pixel can appear in multiple rows if it belongs to multiple regions, because regions are measured independently.

To disable pixel-level output for a quick QC-only run:

```powershell
python FC_analysis.py `
  --input C:\path\to\experiment `
  --root-set both `
  --filter-root1 F513 `
  --filter-root2 F635 `
  --pixel-export none
```

The main pipeline does not write pixel CSV files. If CSV is needed for legacy or manual inspection, convert after the run:

```powershell
python parquet_to_csv.py C:\path\to\experiment\FC1_analysis --recursive
```

or convert a single file:

```powershell
python parquet_to_csv.py C:\path\to\experiment\FC1_analysis\<fc_id>\plant1\Ft_1\<fc_id>_Ft_1_pixels.parquet
```

## Summary output

For each plant and DUMM layer, the script writes a summary CSV:

```text
FC1_analysis/<fc_id>/plant1/Ft_1/<fc_id>_Ft_1_summary.csv
```

The summary file is long-format. For each ROI region it contains:

```text
mean_fluorescence_<region>
n_pixels_<region>
sum_fluorescence_<region>
```

Typical regions include:

```text
main_root
lateral_root
main_root_tip
other_tip
node
dilated_root_exclusive
dilated_root
shoot
root_area, if root_area_mask.png exists
```

`root_area` is optional and is only exported when the root pipeline has produced `root_area_mask.png` for that plant. Existing legacy regions are still exported.

## Master summaries

A master summary is written by default for each FC output folder:

```text
FC1_analysis/master_summary.csv
FC2_analysis/master_summary.csv
```

If both root systems are processed in one run, the script also writes a combined file:

```text
experiment/FC_master_summary.csv
```

The master summary preserves the long-format summary rows and adds context columns such as:

```text
root_type
fc_id
frame_id
plant_folder
dumm_layer_folder
summary_csv
run_parameters_json
source_root_mask
source_fc_tar
filter_type
```

To disable master summary creation:

```powershell
--no-master-summary
```

To change the master summary filename:

```powershell
--master-summary-name my_master_summary.csv
```

## Overlay and alignment QC output

Each plant and DUMM layer receives an overlay image:

```text
FC1_analysis/<fc_id>/plant1/Ft_1/<fc_id>_Ft_1_overlay.png
```

The overlay shows the aligned fluorescence image with root, tip, node, shoot, and peri-root regions overlaid for visual QC.

Frame-level alignment QC is written once per frame/TAR/DUMM layer:

```text
FC1_analysis/<fc_id>/_frame_qc/Ft_1/<fc_id>_Ft_1_aligned_fc_preview.png
FC1_analysis/<fc_id>/_frame_qc/Ft_1/alignment_qc.json
```

`alignment_qc.json` records the selected filter and exact alignment parameters. Check this file first if overlays look systematically shifted.

Preview options:

```powershell
--fc-preview none
--fc-preview aligned
--fc-preview raw
--fc-preview both
```

The default is:

```text
--fc-preview aligned
```

## Run logs and progress

Run logs are written under:

```text
experiment/FC_analysis_logs/<timestamp>/
```

Important files:

```text
fc_start_context.json   startup settings and resolved paths
fc_run_plan.json        discovered jobs and worker plan
fc_job_manifest.jsonl   one JSON record per processed job
```

In an interactive terminal, progress is shown as a single updating progress bar. If output is redirected to a log file, the script writes sparse progress lines instead of thousands of updates.

Useful options:

```powershell
--progress-style auto
--progress-style bar
--progress-style log
--progress-style none
--progress-log-every 500
```

## Parallel processing and resource use

The script is parallelized at the frame/TAR level. Each worker processes one FC TAR/frame job at a time: it reads the `.dumm` layer, reconstructs and aligns the FC image once, then applies that aligned image to all plant masks for the same frame. This avoids repeatedly reading and aligning the same FC image for every plant.

Worker options:

```powershell
--workers auto
--workers 4
--worker-profile conservative
--worker-profile balanced
--worker-profile aggressive
```

Recommended defaults:

```text
Use --workers auto for a normal workstation or server.
Use --worker-profile conservative or a fixed small worker count if the machine is shared, low on RAM, or writing to a slow external drive.
Use --dry-run-plan before a large run to inspect the discovered jobs and worker plan.
```

Important resource notes:

- More workers can improve throughput, but each active worker holds an aligned FC image, masks, extracted pixel tables, and temporary arrays. Very high worker counts can increase RAM pressure.
- Pixel-level Parquet is enabled by default and can produce many files and a large amount of data for full experiments. This is expected, but it can stress slow disks or network drives.
- If the goal is only to check alignment and overlays, run with `--pixel-export none` first.
- If the computer becomes unresponsive, rerun with fewer workers, for example `--workers 2` or `--workers 4`.
- If output is written to an external or network drive, a conservative worker profile is usually safer than maximizing CPU usage.

Resource-conscious examples:

```powershell
python FC_analysis.py --input C:\path\to\experiment --root-set both --filter-root1 F513 --filter-root2 F635 --dry-run-plan
```

```powershell
python FC_analysis.py --input C:\path\to\experiment --root-set both --filter-root1 F513 --filter-root2 F635 --workers 4 --worker-profile conservative
```

```powershell
python FC_analysis.py --input C:\path\to\experiment --root-set both --filter-root1 F513 --filter-root2 F635 --pixel-export none --workers 4
```

## Output layout example

```text
experiment/
  FC1_analysis/
    master_summary.csv
    <fc_id>/
      _frame_qc/
        Ft_1/
          <fc_id>_Ft_1_aligned_fc_preview.png
          alignment_qc.json
      plant1/
        Ft_1/
          <fc_id>_Ft_1_pixels.parquet
          <fc_id>_Ft_1_summary.csv
          <fc_id>_Ft_1_overlay.png
          run_parameters.json
      plant2/
        Ft_1/
          ...
  FC2_analysis/
    master_summary.csv
    ...
  FC_analysis_logs/
    <timestamp>/
      fc_start_context.json
      fc_run_plan.json
      fc_job_manifest.jsonl
  FC_master_summary.csv
```

## Troubleshooting

### Missing filter error

There is no default alignment filter. Provide the filter for every active root set:

```powershell
--filter-root1 F513
--filter-root2 F635
```

### Parquet error

If the script cannot write Parquet, install `pyarrow` in the active environment:

```powershell
python -m pip install pyarrow
```

### No matching TAR

Check that the FC TAR folder corresponds to the root set and that the filename contains the expected frame identifier. Use `--dry-run-plan` to inspect discovered root frames, FC TAR files, and matching decisions before running the full analysis.

### Overlay shifted

Check these first:

1. Was the correct filter selected for the root set?
2. Does `alignment_qc.json` show the expected `filter_type` and alignment parameters?
3. Are the ROOT and FC folders from the same experiment and acquisition round?

### Too much output or high resource use

Pixel Parquet is the default biological output. For a fast QC-only run, disable it:

```powershell
--pixel-export none
```

For high RAM use or slow disks, reduce parallelism:

```powershell
--workers 4 --worker-profile conservative
```

