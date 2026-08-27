`FC_analysis.py`, reads FluorCam tar files, extracts every `.dumm` image inside each tar, reconstructs and aligns the fluorescence image, applies plant/root masks, and exports pixel-level values, summary statistics, and overlay images. Multiple `.dumm` files in one tar are processed as separate layers named `Ft_1`, `Ft_2`, and so on.

## Repository Contents

```text
FC_analysis.py       Main analysis script
requirements.txt     pip dependency list
environment.yml      conda environment definition
.gitignore           Excludes local data, outputs, IDE files, and virtualenvs
```

## Expected Input Layout

Place the analysis folders inside one working directory:

```text
working_directory/
  FC1_TAR/
    *.tar
  FC2_TAR/
    *.tar
  ROOT1_analysis/
    .../plant_1/root_mask.png
    .../plant_1/main_root_mask.png
    .../plant_1/lateral_root_mask.png
    .../plant_1/tip_mask.png
    .../plant_1/node_mask.png
    .../plant_1/shoot_mask.png
  ROOT2_analysis/
    .../plant_1/root_mask.png
    ...
```

The script matches root mask folders to FluorCam tar files by experiment/file naming.

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

## Usage

```python
PYCHARM_SETTINGS = {
    "working_directory": r"C:\path\to\your\working_directory",
    "filter_fc1": "F483",
    "filter_fc2": "F635",
    "root_set": "ROOT2",
}
```

```powershell
python FC_analysis.py
```

`root_set` can be:

```text
ROOT1
ROOT2
both
```

## Outputs

```text
FC1_analysis/
FC2_analysis/
```

For each plant and each `.dumm` layer, outputs are grouped by layer:

```text
FC2_analysis/<fc_file_id>/plant1/Ft_1/
  <fc_file_id>_Ft_1_pixels.csv
  <fc_file_id>_Ft_1_summary.csv
  <fc_file_id>_Ft_1_overlay.png
  run_parameters.json

FC2_analysis/<fc_file_id>/plant1/Ft_2/
  <fc_file_id>_Ft_2_pixels.csv
  <fc_file_id>_Ft_2_summary.csv
  <fc_file_id>_Ft_2_overlay.png
  run_parameters.json
```

