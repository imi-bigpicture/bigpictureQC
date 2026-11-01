# BigpictureQC: multi-stain, multi-species artifact segmentation for whole-slide quality control

This repository contains the implementation of the paper [**BigpictureQC: multi-stain, multi-species artifact segmentation for whole-slide quality control**](https://doi.org/10.1186/s12916-026-05257-1) (*BMC Medicine*, 2026). It provides an end-to-end inference pipeline for whole-slide image (WSI) quality control: tissue segmentation, artifact segmentation, and per-slide quality scoring.

![Bigpicture QC figure](figures/bigpictureqc-plot.png)

## ✨ Key features

- **6-class artifact segmentation**: tissue folds, ink, air bubbles, dust, marker, out-of-focus regions
- **Multi-stain and multi-species**: trained and validated across diverse staining protocols and tissue types
- **Per-slide quality scores**: overall quality and artifact scores computed automatically from predicted masks
- **Reproducible Docker deployment** to run inference on new slides with minimal setup
- **Training code included** for model development and fine-tuning

## 🚀 Getting started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### 1. Clone the repository

```bash
git clone https://github.com/imi-bigpicture/bigpictureQC.git
cd bigpictureQC
```

All commands below are run from this folder.

### 2. Download the model weights

The model weights are available on Zenodo: [https://doi.org/10.5281/zenodo.22734334](https://doi.org/10.5281/zenodo.22734334).

Download the three files into the `weights/` folder of this repository (it is empty in a fresh clone). Keep the file names exactly as they are:

```
weights/
├── tissue_network_unet_mobilenetv2.pth
├── tissue_network_unet_mobilenetv2.yaml
└── artifact_network_unetplusplus_efficientnetb2.pth
```

| File | Description | MD5 checksum |
|---|---|---|
| `tissue_network_unet_mobilenetv2.pth` | Tissue/background segmentation model | `de5ead08b2f23091f4dd4f8739e188d2` |
| `tissue_network_unet_mobilenetv2.yaml` | Tissue model configuration (required to load the tissue model) | `b69233035d234abfe2b7dc49b8d54351` |
| `artifact_network_unetplusplus_efficientnetb2.pth` | Artifact segmentation model | `e8272c15c82717c3465437f0589ede86` |

The weights are not included in the Docker image: they are mounted into the container when you run it (step 4). You can also keep them in another folder and mount that folder instead.

### 3. Build the Docker image

```bash
docker build -t bigpictureqc:latest .
```

### 4. Run inference

Put your slides in a folder and run:

```bash
docker run --rm --gpus all \
  -v /path/to/slides:/inputs \
  -v /path/to/output:/outputs \
  -v "$(pwd)/weights":/home/user/weights \
  bigpictureqc:latest --postprocess
```

Replace `/path/to/slides` and `/path/to/output` with folders on your computer. Each `-v <folder on your computer>:<folder in the container>` option makes a folder visible inside the container: your slides appear as `/inputs`, results are written to `/outputs`, and the weights are read from `/home/user/weights`. Only change the part **before** the colon.

When the run finishes, `/path/to/output` contains a `tissue/` and an `artifacts/` folder (see [Outputs](#-outputs)). `--postprocess` applies region filtering and hole filling to the artifact masks.


## 🛠️ Usage

The container runs `scripts/run_inference.sh`. Without extra arguments, it processes all slides in `/inputs` and writes the results to `/outputs/tissue` and `/outputs/artifacts`, as in [step 4](#4-run-inference). Flags (see [All flags](#all-flags)) are appended after the image name.

> **Paths after the image name are paths inside the container.** In the examples below, arguments such as `/inputs`, `/outputs/tissue` or `/tissue` refer to the container side of a `-v` option (the part after the colon), not to folders on your computer.

### Explicit input and output folders

Instead of relying on the default locations, you can pass the input folder, the tissue-mask output folder and the artifact-mask output folder explicitly (in this order), followed by any flags:

```bash
docker run --rm --gpus all \
  -v /path/to/slides:/inputs \
  -v /path/to/output:/outputs \
  -v "$(pwd)/weights":/home/user/weights \
  bigpictureqc:latest \
  /inputs /outputs/tissue /outputs/artifacts --postprocess
```

### Recursive subfolders

If slides are organised in case subfolders (e.g. `slides/case_001/slide.svs`), recursive mode is **auto-detected** when no files match at the top level. You can also set it explicitly:

```bash
docker run --rm --gpus all \
  -v /path/to/cases:/inputs \
  -v /path/to/output:/outputs \
  -v "$(pwd)/weights":/home/user/weights \
  bigpictureqc:latest --postprocess --recursive
```

### Reusing existing tissue masks

Tissue segmentation always runs first by default. If you already have tissue masks (for example from a previous run), you can skip that step and run only artifact inference. Pass the folder that contains the tissue masks as the second positional argument and add `--skip-tb`:

```bash
docker run --rm --gpus all \
  -v /path/to/slides:/inputs \
  -v /path/to/existing_tissue_masks:/tissue \
  -v /path/to/output:/outputs \
  -v "$(pwd)/weights":/home/user/weights \
  bigpictureqc:latest \
  /inputs /tissue /outputs/artifacts --postprocess --skip-tb
```

Tissue masks must be named `{slide_id}_tissue.tif`, where `{slide_id}` is the slide file name without its extension (change the suffix with `--tb_suffix`). Masks encoded as `{0 = background, 1 = tissue}` or `{1 = background, 2 = tissue}` are both supported; the encoding is detected automatically.

### Re-running

Slides whose output mask already exists are **skipped**, so an interrupted run can simply be restarted. To reprocess a slide (for example with different flags), delete its existing output masks first, or write to a new output folder.

### Single-file mode

Pass the slide and the two output file paths explicitly:

```bash
docker run --rm --gpus all \
  -v /path/to/slides:/inputs \
  -v /path/to/output:/outputs \
  -v "$(pwd)/weights":/home/user/weights \
  bigpictureqc:latest \
  /inputs/slide.svs \
  /outputs/tissue/slide_tissue.tif \
  /outputs/artifacts/slide_artifact.tif \
  --postprocess
```

### DICOM mode

If each case is a subfolder containing DICOM series files, one `.dcm` file is selected per subfolder:

```bash
docker run --rm --gpus all \
  -v /path/to/dicom_cases:/inputs \
  -v /path/to/output:/outputs \
  -v "$(pwd)/weights":/home/user/weights \
  bigpictureqc:latest --postprocess --dicom
```

### All flags

| Flag | Description |
|---|---|
| `--recursive` | Recursively search subdirectories for slides (auto-enabled if no top-level matches) |
| `--dicom` | Treat each subfolder as one DICOM case; pick one `.dcm` per subfolder |
| `--postprocess` | Apply region filtering and hole filling after artifact inference (recommended) |
| `--use_tta` | Test-time augmentation for artifact inference (8 rotations/flips; slower) |
| `--skip-tb` | Skip tissue segmentation (use when tissue masks already exist) |
| `--skip-art` | Skip artifact inference (use when only tissue masks are needed) |
| `--tb_suffix SUFFIX` | Suffix appended to the slide name for tissue mask files (default: `_tissue`) |
| `--spacing_tolerance FLOAT` | Relative tolerance for matching WSI resolution levels (default: `0.25`); increase if slides are skipped with a spacing mismatch warning |
| `--input_filter REGEX` | Override default file filter (default: `.*\.(tif\|tiff\|svs\|mrxs\|ndpi\|scn\|vms\|vmu\|bif)$`) |

## 📊 Outputs

| Path | Description |
|---|---|
| `/outputs/tissue/{image}_tissue.tif` | Tissue mask (suffix configurable via `--tb_suffix`) |
| `/outputs/artifacts/{image}_artifact.tif` | Artifact mask |
| `/outputs/artifacts/artifact_quality_metrics.yaml` | Per-slide quality metrics |

`{image}` is the slide file name without its extension. Both masks are pyramidal TIFF files that can be opened as overlays in viewers such as [ASAP](https://github.com/computationalpathologygroup/ASAP) or [QuPath](https://qupath.github.io/).

### Tissue mask

Written at the slide's pyramid level closest to 4 µm/pixel. Pixel values: `0` = background, `1` = tissue.

### Artifact mask

Written at the slide's pyramid level closest to 2 µm/pixel. Each pixel holds one label:

| Value | Class |
|---|---|
| `0` | Background (outside the tissue mask) |
| `1` | Artifact-free tissue |
| `2` | Tissue fold |
| `3` | Ink |
| `4` | Air bubble |
| `5` | Dust |
| `6` | Marker |
| `7` | Out-of-focus |

### Quality metrics


```yaml
slide_name:
  slide_id: slide_name
  artifact_proportion: 0.2155
  quality_score: 0.7845
  quality_score_excl_ink: 0.8477
  total_tissue_pixels: 5000000
  artifact_proportions:
    tissue_folds: 0.085
    ink: 0.0632
    air_bubbles: 0.032
    dust: 0.018
    marker: 0.009
    out_of_focus: 0.0083
```

| Field | Meaning |
|---|---|
| `slide_id` | Slide file name without extension |
| `total_tissue_pixels` | Number of tissue pixels (labels 1 to 7) at the level used for the metrics |
| `artifact_proportions.<class>` | Fraction of the tissue covered by that artifact class (0 to 1) |
| `artifact_proportion` | Fraction of the tissue covered by any artifact (sum of the six classes) |
| `quality_score` | `1 - artifact_proportion` (1 = no artifacts detected) |
| `quality_score_excl_ink` | Same as `quality_score`, but ink is not counted as an artifact |

## 🧪 Training

Training configuration is in `config/default.yaml`. To train:

```bash
python src/bigpictureQC/train.py --config config/default.yaml
```

See `config/default.yaml` for the full set of options including augmentation, loss, scheduler, and W&B tracking.

## 📄 Citation

If you use BigpictureQC in your research, please cite:

> D'Amato, M., Hoefling, H., Barale-Thomas, E. et al. BigpictureQC: multi-stain, multi-species artifact segmentation for whole-slide quality control. *BMC Med* (2026). https://doi.org/10.1186/s12916-026-05257-1

```bibtex
@article{damato2026bigpictureqc,
  title   = {BigpictureQC: multi-stain, multi-species artifact segmentation for whole-slide quality control},
  author  = {D'Amato, M. and Hoefling, H. and Barale-Thomas, E. and others},
  journal = {BMC Medicine},
  year    = {2026},
  doi     = {10.1186/s12916-026-05257-1}
}
```

Model weights: D'Amato, M. et al. BigpictureQC model weights. Zenodo (2026). https://doi.org/10.5281/zenodo.22734334

## ⚖️ License

The code in this repository is released under the [Apache License 2.0](LICENSE). 

## 💬 Contact

For questions, bug reports or feature requests, please open an [issue](https://github.com/imi-bigpicture/bigpictureQC/issues).

## 🏆 Acknowledgments

This project has received funding from the Innovative Medicines Initiative 2 Joint Undertaking under grant agreement No 945358. This Joint Undertaking receives support from the European Union’s Horizon 2020 research and innovation program and EFPIA ([www.imi.europa.eu](https://www.imi.europa.eu)).

This work builds upon contributions from the following repositories:
- [pathology-whole-slide-data](https://github.com/DIAGNijmegen/pathology-whole-slide-data): on-the-fly WSI patch extraction and data loading
- [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch): segmentation model architectures and encoders
