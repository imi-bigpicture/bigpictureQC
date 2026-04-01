#!/usr/bin/env bash
set -euo pipefail

DEFAULT_INPUT="/inputs"
DEFAULT_TB_OUT="/outputs/tissue"
DEFAULT_ART_OUT="/outputs/artifacts"

DEFAULT_INPUT_FILTER='.*\.(tif|tiff|svs|mrxs|ndpi|scn|vms|vmu|bif)$'

# Flags
DICOM=0
POSTPROCESS=0
USE_TTA=0
RECURSIVE=0
SKIP_TB=0
SKIP_ART=0
TB_SUFFIX="_tissue"
SPACING_TOLERANCE="0.25"
INPUT_FILTER="$DEFAULT_INPUT_FILTER"

usage() {
  cat <<EOF
Usage:
  run_inference.sh [--dicom] [--postprocess] [--use_tta] [--recursive] [--skip-tb] [--skip-art] [--tb_suffix SUFFIX] [--spacing_tolerance FLOAT] [--input_filter REGEX] [INPUT_PATH [TB_OUTPUT ART_OUTPUT]]

Modes:
  - If INPUT_PATH is a file -> single mode
  - If INPUT_PATH is a directory -> folder mode
     - with --dicom: treat each subfolder as one case (pick 1 .dcm per subfolder)
     - with --recursive: search subdirectories for images (auto-detected if no direct matches found)

Defaults:
  INPUT_PATH   = $DEFAULT_INPUT
  TB_OUTPUT    = $DEFAULT_TB_OUT
  ART_OUTPUT   = $DEFAULT_ART_OUT
  input_filter = $DEFAULT_INPUT_FILTER

Notes:
  --postprocess and --use_tta currently apply to ARTIFACT inference only.
EOF
}

POSITIONALS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dicom)
      DICOM=1; shift ;;
    --postprocess)
      POSTPROCESS=1; shift ;;
    --use_tta)
      USE_TTA=1; shift ;;
    --recursive)
      RECURSIVE=1; shift ;;
    --skip-tb)
      SKIP_TB=1; shift ;;
    --skip-art)
      SKIP_ART=1; shift ;;
    --tb_suffix)
      TB_SUFFIX="$2"; shift 2 ;;
    --spacing_tolerance)
      SPACING_TOLERANCE="$2"; shift 2 ;;
    --input_filter)
      INPUT_FILTER="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      POSITIONALS+=("$1"); shift ;;
  esac
done
set -- "${POSITIONALS[@]:-}"

INPUT_PATH="${1:-$DEFAULT_INPUT}"
TB_OUTPUT="${2:-$DEFAULT_TB_OUT}"
ART_OUTPUT="${3:-$DEFAULT_ART_OUT}"

if [[ -f "$INPUT_PATH" ]]; then
  MODE="single"
elif [[ -d "$INPUT_PATH" ]]; then
  MODE="folder"
else
  echo "ERROR: INPUT_PATH '$INPUT_PATH' is neither a file nor a directory." >&2
  exit 1
fi

# Auto-detect recursive: if folder mode, not DICOM, not already set, and no files match at the top level
if [[ "$MODE" == "folder" && "$DICOM" == "0" && "$RECURSIVE" == "0" ]]; then
  DIRECT_COUNT=$(find "$INPUT_PATH" -maxdepth 1 -type f -print0 \
    | grep -czP "$INPUT_FILTER" 2>/dev/null || true)
  if [[ "$DIRECT_COUNT" -eq 0 ]]; then
    echo "[RUN] No direct matches at top level — enabling recursive search"
    RECURSIVE=1
  fi
fi

echo "[RUN] MODE         = $MODE"
echo "[RUN] INPUT_PATH   = $INPUT_PATH"
echo "[RUN] TB_OUTPUT    = $TB_OUTPUT"
echo "[RUN] ART_OUTPUT   = $ART_OUTPUT"
echo "[RUN] DICOM        = $DICOM"
echo "[RUN] POSTPROCESS  = $POSTPROCESS"
echo "[RUN] USE_TTA      = $USE_TTA"
echo "[RUN] RECURSIVE    = $RECURSIVE"
echo "[RUN] SKIP_TB      = $SKIP_TB"
echo "[RUN] SKIP_ART     = $SKIP_ART"
echo "[RUN] TB_SUFFIX         = $TB_SUFFIX"
echo "[RUN] SPACING_TOLERANCE = $SPACING_TOLERANCE"
echo "[RUN] INPUT_FILTER      = $INPUT_FILTER"

export INPUT_PATH TB_OUTPUT ART_OUTPUT MODE DICOM POSTPROCESS USE_TTA RECURSIVE SKIP_TB SKIP_ART TB_SUFFIX SPACING_TOLERANCE INPUT_FILTER

if [[ "$SKIP_TB" == "0" ]]; then
  /home/user/generate_tb_mask.sh
else
  echo "[RUN] Skipping tissue segmentation (--skip-tb)"
fi

if [[ "$SKIP_ART" == "0" ]]; then
  /home/user/generate_artifact_mask.sh
else
  echo "[RUN] Skipping artifact detection (--skip-art)"
fi
