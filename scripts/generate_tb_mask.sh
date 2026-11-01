#!/usr/bin/env bash
set -euo pipefail

for v in INPUT_PATH TB_OUTPUT MODE DICOM INPUT_FILTER TB_SUFFIX; do
  if [[ -z "${!v:-}" ]]; then
    echo "[TB] ERROR: $v must be set by run_inference.sh" >&2
    exit 1
  fi
done

MODEL_PATH="/home/user/weights/tissue_network_unet_mobilenetv2.pth"

if [[ "$MODE" == "folder" ]]; then
  mkdir -p "$TB_OUTPUT"
  OUTPUT_PATH="$TB_OUTPUT/{image}${TB_SUFFIX}.tif"
  FILTER_ARG=(--input_filter "$INPUT_FILTER")
else
  mkdir -p "$(dirname "$TB_OUTPUT")"
  OUTPUT_PATH="$TB_OUTPUT"
  FILTER_ARG=()
fi

DICOM_ARG=()
if [[ "$MODE" == "folder" && "$DICOM" == "1" ]]; then
  DICOM_ARG=(--dicom)
fi

RECURSIVE_ARG=()
if [[ "$MODE" == "folder" && "${RECURSIVE:-0}" == "1" ]]; then
  RECURSIVE_ARG=(--recursive)
fi

echo "[TB] MODE        = $MODE"
echo "[TB] INPUT_PATH  = $INPUT_PATH"
echo "[TB] OUTPUT_PATH = $OUTPUT_PATH"
echo "[TB] MODEL       = $MODEL_PATH"
echo "[TB] DICOM       = $DICOM"

exec /usr/local/bin/python3 /home/user/source/pathology-fast-inference/scripts/applynetwork_multiproc.py \
  --input_wsi_path="$INPUT_PATH" \
  --spacing_tolerance="${SPACING_TOLERANCE:-0.25}" \
  "${FILTER_ARG[@]}" \
  "${DICOM_ARG[@]}" \
  "${RECURSIVE_ARG[@]}" \
  --output_wsi_path="$OUTPUT_PATH" \
  --model_path="$MODEL_PATH" \
  --read_spacing=4.0 \
  --write_spacing=4.0 \
  --tile_size=512 \
  --readers=5 \
  --writers=5 \
  --batch_size=70 \
  --gpu_count=1 \
  --axes_order='cwh' \
  --custom_processor="torch_processor" \
  --reconstruction_information="[[0,0,0,0],[1,1],[100,100,100,100]]" \
  --quantize
