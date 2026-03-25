#!/usr/bin/env bash
set -euo pipefail

for v in INPUT_PATH TB_OUTPUT ART_OUTPUT MODE DICOM INPUT_FILTER POSTPROCESS USE_TTA; do
  if [[ -z "${!v:-}" ]]; then
    echo "[ART] ERROR: $v must be set by run_inference.sh" >&2
    exit 1
  fi
done

MODEL_PATH="/home/user/weights/artifact_network_unet++_efficientnetb2.pth"

if [[ "$MODE" == "folder" ]]; then
  mkdir -p "$ART_OUTPUT"
  OUTPUT_PATH="$ART_OUTPUT/{image}_artifact.tif"
  MASK_PATH="$TB_OUTPUT/{image}_tissue.tif"
  FILTER_ARG=(--input_filter "$INPUT_FILTER")
else
  mkdir -p "$(dirname "$ART_OUTPUT")"
  OUTPUT_PATH="$ART_OUTPUT"
  MASK_PATH="$TB_OUTPUT"
  FILTER_ARG=()
fi

DICOM_ARG=()
if [[ "$MODE" == "folder" && "$DICOM" == "1" ]]; then
  DICOM_ARG=(--dicom)
fi

POSTPROCESS_ARG=()
if [[ "$POSTPROCESS" == "1" ]]; then
  POSTPROCESS_ARG=(--postprocess)
fi

USE_TTA_ARG=()
if [[ "$USE_TTA" == "1" ]]; then
  USE_TTA_ARG=(--use_tta)
fi

echo "[ART] MODE        = $MODE"
echo "[ART] INPUT_PATH  = $INPUT_PATH"
echo "[ART] OUTPUT_PATH = $OUTPUT_PATH"
echo "[ART] MASK_PATH   = $MASK_PATH"
echo "[ART] MODEL       = $MODEL_PATH"
echo "[ART] DICOM       = $DICOM"
echo "[ART] POSTPROCESS = $POSTPROCESS"
echo "[ART] USE_TTA     = $USE_TTA"

MASK_CLASS=""

if [[ "$MODE" == "folder" ]]; then
  # detect from the first tissue mask in the output folder
  FIRST_MASK="$(find "$TB_OUTPUT" -type f \( -name "*_tissue.tif" -o -name "*.tif" \) | head -n 1)"
  if [[ -z "${FIRST_MASK:-}" ]]; then
    echo "[ART] ERROR: could not find any tissue mask in $TB_OUTPUT" >&2
    exit 1
  fi
  MASK_CLASS="$(/usr/local/bin/python3 /home/user/scripts/detect_mask_class.py --mask "$FIRST_MASK")"
else
  MASK_CLASS="$(/usr/local/bin/python3 /home/user/scripts/detect_mask_class.py --mask "$TB_OUTPUT")"
fi

echo "[ART] MASK_CLASS  = $MASK_CLASS"

/usr/local/bin/python3 /home/user/source/pathology-fast-inference/scripts/applynetwork_multiproc.py \
  --input_wsi_path="$INPUT_PATH" \
  "${FILTER_ARG[@]}" \
  "${DICOM_ARG[@]}" \
  --mask_wsi_path="$MASK_PATH" \
  --output_wsi_path="$OUTPUT_PATH" \
  --model_path="$MODEL_PATH" \
  --read_spacing=2.0 \
  --write_spacing=2.0 \
  --tile_size=512 \
  --readers=5 \
  --writers=5 \
  --batch_size=32 \
  --gpu_count=1 \
  --axes_order='cwh' \
  --custom_processor="artifact_processor" \
  --reconstruction_information="[[0,0,0,0],[1,1],[100,100,100,100]]" \
  --quantize \
  --decoder_name-artifact="unetplusplus" \
  --encoder_name-artifact="efficientnet-b2" \
  --mask_class="$MASK_CLASS" \
  --normalizer="imagenet" \
  "${POSTPROCESS_ARG[@]}" \
  "${USE_TTA_ARG[@]}"

echo "[ART] Inference complete. Computing quality metrics..."

if [[ "$MODE" == "folder" ]]; then
  /usr/local/bin/python3 /home/user/scripts/compute_artifact_metrics.py \
    --folder "$ART_OUTPUT" \
    --output "$ART_OUTPUT/artifact_quality_metrics.yaml" \
    --format yaml \
    --pattern "*_artifact.tif"
else
  SLIDE_ID="$(basename "$OUTPUT_PATH" | sed 's/_artifact\.tif$//')"
  METRICS_OUTPUT="$(dirname "$ART_OUTPUT")/artifact_quality_metrics.yaml"

  /usr/local/bin/python3 /home/user/scripts/compute_artifact_metrics.py \
    --mask "$ART_OUTPUT" \
    --slide_id "$SLIDE_ID" \
    --output "$METRICS_OUTPUT" \
    --format yaml
fi

echo "[ART] Done!"