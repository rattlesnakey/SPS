set -e
MODEL=${1:-qwen3_4b}
CONFIG=${2:-configs/sps.yaml}
if [ ! -f data/dedup_thresholds.json ]; then
  python analysis/calibrate_dedup_thresholds.py --config "$CONFIG"
fi
python offline/prepare_calibration.py --config "$CONFIG" --model "$MODEL"
python offline/generate_and_annotate.py --config "$CONFIG" --model "$MODEL"
python offline/construct_bank.py --config "$CONFIG" --model "$MODEL"
