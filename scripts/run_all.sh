set -e
MODEL=${1:-qwen3_4b}
CONFIG=${2:-configs/sps.yaml}
SAMPLES=${3:-4}
bash scripts/run_offline.sh "$MODEL" "$CONFIG"
bash scripts/evaluate_all.sh "$MODEL" "$CONFIG" "$SAMPLES"
