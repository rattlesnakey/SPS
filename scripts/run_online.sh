set -e
MODEL=${1:-qwen3_4b}
DATA=${2:-data/aime24.jsonl}
SAMPLES=${3:-4}
CONFIG=${4:-configs/sps.yaml}
python online/run_sps.py --config "$CONFIG" --model "$MODEL" --data "$DATA" --num_samples "$SAMPLES"
