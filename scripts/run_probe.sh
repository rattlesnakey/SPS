set -e
MODEL=${1:-qwen3_4b}
CONFIG=${2:-configs/sps.yaml}
python analysis/probe_layers.py --config "$CONFIG" --model "$MODEL"
