set -e
CONFIG=${1:-configs/sps.yaml}
python analysis/calibrate_dedup_thresholds.py --config "$CONFIG"
