set -e
MODEL=${1:-qwen3_4b}
CONFIG=${2:-configs/sps.yaml}
SAMPLES=${3:-4}
for DATA in data/math500.jsonl data/Minerva-Math-272.jsonl data/olym.jsonl data/aime24.jsonl data/aime25.jsonl data/hmmt25.jsonl data/gpqa_diamond.jsonl data/livecodebench_release_v5_2024_08_01.jsonl data/strategyqa.jsonl
do
  python online/run_sps.py --config "$CONFIG" --model "$MODEL" --data "$DATA" --num_samples "$SAMPLES"
  STEM=$(basename "$DATA" .jsonl)
  python online/evaluate.py --predictions "outputs/$MODEL/${STEM}_sps.jsonl" --output "outputs/$MODEL/${STEM}_metrics.json"
done
