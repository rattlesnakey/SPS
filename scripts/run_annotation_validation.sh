set -e
MODE=${1:-sample}
MODEL=${2:-qwen3_4b}
if [ "$MODE" = "sample" ]; then
  python analysis/annotation_validation.py sample --annotations "outputs/$MODEL/candidate_annotations.jsonl" --output "outputs/$MODEL/human_validation.jsonl"
else
  python analysis/annotation_validation.py aggregate --reviews "outputs/$MODEL/human_validation.jsonl" --output "outputs/$MODEL/human_validation_summary.json"
fi
