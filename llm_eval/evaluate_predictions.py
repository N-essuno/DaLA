import os
import ssl
from collections import defaultdict

from llm_gen_eval import evaluate_dataset

# Bypass SSL verification for NLTK/HuggingFace downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

predictions_dir = "predictions/"

metrics_dict = defaultdict()

# Pre-load metrics once to avoid repeated loading and network issues
print("Loading evaluation metrics...")
import evaluate
from bert_score import BERTScorer

meteor_metric = evaluate.load("meteor")
# bertscore_hf_metric = evaluate.load("bertscore")
bertscore_scorer = BERTScorer(lang="da")
print("Metrics loaded successfully.\n")

# Evaluate the predictions against the gold answers
for filename in os.listdir(predictions_dir):
    model_name = filename.replace(".csv", "")
    print(f"Evaluating {model_name}...")
    metrics = evaluate_dataset(
        path=predictions_dir+filename,
        gold_path="../la_output/gen_v2/dala_da_gen_v2_val_formatted_full.csv",
        pred_col="Answer",
        meteor_metric=meteor_metric,
        # bertscore_hf_metric=bertscore_hf_metric,
        bertscore_scorer=bertscore_scorer
    )
    metrics_dict[model_name] = metrics


for model_name, metrics in metrics_dict.items():
    print(f"\n=== {model_name} ===")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

