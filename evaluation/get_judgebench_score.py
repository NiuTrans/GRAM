import json
from typing import Dict, List

result_file_path = "/home/user/if/wang/models/saves/qwen3-14b/sft-rm-skywork-v0.2/checkpoint-1500/judge-bench.res"

subset_results: Dict[str, List[bool]] = {}

group_accuracies: List[float] = []

with open(result_file_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        subset = item["source"]
        correct = item["correct"]

        if subset not in subset_results:
            subset_results[subset] = []
        subset_results[subset].append(correct)

        group_accuracies.append(item["correct"])

task_groups = {
    "Knowledge": "mmlu-pro",
    "Reasoning": "livebench-reasoning",
    "Math": "livebench-math",
    "Coding": "livecodebench"
}

for group_name, subsets in task_groups.items():
    subset_corrects = []

    for subset_result_item in subset_results.keys():
        if subsets in subset_result_item:
            subset_corrects += subset_results.get(subset_result_item, [])

    if subset_corrects:
        accuracy = sum(subset_corrects) / len(subset_corrects)
        print(f"{group_name}: {accuracy:.4f}")
        group_accuracies.append(accuracy)
    else:
        print(f"{group_name}: No data available.")

if group_accuracies:
    overall_accuracy = sum(group_accuracies) / len(group_accuracies)
    print(f"Overall average accuracy: {overall_accuracy:.4f}")
else:
    print("No group accuracy to report.")
