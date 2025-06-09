import json
from typing import Dict, List

result_file_path = "/home/user/if/wang/models/saves/qwen3-14b/sft-rm-skywork-v0.2-ls0.05/reward-bench.res"

subset_results: Dict[str, List[bool]] = {}

with open(result_file_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        subset = item["subset"]
        correct = item["correct"]

        if subset not in subset_results:
            subset_results[subset] = []
        subset_results[subset].append(correct)

task_groups = {
    "chat": [
        "alpacaeval-easy", "alpacaeval-length", "alpacaeval-hard",
        "mt-bench-easy", "mt-bench-med"
    ],
    "chat-hard": [
        "mt-bench-hard", "llmbar-natural", "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst", "llmbar-adver-GPTOut", "llmbar-adver-manual"
    ],
    "safety": [
        "refusals-dangerous", "refusals-offensive",
        "xstest-should-refuse", "xstest-should-respond", "donotanswer"
    ],
    "reasoning": [
        "math-prm", "hep-cpp", "hep-go", "hep-java",
        "hep-js", "hep-python", "hep-rust"
    ]
}

group_accuracies: List[float] = []

for group_name, subsets in task_groups.items():
    subset_corrects = []
    for subset in subsets:
        subset_corrects += subset_results.get(subset, [])
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
