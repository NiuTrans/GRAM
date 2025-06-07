import json
import pandas as pd


file_path = [
    "./ScalerLab/JudgeBench/data/claude-00000-of-00001.jsonl",
    "./ScalerLab/JudgeBench/data/gpt-00000-of-00001.jsonl"
]

data = {}
for file_item in file_path:
    _data = []
    with open(file_item, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            _item = json.loads(line)
            _item["prompt"] = _item.pop("question")
            label = _item.pop("label")
            chosen_res, rejected_res = ("response_A", "response_B") if label == 'A>B' else \
                ("response_B", "response_A")
            _item["chosen"] = _item.pop(chosen_res)
            _item["rejected"] = _item.pop(rejected_res)
            _data.append(_item)

    data[file_item] = _data

target_map = {
    "./ScalerLab/JudgeBench/data/claude-00000-of-00001.jsonl": "./claude.json",
    "./ScalerLab/JudgeBench/data/gpt-00000-of-00001.jsonl": "./gpt.json",
}

for k,v in target_map.items():
    with open(v, 'w', encoding='utf-8') as f:
        json.dump(data[k], f, ensure_ascii=False, indent=2)
            
