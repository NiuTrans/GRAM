import json
import copy
import pandas as pd


file_path = [
    "./THU-KEG/RM-Bench/total_dataset.json"
]

data = {}
for file_item in file_path:
    _data = []
    with open(file_item, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    for item in raw:
        for c_idx, chosen_item in enumerate(item["chosen"]):
            for r_idx, rejected_item in enumerate(item["rejected"]):
                _data_item = copy.deepcopy(item)
                _data_item["chosen"] = chosen_item
                _data_item["rejected"] = rejected_item
                _data_item["matrix_id"] = f"{c_idx}-{r_idx}"
                _data.append(_data_item)

    data[file_item] = _data

target_map = {
    "./THU-KEG/RM-Bench/total_dataset.json": "./total_dataset.json",
}

for k,v in target_map.items():
    with open(v, 'w', encoding='utf-8') as f:
        json.dump(data[k], f, ensure_ascii=False, indent=2)
            
