import json
import pandas as pd


file_path = [
    "./allenai/reward-bench/data/filtered-00000-of-00001.parquet",
    "./allenai/reward-bench/data/raw-00000-of-00001.parquet"
]

data = {}
for f in file_path:
    df = pd.read_parquet(f)
    _data = []
    for df_item in df.values:
        _data.append({k:v for k,v in zip(df.columns, df_item)})

    data[f] = _data

target_map = {
    "./allenai/reward-bench/data/filtered-00000-of-00001.parquet": "./filtered.json",b
    "./allenai/reward-bench/data/raw-00000-of-00001.parquet": "./raw.json",
}

for k,v in target_map.items():
    with open(v, 'w', encoding='utf-8') as f:
        json.dump(data[k], f, ensure_ascii=False, indent=2)
            
