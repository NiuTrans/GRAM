# coding=utf-8
import openai
import sys
import json
import os
import concurrent.futures
from tqdm import tqdm

# ==== settings ====
base_port = 8001   # Starting port number
num_shards = 8     # Number of shards (also corresponds to the number of ports)
max_workers = 16   # Maximum number of concurrent workers per port
timeout_per_request = 40  # Timeout per request (in seconds)
# ==================

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

src_path = sys.argv[1]
trg_path = sys.argv[2]

def load_convert_data(path):
    with open(path) as src:
        src_list = json.load(src)
        output_list = []
        for i, item in enumerate(src_list):
            output_list.append({
                "instruction": item["instruction"].replace("User: ", "").replace("\nAssistant: ", ""),
                "input": item["input"],
                "output": item["output"],
                "id": i,
                "num": 0
            })
    return output_list

def load_existing_ids(path):
    existing_ids = set()
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing_ids.add(obj["id"])
                except:
                    continue
    return existing_ids

def create_client(port):
    return openai.OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{port}/v1",
        timeout=timeout_per_request
    )

def vllm_request(item, client):
    prompt = item["instruction"]
    content = prompt + "\n" + item["input"]
    messages = [{"role": "user", "content": content}]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=client.models.list().data[0].id,
            seed=42,
            logprobs=1,
            max_tokens=2048,
            temperature=0.75
        )

        all_logprobs = chat_completion.choices[0].logprobs.content
        logprobs = [logprob.logprob for logprob in all_logprobs]

        return {
            "id": item["id"],
            "num": item["num"],
            "instruction": prompt,
            "output": chat_completion.choices[0].message.content,
            "input": item["input"],
            "logprob": sum(logprobs) / len(logprobs) if logprobs else -9999
        }
    except Exception as e:
        print(f"[ERROR] ID {item['id']} failed: {e}")
        return {
            "id": item["id"],
            "num": item["num"],
            "instruction": item["instruction"],
            "output": "",
            "input": item["input"],
            "logprob": -9999
        }

def process_shard(shard_data, port, out_file_path):
    client = create_client(port)
    with open(out_file_path, 'a+', encoding='utf-8') as out_file:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {
                executor.submit(vllm_request, item, client): item for item in shard_data
            }

            for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(shard_data), desc=f"Port {port}"):
                item = future_to_item[future]
                result = future.result()
                out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_file.flush()

def main():
    src_list = load_convert_data(src_path)
    existing_ids = load_existing_ids(trg_path)
    to_process = [item for item in src_list if item["id"] not in existing_ids]
    print(f"[INFO] Loaded {len(src_list)} samples, {len(to_process)} to process.")

    shards = [[] for _ in range(num_shards)]
    for idx, item in enumerate(to_process):
        shards[idx % num_shards].append(item)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_shards) as executor:
        futures = []
        for shard_id, shard_data in enumerate(shards):
            port = base_port + shard_id
            futures.append(
                executor.submit(process_shard, shard_data, port, trg_path)
            )
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    main()
