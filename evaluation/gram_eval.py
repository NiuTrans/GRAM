import os
import json
import torch
import argparse
import accelerate
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForCausalLM


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input")
parser.add_argument("-m", "--model")
parser.add_argument("-o", "--output")
parser.add_argument("-b", "--batch-size", default=1)
args = parser.parse_args()

if os.path.exists(args.output):
    os.remove(args.output)

system_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user\'s instructions and answers the user\'s question better.
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.
Please directly output your final verdict by strictly following this format: "A" if assistant A is better, "B" if assistant B is better.

[User Question]
{input}

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]

#Preferred: """

model_name_or_path = args.model
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.padding_side = "left"
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")

# target at chosen, rejected respectively
target_choices_response1 = ["A", "B"]
target_choices_response1_token_ids = torch.tensor([tokenizer(item, add_special_tokens=False).input_ids for item in target_choices_response1], device=model.device)
target_choices_response2_token_ids = torch.flip(target_choices_response1_token_ids, dims=(0,))
target_choices_token_ids = torch.cat((target_choices_response1_token_ids, target_choices_response2_token_ids), dim=1)

with open(args.input, 'r', encoding='utf-8') as f:
    input_data = json.load(f)

for idx in trange(0, len(input_data), args.batch_size):
    res = []
    batch_data = input_data[idx:idx+args.batch_size]
    messages = []
    for item in batch_data:
        messages += [
            [{"role": "user", "content": system_prompt.format(input=item["prompt"], response_a=item["chosen"], response_b=item["rejected"])}],
            [{"role": "user", "content": system_prompt.format(input=item["prompt"], response_a=item["rejected"], response_b=item["chosen"])}],
        ]

    prompt = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False) for message in messages]

    # enable_thinking=False
    # tokenizer.decode(output.logits.max(-1)[-1][0])

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        output = model(**inputs)
        logits = torch.gather(output.logits[..., -1, :], 1, target_choices_token_ids.repeat(args.batch_size, 1))
        p = torch.nn.Softmax(dim=1)(logits).view(args.batch_size, -1, 2)
        scores = torch.mean(p, dim=1).tolist()

    # import pdb; pdb.set_trace()

    for data_item, res_item in zip(batch_data, scores):
        data_item["score_chosen"] = res_item[0]
        data_item["score_rejected"] = res_item[1]
        data_item["correct"] = res_item[0] > res_item[1]
        res.append(data_item)

    if os.path.exists(args.output):
        mode = 'a'
    else:
        mode = 'w'

    with open(args.output, mode, encoding='utf-8') as f:
        for item in res:
            f.write(json.dumps(item, ensure_ascii=False)+'\n')
