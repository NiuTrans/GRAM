from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import accelerate

prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user\'s instructions and answers the user\'s question better.
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
"""

query = "What is the Russian word for frog?"
response1 = "The Russian word for frog is \"лягушка\" (pronounced \"lyagushka\")."
response2 = "The Russian word for frog is \"жаба\" (pronounced as \"zhaba\"). This word can also be written in Cyrillic as жа́ба. If you're learning Russian, here's a sentence with the word: Меня зовут Иван, и я люблю лезечку на спину жабы, which translates to \"My name is Ivan, and I like sitting on the back of a frog.\" (Keep in mind that in real life, it is best not to disturb or harm frogs.)"

model_name_or_path = "/path/to/gram/model"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.padding_side = "left"
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")

messages = [
    [{"role": "user", "content": prompt.format(input=query, response_a=response1, response_b=response2)}],
    [{"role": "user", "content": prompt.format(input=query, response_a=response2, response_b=response1)}],
]

# target at response1, response2 respectively
target_choices_response1 = ["A", "B"]
target_choices_response1_token_ids = torch.tensor([tokenizer(item, add_special_tokens=False).input_ids for item in target_choices_response1], device=model.device)
target_choices_response2_token_ids = torch.flip(target_choices_response1_token_ids, dims=(0,))
target_choices_token_ids = torch.cat((target_choices_response1_token_ids, target_choices_response2_token_ids), dim=1)

prompt = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

with torch.no_grad():
    output = model(**inputs)
    logits = torch.gather(output.logits[..., -1, :], 1, target_choices_token_ids)
    p = torch.nn.Softmax(dim=0)(logits)
    score_response1, score_response2 = torch.mean(p, dim=1).tolist()

print({
    "query": query,
    "response1": response1,
    "response2": response2,
    "score_response1": score_response1,
    "score_response2": score_response2,
    "response1_is_better": score_response1 > score_response2,
})
