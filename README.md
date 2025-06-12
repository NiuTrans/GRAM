
# A Generative Foundation Reward Model (GRAM)

This repository contains the code and released models for our paper [GRAM: A Generative Foundation Reward Model for Reward Generalization 📝](). We propose a more effective approach to reward model training by combining both labeled and unlabeled data. Our method introduces a generative reward model that first learns from a large corpus of unlabeled data and is then fine-tuned with supervised data. Please find all the released model checkpoints at [this link 🤗]().

<img src="./gram.png" width="1000px"></img>


## 🆕 Changelog
- [2025/6/15]
- [2025/6/1] We performed additional data cleaning, such as the removal of overly long or corrupted samples, to help GRAM achieve better performance. The processed dataset is available at this [link](https://huggingface.co/datasets/wangclnlp/GRAM-pre-training-566k).
- [2025/5/1] Our paper has been accepted by ICML 2025!


## 🔗 Quick Links
* [GRAM: A Generative Foundation Reward Model for Reward Generalization](#a-generative-foundation-reward-model-gram)

  * [Changelog](#changelog)
  * [Released Models](#released-models)
  * [Installation Guide](#installation)
  * [Preparing Models and Datasets](#preparing-models-and-datasets)
  * [Training Scripts](#training-scripts)
  * [Using GRAM in RLHF](#how-to-use-gram-in-rlhf)
  * [Citation](#citation)
  * [Acknowledgement](#acknowledgement)


---

## Released Models

Check out our GRAM series below. The models were first pre-trained on the dataset available [here](https://huggingface.co/datasets/wangclnlp/GRAM-pre-training-566k), and then fine-tuned on the dataset available [here](https://huggingface.co/datasets/wangclnlp/GRAM-fine-tuning-65k).

- We evaluate our reward model on the recently introduced [RM-Bench](https://github.com/THU-KEG/RM-Bench), a challenging benchmark for reward models, and present the results as follows:

| Model | Chat |	Code |	Math |	Safety |	Avg. | 
|:-:|:-:|:-:|:-:|:-:|:-:|
|nvidia/Llama-3.1-Nemotron-70B-Reward|70.7|57.4|64.3|90.3|70.7|
|Skywork/Skywork-Reward-Gemma-2-27B-v0.2|71.8|56.6|59.2|94.3|70.5|
|Skywork/Skywork-Reward-Llama-3.1-8B-v0.2|69.5|54.5|60.6|95.7|70.1|
|nvidia/Nemotron-340B-Reward|71.2|59.4|59.8|87.5|69.5|
|internlm/internlm2-20b-reward|63.1|56.7|66.8|86.5|68.3|
|[GRAM-Qwen3-1.7B-RewardBench](https://huggingface.co/wangclnlp/GRAM-Qwen3-1.7B-RewardModel)   |59.6|53.6|59.6|91.8|66.2|
|[GRAM-LLaMA3.2-3B-RewardBench](https://huggingface.co/wangclnlp/GRAM-LLaMA3.2-3B-RewardModel) |56.8|50.0|56.3|88.7|63.0|

- We also evaluate our reward model on the [JudgeBench](https://huggingface.co/datasets/ScalerLab/JudgeBench), a benchmark for evaluating LLM-as-a-Judge applications, and present the results as follows:

| Model | Chat |	Code |	Math |	Safety |	Avg. | 
|:-:|:-:|:-:|:-:|:-:|:-:|
|[GRAM-LLaMA3.2-3B-RewardBench](https://huggingface.co/wangclnlp/GRAM-LLaMA3.2-3B-RewardModel) |59.7|64.3|84.0|71.4|69.9|
|nvidia/Llama-3.1-Nemotron-70B-Reward|62.3|72.5|76.8|57.1|67.2|
|[GRAM-Qwen3-1.7B-RewardBench](https://huggingface.co/wangclnlp/GRAM-Qwen3-1.7B-RewardModel)   |60.4|65.3|78.6|57.1|65.4|
|Skywork/Skywork-Reward-Gemma-2-27B-v0.2|59.7|66.3|83.9|50.0|65.0|
|Skywork/Skywork-Reward-Llama-3.1-8B-v0.2|59.1|64.3|76.8|50.0|62.6|
|internlm/internlm2-20b-reward|62.3|69.4|66.1|50.0|62.0|


## Installation Guide

The code of this repo is modified from [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). If you encounter installation issues (e.g., related to PyTorch or CUDA), we recommend first checking the LLaMA-Factory [issues](https://github.com/hiyouga/LLaMA-Factory/issues) for potential solutions. If the problem persists, please feel free to submit an issue in this repository.

```bash
git clone --depth 1 https://gitee.com/wangclnlp/gram
cd gram
pip install -e ".[torch,metrics]" --no-build-isolation
```

## Preparing Models and Datasets

### Datasets

#### Pre-Training

Each item of the dataset for GRAM pre-training should include at least two keys:

- `instruction`: any prompt in following template:
  ```text
  [User Question]
  {your prompt here}
  ```
- `input`: the input for above prompt, can be empty if there is not.
- `output`: two responses in following template:
  ```text
  [The Start of Assistant A's Answer]
  {answer of assistant A}
  [The End of Assistant A's Answer]

  [The Start of Assistant B's Answer]
  {answer of assistant B}
  [The End of Assistant B's Answer]
  ```

An example in json format:

```json
[
  {
    "instruction": "[User Question]\nCan dogs get covid?\n\n",
    "input": "",
    "output": "[The Start of Assistant A's Answer]\nYes, indeed. ... [The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\nMany of the symptoms are similar, including fever, coughing, loss of smell, etc. ...\n[The End of Assistant B's Answer]"
  },
  ...
]
```


#### Fine-Tuning

Each item of the dataset for GRAM fine-tuning should include at least two keys:

- `instruction`: any prompt with corresponding two responses in following template:
  ```text
  Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better.
  Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.
  Please directly output your final verdict by strictly following this format: "A" if assistant A is better, "B" if assistant B is better.

  [User Question]
  {your prompt here}

  [The Start of Assistant A's Answer]
  {answer of assistant A}
  [The End of Assistant A's Answer]

  [The Start of Assistant B's Answer]
  {answer of assistant B}
  [The End of Assistant B's Answer]

  #Preferred:
  ```
- `input`: leave it empty.
- `output`: the correct option, "A" or "B".

An example in json format:

```json
[
  {
    "instruction": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. ... [User Question]... [The Start of Assistant A's Answer] ... [The Start of Assistant B's Answer] ...",
    "input": "",
    "output": "B"
  },
  ...
]
```


#### Note
> After pre-processing the datasets as above, DO NOT forget to register the dataset in [`data/dataset_info.json`](data/dataset_info.json).
> Example:
> ```json
>  {
>    ...,
>    "dataset_name": {
>      "file_name": "/path/to/your/dataset"
>    },
>    ...,
>  },
> ```


## Training Scripts

### Pre-Training

```bash
llamafactory-cli train examples/train_full/qwen3_pre_training_rm.yaml
```

### Fine-Tuning
```bash
llamafactory-cli train examples/train_full/qwen3_fine_tuning_rm.yaml
```
**We recommend directly fine-tuning our open-source reward model using your own task-specific (or domain-specific) preference data.** This approach may reduce reliance on large-scale task-specific data, which allows you to obtain a strong reward model for your target task with only a small amount of preference annotations.


### Evaluation

The evaluation scripts are in the subdirectory `evaluation/`:
```bash
cd evaluation/
ckpt_path=/path/to/your/model
```

- Evaluation with Rewardbench

  ```bash
  python gram_eval.py -i allenai_reward_bench/filtered.json -m $ckpt_path -o $ckpt_path/reward-bench.res
  echo -e "RewardBench Evaluation Summary:\n"
  python get_reward_bench_score.py $ckpt_path/reward-bench.res
  ```

- Evaluation with Judgebench
  ```bash
  python gram_eval.py -i scalerlab_judgebench/gpt.json -m $ckpt_path -o $ckpt_path/judge-bench.res
  echo -e "JudgeBench Evaluation Summary:\n"
  python get_judgebench_score.py $ckpt_path/judge-bench.res
  ```

- Evaluation with RM-Bench
  ```bash
  python gram_eval.py -i thu_keg_rm_bench/total_dataset.json -m $ckpt_path -o $ckpt_path/reward-bench.res
  echo -e "RM-bench Evaluation Summary:\n"
  python thu_keg_rm_bench/compute_accuracy.py $ckpt_path/reward-bench.res
  ```

## Using GRAM in RLHF

### Computing Rewards of a pair of Samples

```python
def compute_pair_rewards(response_a, response_b):
    # compute rewards for response_a and response_b as in the demo `evaluation/gram_demo.py`
    ...
    return reward_response_a, reward_response_b
```

### PPO

When applying GRAM to PPO training, we first generate a **reference response** and then compute a reward score using GRAM, which quantifies **how much better the sampled response is compared to the reference**. This score serves as the reward signal during PPO training. The basic idea, using the difference between the sampled and reference responses as the reward, has been shown effective in prior baseline methods. Additionally, inspired by [*ReMax*](https://arxiv.org/abs/2310.10505), we can use **greedy search** to construct the reference response. The detailed procedure is described below:

```python
def ppo():
    # Init dataset and model
    ...
    ref_model, policy_model, reward_model, value_model = ...
    # Sample from policy model: generate with greedy search first, then sample as normal with top-p/top-k
    response_greedy_search = greedy_search(policy_model, query)
    response_normal = generate(policy_model, query, top_p=..., top_k=...)
    _, reward_response_normal = compute_pair_rewards(response_greedy_search, response_normal)
    # Compute logits from ref_model, values from value_model and update with PPO loss
    ...
```

### List-wise Response Ranking
A common use case for list-wise response ranking is best-of-n sampling, where the goal is to select the single best response from a list. This can be accomplished using GRAM with a linear search approach, as illustrated below. To support parallel computation and improve efficiency, we also incorporate optimization strategies such as divide-and-conquer.

```python
def list_wise_response_ranking():
    # Init dataset and model
    ...
    # Generate from model
    responses = [response0, response1, responses2, ...]
    # Compute rewards and choose one with highest score
    best_response = response0
    for response in responses[1:]:
        score_a, score_b = compute_pair_rewards(best_response, response)
        if score_a < score_b:
           best_response = response

    return best_response
```

## Citation
```bash
our paper bib
```


## Acknowledgement
We commence by utilizing the exceptional codebase provided by [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 🌹🌹🌹.

We would like to thank [Hang Zhou](https://github.com/stceum) for his help in open-sourcing the GRAM model series.

We thank the contributions of the following papers:
```bash
[1] Lambert, Nathan, et al. "Rewardbench: Evaluating reward models for language modeling." arXiv preprint arXiv:2403.13787 (2024).
[2] Liu, Yantao, et al. "RM-bench: Benchmarking reward models of language models with subtlety and style." arXiv preprint arXiv:2410.16184 (2024).
[3] Tan, Sijun, et al. "Judgebench: A benchmark for evaluating llm-based judges." arXiv preprint arXiv:2410.12784 (2024).
```
