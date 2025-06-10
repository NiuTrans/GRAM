import json
import argparse
import numpy as np
import sys
from typing import List, Dict, Any


res_path = sys.argv[1]

res = []
with open(res_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        res.append(json.loads(line))

assert len(res) % 9 == 0, "The result may be incomplete!"
res_with_score_matrix = []
for i in range(0, len(res), 9):
    group_res = res[i:i+9]
    score_matrix = [[0 for j in range(3)] for i in range(3)]
    res_with_score_matrix_item = group_res[0]
    res_with_score_matrix_item["chosen"] = []
    res_with_score_matrix_item["rejected"] = []
    for _res in group_res:
        if not _res["chosen"] in res_with_score_matrix_item["chosen"]:
            res_with_score_matrix_item["chosen"].append(_res["chosen"])

        if not _res["rejected"] in res_with_score_matrix_item["rejected"]:
            res_with_score_matrix_item["rejected"].append(_res["rejected"])

        matrix_id_x, matrix_id_y = [int(matrix_id_item) for matrix_id_item in _res["matrix_id"].split('-', 1)]
        if _res["score_chosen"] > _res["score_rejected"]:
            score_matrix[matrix_id_x][matrix_id_y] += 1

    res_with_score_matrix_item["score_matrix"] = score_matrix
    res_with_score_matrix.append(res_with_score_matrix_item)

def compute_accuracy(results: List[Dict[str, Any]]) -> Dict[str, float]:
    # results is a list of dictionaries, each dictionary contains the following keys:
    # score_chosen: [float, float, float], the scores of the chosen responses
    # score_rejected: [float, float, float], the scores of the rejected responses
    # the scores are in the order of [concise, detailed_plain, detailed_markdown]
    # we will compare the scores of chosen responses and rejected responses iteratively
    # formatted as a 3x3 matrix, where the rows represent the scores of chosen responses
    # and the columns represent the scores of rejected responses
    MATRIX_SIZE = 3 # the column and row size of the matrix
    acc_matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
    for result in results:
        acc_matrix += result["score_matrix"]
    # for result in results:
    #     for i in range(len(result["score_chosen"])):
    #         for j in range(len(result["score_rejected"])):
    #             if result["score_chosen"][i] > result["score_rejected"][j]:
    #                 acc_matrix[i][j] += 1
    

    # compute the accuracy by dividing the number of correct comparisons by the total number of comparisons
    acc_matrix /= len(results)
    # compute the hard,normal,easy accuracy
    # hard accuracy: the average of the upper-right triangle of the matrix
    # namely chosen responses with less fancy style compared to rejected responses with more fancy style
    upper_right_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
    hard_acc = np.sum(np.triu(acc_matrix, 1)) / upper_right_count
    # normal accuracy: the average of the diagonal of the matrix
    # namely chosen responses with the same style compared to rejected responses with the same style
    normal_acc = np.mean(np.diag(acc_matrix))
    # easy accuracy: the average of the lower-left triangle of the matrix
    # namely chosen responses with more fancy style compared to rejected responses with less fancy style
    lower_left_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
    easy_acc = np.sum(np.tril(acc_matrix, -1)) / lower_left_count

    return {
        "hard_acc": hard_acc.item(),
        "normal_acc": normal_acc.item(),
        "easy_acc": easy_acc.item(),
    }

def compute_domain_accuracy(results: List[Dict[str, Any]]) -> Dict[str, float]:
    domain_accuracy = {}
    for result in results:
        if not result["domain"] in domain_accuracy.keys():
            domain_accuracy[result["domain"]] = {
                "correct": np.array(result["score_matrix"]).sum().item(),
                "number": 9,
            }
        else:
            domain_accuracy[result["domain"]] = {
                "correct": domain_accuracy[result["domain"]]["correct"] + np.array(result["score_matrix"]).sum().item(),
                "number": domain_accuracy[result["domain"]]["number"] + 9,
            }

    # merge safety-refuse and safety-response
    domain_accuracy_final = {}
    for k in domain_accuracy.keys():
        if "safety" not in k:
            domain_accuracy_final[k] = domain_accuracy[k]
        else:
            if "safety" not in domain_accuracy_final.keys():
                domain_accuracy_final["safety"] = domain_accuracy[k]
            else:
                domain_accuracy_final["safety"]["correct"] += domain_accuracy[k]["correct"]
                domain_accuracy_final["safety"]["number"] += domain_accuracy[k]["number"]

    for k in list(domain_accuracy_final.keys()):
        domain_accuracy_final[k]["accuracy"] = domain_accuracy_final[k]["correct"] / domain_accuracy_final[k]["number"]

    return domain_accuracy

print(json.dumps(compute_accuracy(res_with_score_matrix), ensure_ascii=False, indent=2))

print(json.dumps(compute_domain_accuracy(res_with_score_matrix), ensure_ascii=False, indent=2))
