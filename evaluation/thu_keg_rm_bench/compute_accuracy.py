import json
import argparse
import numpy as np
from typing import List, Dict, Any


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--res")
args = parser.parse_args()


res = []
with open(args.r, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        res.append(json.load(line))

assert len(res) % 9 == 0, "The result may be incomplete!"
res_with_score_matrix = []
for i in range(0, len(res), 9):
    group_res = res[i:i+9]
    score_matrix = [[None for j in range(3)] for i in range(3)]
    res_with_score_matrix_item = group_res[0]
    res_with_score_matrix_item["chosen"] = []
    res_with_score_matrix_item["rejected"] = []
    for _res in group_res:
        if not _res["chosen"] in res_with_score_matrix_item["chosen"]:
            res_with_score_matrix["chosen"].append(_res["chosen"])

        if not _res["rejected"] in res_with_score_matrix_item["rejected"]:
            res_with_score_matrix["rejected"].append(_res["rejected"])

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
        "hard_acc": hard_acc,
        "normal_acc": normal_acc,
        "easy_acc": easy_acc
    }

print(compute_accuracy(res_with_score_matrix))
