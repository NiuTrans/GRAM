
ckpt_path="../saves/qwen3-4b/sft-rm-skywork-v0.2/checkpoint-1028"

base_model=$ckpt_path

echo "#######################Evaluating ${ckpt_path}#######################"

# evaluation with rewardbench
python gram_eval.py -i allenai_reward_bench/filtered.json -bm $base_model -m $ckpt_path -o $ckpt_path/reward-bench.res
echo -e "RewardBench Evaluation Summary:\n"
python get_reward_bench_score.py $ckpt_path/reward-bench.res

# evaluation with judgebench
python gram_eval.py -i scalerlab_judgebench/gpt.json -bm $base_model -m $ckpt_path -o $ckpt_path/judge-bench.res
echo -e "JudgeBench Evaluation Summary:\n"
python get_judgebench_score.py $ckpt_path/judge-bench.res

# evaluation with RM-bench
python gram_eval.py -i thu_keg_rm_bench/total_dataset.json -bm $base_model -m $ckpt_path -o $ckpt_path/reward-bench.res
echo -e "RM-bench Evaluation Summary:\n"
python thu_keg_rm_bench/compute_accuracy.py $ckpt_path/reward-bench.res