

# qwen3-4B pre-training-rm
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train examples/train_full/qwen3_pre_training_rm.yaml


# qwen3-30B-A3B pre-training-rm
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train examples/train_full/qwen3_pre_training_ft.yaml

wait

# qwen3-30B-A3B pre-training-rm
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train examples/train_full/qwen3_fine_tuning_rm.yaml