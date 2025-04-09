# pip install math_verify # reward function
# pip install -U trl
# GPU: 4090PLUS-48G*2
# GPU memory: 96G
# You can set `--reward_model` to use a reward model to provide rewards.
# 确保使用 torchrun 或 swift cli 能正确识别多卡环境
export WANDB_API_KEY=key
export CUDA_VISIBLE_DEVICES=0,1 # 如果需要显式指定
export NPROC_PER_NODE=2
swift rlhf \
    --rlhf_type grpo \
    --model /root/Qwen2.5-0.5B-Instruct \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_ebitda_predictor external_progressive_format \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset /root/ebitda_prediction_grpo_dataset/ebitda_prediction_grpo_dataset_v5.jsonl \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --eval_steps 1000 \
    --save_steps 100 \
    --save_total_limit 1 \
    --logging_steps 5 \
    --output_dir output_ds_bs8 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 0.9 \
    --log_completions true \
    --deepspeed zero2 \
    --report_to wandb





