export WANDB_API_KEY=key
export CUDA_VISIBLE_DEVICES=0 # 如果需要显式指定
export NPROC_PER_NODE=1
swift rlhf \
    --rlhf_type grpo \
    --model /root/Qwen2.5-3B-Instruct \
    --ref_model /root/Qwen2.5-3B-Instruct \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_correctness external_integer_format external_strict_format external_soft_format external_xml_count \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset modelscope/gsm8k \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --eval_steps 1000 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir Qwen2.5-3B-gpro-math-output-full \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 0.9 \
    --log_completions true \
    --deepspeed zero2 \
    --report_to wandb