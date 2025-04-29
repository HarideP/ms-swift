export WANDB_API_KEY=key
export CUDA_VISIBLE_DEVICES=0 # 如果需要显式指定
export NPROC_PER_NODE=1
swift rlhf \
    --rlhf_type grpo \
    --model /home/Qwen2.5-3B-Instruct \
    --ref_model /home/Qwen2.5-3B-Instruct \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs gsm8k_correctness gsm8k_is_int gsm8k_strict_format gsm8k_soft_format gsm8k_xml_count \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset /home/ms-swift/gsm8k_swift_grpo/train.jsonl \
    --val_dataset /home/ms-swift/gsm8k_swift_grpo/test.jsonl \
    --dataset_shuffle false \
    --max_length 512 \
    --max_completion_length 512 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 1 \
    --eval_steps 1000 \
    --save_steps 250 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --output_dir Qwen2.5-3B-gpro-math-output-full \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 0.9 \
    --log_completions true \
    --report_to wandb