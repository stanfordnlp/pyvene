#!/bin/bash
# Read a string with spaces using for loop
# This on a single A100/A6000 takes 24 hours to run for a single seed!
# first argument is like "79:80" -> aligning on 79th token.
# second argument is like "3.50;8.50;0.00;9.99"
for seed in 77 # 42 66 77
do
    for layer in 0 5 10 15 20 25 30
    do
        for task in pricing_tag_lb pricing_tag_lub pricing_tag_mid_diff pricing_tag_bracket pricing_tag_fixed
        do
            python run_llm_alignment.py \
            --train_batch_size 8 \
            --eval_batch_size 8 \
            --gradient_accumulation_steps 8 \
            --lr 1e-3 \
            --seed $seed \
            --output_dir ./results_alpaca-7b/ \
            --epochs 3 \
            --do_align \
            --aligning_layer_n $layer \
            --aligning_tokens $1 \
            --n_training_examples 20000 \
            --n_eval_examples 200 \
            --task_name $task \
            --bf16 \
            --is_wandb
        done
    done
done
