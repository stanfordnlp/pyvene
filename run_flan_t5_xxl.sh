for LAYER in 20 15 10 5 0
do
  for TASK_NAME in price_tagging_ub price_tagging_mid_diff
  do
    time python run_alignment.py \
    --model_path ./weights/flan-t5-xxl \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --lr 1e-3 \
    --seed 0 \
    --output_dir ./results_test/ \
    --epochs 3 \
    --do_align \
    --do_eval \
    --do_test \
    --n_training_examples 9000 \
    --n_eval_examples 1000 \
    --is_wandb --wandb_username dduan97 --bf16 \
    --task_name $TASK_NAME \
    --model_type t5 \
    --valid_steps 250 \
    --layer $LAYER \
    --token_start 0 \
    --token_end 1
  done
done
