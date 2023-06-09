for LAYER in 0 5 10 15 20
do
  for TOKEN_START in 24 25 26 27
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
    --n_training_examples 5000 \
    --n_eval_examples 1000 \
    --is_wandb --wandb_username dduan97 --bf16 \
    --task_name price_tagging_lb \
    --model_type t5 \
    --valid_steps 250 \
    --layer $LAYER \
    --token_start $TOKEN_START \
    --token_end $((TOKEN_START+1))
  done
done
