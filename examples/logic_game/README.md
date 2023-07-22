## Word Logic Game

#### Alignment Training
Once you obtain the model that performs well on this task, you can use commands like the following one to find alignment in a specific location in the neural network,
```bash
CUDA_VISIBLE_DEVICES=1 python run_tutorial.py \
    --model_name_or_path ./tutorial_results/checkpoint-200 \
    --n_training_examples 22000 \
    --do_align \
    --bf16 \
    --output_dir ./alignment_results/ \
    --epochs 3 \
    --batch_size 128 \
    --log_step 1 \
    --valid_steps 20 \
    --lr 1e-3 \
    --alignment_layer 0 \
    --alignment_variable op1 \
    --token_position_strategy 8_9 \
    --task_name 07065a \
    --is_wandb
```