#!/usr/bin/env python
# coding: utf-8
from utils.train_utils import *

if __name__ == '__main__':
    is_notebook = False
    try:
        cmd = argparse.ArgumentParser('The testing components of')
        cmd.add_argument('--train_batch_size', default=128, type=int, help='training batch size')
        cmd.add_argument('--eval_batch_size', default=128, type=int, help='training batch size')
        cmd.add_argument('--lr', default=0.01, type=float, help='learning rate')
        cmd.add_argument(
            '--encoder_config_path', 
            type=str, help='path to the encoder config'
        )
        cmd.add_argument(
            '--decoder_config_path', 
            type=str, help='path to the decoder config'
        )
        cmd.add_argument('--max_seq_len', default=512, type=int)
        cmd.add_argument('--seed', default=42, type=int)
        cmd.add_argument('--gradient_accumulation_steps', default=1, type=int)
        cmd.add_argument('--output_dir', required=True, type=str, help='save dir')
        cmd.add_argument('--local_rank', default=-1, type=int, help='multi gpu training')
        cmd.add_argument('--epochs', default=10, type=int, help='training epochs')
        cmd.add_argument('--model_path', type=str, required=False, default=None)
        cmd.add_argument('--warm_up', type=float, default=0.1)
        cmd.add_argument('--is_wandb', default=False, action='store_true')
        cmd.add_argument('--bf16', default=False, action='store_true')
        cmd.add_argument('--log_step', default=10, type=int)
        cmd.add_argument('--valid_steps', default=500, type=int)
        cmd.add_argument('--early_stopping', default=5, type=int)
        cmd.add_argument('--device', default="cuda", type=str, help='')
        cmd.add_argument('--do_align', default=False, action='store_true')
        cmd.add_argument('--do_eval', default=False, action='store_true')
        cmd.add_argument('--do_test', default=False, action='store_true')
        
        cmd.add_argument('--aligning_layer_n', default=0, type=int)
        cmd.add_argument('--aligning_tokens', default="", type=str, help='[START_TOKEN];[END_TOKEN]')
        cmd.add_argument('--n_training_examples', default=10000, type=int)
        cmd.add_argument('--n_eval_examples', default=1000, type=int)
        cmd.add_argument('--task_name', default="cost_no_type", type=str, help='')
        cmd.add_argument('--task_config', default="", type=str, help='')
        cmd.add_argument('--aligning_var_n', default=0, type=int)
        cmd.add_argument('--aligning_basis_n_per_variable', default=0, type=int)
        cmd.add_argument('--unit_test_mode', default=False, action='store_true')
        
        args = cmd.parse_args(sys.argv[1:])
    except:
        assert False
        is_notebook = True
        parser = argparse.ArgumentParser()
        args = parser.parse_args([])
        args.train_batch_size = 8
        args.eval_batch_size = 8
        args.gradient_accumulation_steps = 16
        args.lr = 1e-4
        args.seed = 42
        args.output_dir = "./results_notebook/"
        args.epochs = 1
        args.warm_up = 0.1
        args.is_wandb = False
        args.log_step = 10
        args.valid_steps = 100 # -1 not do training eval!
        args.early_stopping = 999 # large == never early stop!
        args.device = "cuda"
        args.do_align = True
        args.do_eval = True
        args.n_gpu = 1
        
        # alignment search setting
        args.aligning_layer_n = 16
        args.aligning_tokens = "79;80"
        
        args.aligning_var_n = 2
        args.task_config = "3.50;8.50;0.00;9.99"
        args.n_training_examples = 1000
        args.n_eval_examples = 200
        args.task_name = "cost_no_type"
        
        args.unit_test_mode = False
        
        print("Using in a notebook env.")

###################
# data loaders
###################
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="../alpaca_7b/",
    cache_dir=CACHE_DIR
)
prealign_dataloader, train_dataloader, eval_dataloader = prepare_dataloader(args, tokenizer)

###################
# model object loading
###################
alignment_config = {
    'layer': args.aligning_layer_n,
    "token_range" : [
        int(args.aligning_tokens.split(";")[0]), 
        int(args.aligning_tokens.split(";")[1]), 
    ]
}
logger.info(f"alignment_config = {alignment_config}")

if args.unit_test_mode:
    logger.info("Loading Dummy Model for Testing ...")
    # Testing code.
    model = AlignableLlamaForCausalLM.from_pretrained(
        "../alpaca_test/",
        alignment_config=alignment_config,
        torch_dtype=torch.bfloat16 if args.bf16 else None
    )
else:
    logger.info("Loading Alpaca 7B, Takes 2 Mins ...")
    model = AlignableLlamaForCausalLM.from_pretrained(
        "../alpaca_7b/",
        alignment_config=alignment_config,
        torch_dtype=torch.bfloat16 if args.bf16 else None
    )

# set off the gradients among all other layers.
for name, param in model.named_parameters():
    if "rotate_layer" not in name and "intervention_boundaries" not in name:
        param.requires_grad = False
    else:
        logger.info(f"Requiring gradients on layer: {name}")
        
t_total = int(len(train_dataloader) * args.epochs)
warm_up_steps = args.warm_up * t_total
optimizer = torch.optim.Adam(
    [{'params': model.model.rotate_layer.parameters()},
    {'params': model.model.intervention_boundaries, 'lr': 1e-2}],
    lr=args.lr
)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warm_up_steps,
    num_training_steps=t_total
)
    
device = "cuda"
model.to(device)

###################
# trainer loading
###################
run_name = f"alpaca-7B.task.{args.task_name}.config.{args.task_config}."\
           f"seed.{args.seed}.intl.{args.aligning_layer_n}.intr.{alignment_config['token_range'][0]}."\
           f"{alignment_config['token_range'][1]}"

is_master = True
if not os.path.exists(args.output_dir) and is_master:
    os.mkdir(args.output_dir)
os.environ["WANDB_PROJECT"] = f"ToM-DAS"
output_dir = os.path.join(args.output_dir, run_name)
if not os.path.exists(output_dir) and is_master:
    os.mkdir(output_dir)

aligner = AlpacaAligner(
    model,
    logger=logger,
    args=args,
    is_master=is_master,
    n_gpu=torch.cuda.device_count(),
    model_name=run_name,
    device=device
)

# Prealign Eval is a must
aligner.prealign_eval(prealign_dataloader, output_dir)

FAIL()

# Train
if args.do_align:
    aligner.train(
        train_dataloader, eval_dataloader,
        optimizer, scheduler, 
        log_step=args.log_step, valid_steps=args.valid_steps,
        output_dir=output_dir, epochs=args.epochs, 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

