from utils import *

import argparse

if __name__ == '__main__':
    is_notebook = False
    try:
        cmd = argparse.ArgumentParser('The testing components of')
        cmd.add_argument('--batch_size', default=128, type=int, help='training batch size')
        cmd.add_argument('--lr', default=0.01, type=float, help='learning rate')
        cmd.add_argument('--model_name_or_path', type=str, help='path to the model')
        cmd.add_argument('--max_seq_len', default=512, type=int)
        cmd.add_argument('--seed', default=42, type=int)
        cmd.add_argument('--gradient_accumulation_steps', default=1, type=int)
        cmd.add_argument('--output_dir', required=True, type=str, help='save dir')
        cmd.add_argument('--local_rank', default=-1, type=int, help='multi gpu training')
        cmd.add_argument('--epochs', default=10, type=int, help='training epochs')
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
        cmd.add_argument('--n_training_examples', default=22000, type=int)
        cmd.add_argument('--task_name', default="07065a", type=str, help='')
        cmd.add_argument('--alignment_variable', default="op1", type=str, help='')
        cmd.add_argument('--wandb_project', default="Boundless-DAS-word-logic", type=str, help='')
        cmd.add_argument('--alignment_layer', default=6, type=int)
        cmd.add_argument('--token_position_strategy', default="8_9", type=str, help='')
        cmd.add_argument('--preload_dataset', default=False, action='store_true')
        
        args = cmd.parse_args(sys.argv[1:])
    except:
        assert False

set_seed(args.seed)

programs = json.load(open("seed_programs.json"))
program = (args.task_name, programs[args.task_name])
target_word_beam = max(program[1][0][2][:2])

###################
# data loaders
###################
logger.info(f"Loading data for program={args.task_name}")
token_position_strategy = [int(t) for t in args.token_position_strategy.split("_")]
if args.preload_dataset:
    logger.info(f"Preloading the created dataset for training to save time...")
    preload_cache_dir = "./preload_datasets/"
    train_json = os.path.join(preload_cache_dir, f"train_{program[0]}_{args.alignment_variable}.json")
    validation_json = os.path.join(preload_cache_dir, f"validation_{program[0]}_{args.alignment_variable}.json")
    test_json = os.path.join(preload_cache_dir, f"test_{program[0]}_{args.alignment_variable}.json")
    train_cdataset = load_dataset('json', data_files=train_json)
    train_cdataset = make_supervised_counterfactual_data_module_single_preload(
        program,
        args.alignment_variable,
        args.n_training_examples,
        max(program[1][0][2][:2]), 
        token_position_strategy, # this will overwrite the previous arg!
        tokenizer=AutoTokenizer.from_pretrained("gpt2"),
        preload_dataset=train_cdataset
    )["train"]
    validation_cdataset = load_dataset('json', data_files=validation_json)
    validation_cdataset = make_supervised_counterfactual_data_module_single_preload(
        program,
        args.alignment_variable,
        args.n_training_examples,
        max(program[1][0][2][:2]), 
        token_position_strategy, # this will overwrite the previous arg!
        tokenizer=AutoTokenizer.from_pretrained("gpt2"),
        preload_dataset=validation_cdataset
    )["train"]
    test_cdataset = load_dataset('json', data_files=test_json)
    test_cdataset = make_supervised_counterfactual_data_module_single_preload(
        program,
        args.alignment_variable,
        args.n_training_examples,
        max(program[1][0][2][:2]), 
        token_position_strategy, # this will overwrite the previous arg!
        tokenizer=AutoTokenizer.from_pretrained("gpt2"),
        preload_dataset=test_cdataset
    )["train"]
else:
    counterfactual_data_module = make_supervised_counterfactual_data_module(
        program,
        args.alignment_variable,
        args.n_training_examples,
        max(program[1][0][2][:2]), 
        token_position_strategy, # this will overwrite the previous arg!
        tokenizer=AutoTokenizer.from_pretrained("gpt2")
    )
    train_cdataset = counterfactual_data_module["train_dataset"]
    validation_cdataset = counterfactual_data_module["eval_dataset"]
    test_cdataset = counterfactual_data_module["test_dataset"]

left_tokenizer = AutoTokenizer.from_pretrained(
    "gpt2",
    padding_side='left',
    use_fast=False
)
left_tokenizer.pad_token = left_tokenizer.eos_token

train_dataloader = DataLoader(
    train_cdataset,
    batch_size=args.batch_size, 
    sampler=RandomSampler(train_cdataset),
    collate_fn=DataCollatorForAlignmentDataset(
        left_tokenizer,
        label_pad_token_id=IGNORE_INDEX,
        padding="longest"
    )
)
# use different tokenizer padding
eval_dataloader = DataLoader(
    validation_cdataset,
    batch_size=args.batch_size, 
    sampler=RandomSampler(validation_cdataset),
    collate_fn=DataCollatorForAlignmentDataset(
        left_tokenizer,
        label_pad_token_id=IGNORE_INDEX,
        padding="longest"
    )
)
test_dataloader = DataLoader(
    test_cdataset,
    batch_size=args.batch_size, 
    sampler=RandomSampler(test_cdataset),
    collate_fn=DataCollatorForAlignmentDataset(
        left_tokenizer,
        label_pad_token_id=IGNORE_INDEX,
        padding="longest"
    )
)

def compute_metrics(eval_preds, eval_labels):
    total_count = 0
    correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        actual_test_labels = eval_label[:, -1]
        pred_test_labels = torch.argmax(eval_pred[:, -2], dim=-1)
        correct_labels = (actual_test_labels==pred_test_labels)
        total_count += len(correct_labels)
        correct_count += correct_labels.sum().tolist()
    accuracy = round(correct_count/total_count, 2)
    return {"accuracy" : accuracy}

###################
# model object loading
###################
alignment_config = {
    'layer': args.alignment_layer,
    'num_of_das_token' : 1
}
model = AutoAlignableModel.from_pretrained(
    args.model_name_or_path,
    alignment_config=alignment_config,
    torch_dtype=torch.bfloat16 if args.bf16 else None,
    cache_dir="../../../.cache_dir/"
)
# set off the gradients among all other layers.
for name, param in model.named_parameters():
    if "rotate_layer" not in name and "intervention_boundaries" not in name:
        param.requires_grad = False
    else:
        logger.info(f"Requiring gradients on layer: {name}")
t_total = int(len(train_dataloader) * args.epochs)
warm_up_steps = 0.1 * t_total
optimizer = torch.optim.Adam(
    [{'params': model.transformer.rotate_layer.parameters()},
    {'params': model.transformer.intervention_boundaries, 'lr': 5e-3}],
    lr=args.lr
)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warm_up_steps,
    num_training_steps=t_total
)

model_type = AutoConfig.from_pretrained(
    args.model_name_or_path
).architectures[0]

run_name = f"{model_type}.alignment_variable.{args.alignment_variable}.intl.{alignment_config['layer']}.word_beam.{target_word_beam}.token_position_strategy."\
           f"{args.token_position_strategy}.lr.{args.lr}.seed.{args.seed}"
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
os.environ["WANDB_PROJECT"] = args.wandb_project
output_dir = os.path.join(args.output_dir, run_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if args.is_wandb:
    import wandb
    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
    )
    
n_gpu = torch.cuda.device_count()
if n_gpu > 1:
    print("Multi-gpu training initialized.")
    model = torch.nn.DataParallel(model)
model.to(torch.device("cuda")) # no rank is needed!

aligner = Aligner(
    model,
    logger=logger,
    is_wandb=args.is_wandb,
    is_master=True,
    n_gpu=n_gpu,
    model_name=run_name,
    device="cuda",
    compute_metrics=compute_metrics
)

# Train
if args.do_align:
    aligner.train(
        train_dataloader, 
        eval_dataloader, 
        test_dataloader,
        optimizer, 
        scheduler, 
        log_step=args.log_step, 
        valid_steps=args.valid_steps,
        output_dir=output_dir, 
        epochs=args.epochs, 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )