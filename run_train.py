#!/usr/bin/env python
# coding: utf-8

from utils.train_utils import *
from logic_data.utils import *
from transformers import AutoTokenizer

if __name__ == '__main__':
    is_notebook = False
    try:
        cmd = argparse.ArgumentParser('The testing components of')
        cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
        cmd.add_argument('--train_batch_size', default=128, type=int, help='training batch size')
        cmd.add_argument('--eval_batch_size', default=128, type=int, help='training batch size')
        cmd.add_argument('--lr', default=0.01, type=float, help='learning rate')
        cmd.add_argument('--data_path', required=True, type=str, help='path to the training corpus')
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
        cmd.add_argument('--hf_model_path', type=str, required=False, default=None)
        cmd.add_argument('--model_path', type=str, required=False, default=None)
        cmd.add_argument('--warm_up', type=float, default=0.1)
        cmd.add_argument('--is_wandb', default=False, action='store_true')
        cmd.add_argument('--log_step', default=10, type=int)
        cmd.add_argument('--valid_steps', default=100, type=int)
        cmd.add_argument('--early_stopping', default=999, type=int)
        cmd.add_argument('--device', default="cuda", type=str, help='')
        cmd.add_argument('--do_train', default=False, action='store_true')
        cmd.add_argument('--do_eval', default=False, action='store_true')
        cmd.add_argument('--do_test', default=False, action='store_true')
        
        cmd.add_argument('--n_training_program', default=11, type=int)
        cmd.add_argument('--n_fewshot', default=10, type=int)
        
        cmd.add_argument('--n_training_examples', default=100000, type=int)
        cmd.add_argument('--n_eval_examples', default=1000, type=int)
        cmd.add_argument('--n_test_examples', default=1000, type=int)
        
        args = cmd.parse_args(sys.argv[1:])
    except:
        assert False # NEVER!

set_seed(args.seed)
model_name = args.hf_model_path if args.hf_model_path else "gpt2"
run_name = f"logic_pipeline.model.{model_name}.n_rule.{args.n_training_program}.n_shot.{args.n_fewshot}.seed.{args.seed}"
logger = logging.getLogger()

# Data is generated on-fly
training_clauses = pickle.load(open(os.path.join(args.data_path, "cfg_train.pkl"), 'rb'))
training_clauses = random.sample(training_clauses, k=args.n_training_program)
eval_clauses = pickle.load(open(os.path.join(args.data_path, "cfg_train.pkl"), 'rb'))
tests_clauses = pickle.load(open(os.path.join(args.data_path, "cfg_test.pkl"), 'rb'))

tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path if args.hf_model_path is not None else args.model_path)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.true_token_id = tokenizer("True")['input_ids'][0]
tokenizer.false_token_id = tokenizer("False")['input_ids'][0]
tokenizer.sep_token_id = tokenizer(";")['input_ids'][0]
tokenizer.input_token_id = tokenizer("Input")['input_ids'][0]
tokenizer.output_token_id = tokenizer("Output")['input_ids'][0]

logging.info(f'Generating data on-fly ...')
raw_train = sample_factual_demonstrations(
    training_clauses, args.n_training_examples, 
    args.n_fewshot+1, tokenizer
)
raw_eval = sample_factual_demonstrations(
    eval_clauses, args.n_eval_examples, 
    args.n_fewshot+1, tokenizer
)
raw_test = sample_factual_demonstrations(
    tests_clauses, args.n_test_examples, 
    args.n_fewshot+1, tokenizer
)
train_dataset = Dataset.from_dict(
    {"input_ids": raw_train[0], "labels": raw_train[1]}
).with_format("torch")
train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)

dev_dataset = Dataset.from_dict(
    {"input_ids": raw_eval[0], "labels": raw_eval[1]}
).with_format("torch")
dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size)

test_dataset = Dataset.from_dict(
    {"input_ids": raw_test[0], "labels": raw_test[1]}
).with_format("torch")
test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size)

# Model
torch.cuda.empty_cache()

if args.hf_model_path is None:
    configuration = GPT2Config.from_pretrained(os.path.join(args.data_path, "decoder_config.json"))
    model = CustomizedGPT2LMHeadModel(configuration)
else:
    logging.info(f'Loading models from HF: {args.hf_model_path}')
    model = CustomizedGPT2LMHeadModel.from_pretrained(
        args.hf_model_path,
        cache_dir="../huggingface_cache"
    )
    
if args.model_path is not None:
    logging.info("Loading pretrained model.")
    raw_weights = torch.load(os.path.join(args.model_path, 'pytorch_model.bin'))
    model.load_state_dict(raw_weights)

device = torch.device(args.device)
if "cuda:" not in args.device:
    n_gpu = torch.cuda.device_count()
    logging.info(f'__Number CUDA Devices: {n_gpu}')
else:
    n_gpu = 1
    logging.info(f'__Number CUDA Devices: {n_gpu}')

if n_gpu > 1:
    model = torch.nn.DataParallel(model)
_ = model.to(device)

t_total = int(len(train_dataloader) * args.epochs)

warm_up_steps = args.warm_up * t_total
optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.lr
)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps,
                                            num_training_steps=t_total)
is_master = True                                    
if not os.path.exists(args.output_dir) and is_master:
    os.mkdir(args.output_dir)

os.environ["WANDB_PROJECT"] = f"ToM-DAS"

output_dir = os.path.join(args.output_dir, run_name)
if args.do_train and args.is_wandb:
    run = wandb.init(
        project="ToM-DAS-GPT2", 
        entity="wuzhengx",
        name=run_name,
    )
    wandb.config.update(args)
if not os.path.exists(args.output_dir) and is_master:
    os.mkdir(args.output_dir)
    
trainer = LogicSolverTrainer(
    model, device=device, 
    logger=logger,
    is_master=is_master, 
    n_gpu=n_gpu,
    is_wandb=args.is_wandb, 
    model_name=model_name,
)
num_params = count_parameters(model)
logging.info(f'Number of {model_name} model params: {num_params}')

# Train
if args.do_train:
    logging.info(f"OUTPUT DIR: {output_dir}")
    trainer.train(
        train_dataloader, dev_dataloader,
        optimizer, scheduler, 
        log_step=args.log_step, valid_steps=args.valid_steps,
        output_dir=output_dir, epochs=args.epochs, 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

if args.do_train and args.is_wandb:
    wandb.finish()

# Dev
if args.do_eval:
    total_count = 0
    correct_count = 0
    if args.do_eval:
        trainer.model.eval()
        epoch_iterator = tqdm(dev_dataloader, desc="Iteration", position=0, leave=True)
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            outputs = model(**inputs)

            actual_test_labels = inputs['labels'][:, -3]
            pred_test_labels = torch.argmax(outputs.logits[:, -4], dim=-1)
            correct_labels = (actual_test_labels==pred_test_labels)

            total_count += len(correct_labels)
            correct_count += correct_labels.sum().tolist()

            current_acc = round(correct_count/total_count, 2)
            epoch_iterator.set_postfix({'acc': current_acc})
            
# Test
if args.do_test:
    total_count = 0
    correct_count = 0
    if args.do_test:
        trainer.model.eval()
        epoch_iterator = tqdm(test_dataloader, desc="Iteration", position=0, leave=True)
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            outputs = model(**inputs)

            actual_test_labels = inputs['labels'][:, -3]
            pred_test_labels = torch.argmax(outputs.logits[:, -4], dim=-1)
            correct_labels = (actual_test_labels==pred_test_labels)

            total_count += len(correct_labels)
            correct_count += correct_labels.sum().tolist()

            current_acc = round(correct_count/total_count, 2)
            epoch_iterator.set_postfix({'acc': current_acc})
