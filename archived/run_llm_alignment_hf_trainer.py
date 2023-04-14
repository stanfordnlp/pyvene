from utils.train_utils import *

class LLMsDASTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # source activations.
        source_hidden_states = self.model(
           input_ids=inputs['source_input_ids'],
           output_rotated_hidden_states_only=True
        ).rotated_hidden_states
        # base forward + source activations.
        outputs = self.model(
            input_ids=inputs['input_ids'],
            source_hidden_states=source_hidden_states,
            intervention_ids=inputs['intervention_ids'],
            labels=inputs['labels']
        )
        return (outputs.loss, outputs) if return_outputs else outputs.loss
    
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
        
    task_config: str = field(
        default="3.50;8.50;0.00;9.99",
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="../alpaca_7b/",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

    unit_test_mode: bool = field(
        default=False,
        metadata={"help": "Will load a dummy model instead in testing mode."},
    )
    aligning_tokens: str = field(
        default="79;80",
        metadata={"help": "Aligning tokens in string format. Zero-indexed. <Start Token Idx>:<End Token Idx>"},
    ) 
    aligning_var_n: int = field(
        default=2,
        metadata={"help": "Number of high-level variables we are aligning. Max 2."},
    )
    aligning_basis_n_per_variable: int = field(
        default=128,
        metadata={"help": "Number of basis per high-level variable aligning."},
    ) 
    aligning_layer_n: int = field(
        default=15,
        metadata={"help": "Aligning layer number. Zero-indexed."},
    )
        
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    alignment_config = {
        'layer': model_args.aligning_layer_n,
        "token_range" : [
            int(model_args.aligning_tokens.split(";")[0]), 
            int(model_args.aligning_tokens.split(";")[1]), 
        ]
    }
    logger.info(f"alignment_config = {alignment_config}")
    if model_args.aligning_var_n == 1:
        intervention_config = {
            0: [[0, model_args.aligning_basis_n_per_variable]]
        }
    elif model_args.aligning_var_n == 2:
        intervention_config = {
            0: [[0, model_args.aligning_basis_n_per_variable]],
            1: [[model_args.aligning_basis_n_per_variable, 2*model_args.aligning_basis_n_per_variable]],
        }
    logger.info(f"intervention_config = {intervention_config}")
    
    run_name = f"alpaca-7B.task.{data_args.task_name}.config.{data_args.task_config}."\
               f"seed.{training_args.seed}.intl.{model_args.aligning_layer_n}.intr.{alignment_config['token_range'][0]}to"\
               f"{alignment_config['token_range'][1]}"
    training_args.output_dir = os.path.join(training_args.output_dir, run_name)
    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    
    train_dataset, eval_dataset = prepare_dataset(data_args, tokenizer)
    
    ###################
    # model object loading
    ###################
    
    if model_args.unit_test_mode:
        logger.info("Loading Dummy Model for Testing ...")
        # Testing code.
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
        )
        config.intermediate_size = 512
        config.hidden_size = 512
        config.num_attention_heads = 2
        config.num_hidden_layers = 32
        model = AlignableLlamaForCausalLM(
            config=config,
            alignment_config=alignment_config,
            intervention_config=intervention_config
        )
    else:
        logger.info("Loading Alpaca 7B, Takes 2 Mins ...")
        model = AlignableLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            alignment_config=alignment_config,
            intervention_config=intervention_config
        )
    
    # set off the gradients among all other layers.
    for name, param in model.named_parameters():
        if "rotate_layer" not in name:
            param.requires_grad = False
    
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        
        actual_test_labels = p.label_ids[:, -1]
        pred_test_labels = np.argmax(preds[:, -1], axis=1)
        correct_labels = (actual_test_labels==pred_test_labels)
        step_accuracy = correct_labels.sum() / p.label_ids.shape[0]
        step_accuracy = step_accuracy.tolist()

        return {"accuracy": step_accuracy}
        
    # Initialize our Trainer
    trainer = LLMsDASTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=None)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)
            
if __name__ == "__main__":
    main()
