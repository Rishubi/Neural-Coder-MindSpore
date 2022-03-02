import collections.abc
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets.utils as datasets_utils

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    GPTJForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from codex.data import load_tokenized_codeparrot, load_tokenized_code_data, load_fused_code, load_tokenized_coq_data

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0.dev0")

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# The first id of the extra token we use to encode whitespaces of different length.
# By setting this to 10, we will encode 2 whitespaces as <|extratoken_10|>,
#   3 as <|extratoken_11|>, ..., and n whitespaces as <|extra_token_{n+2}|>.
EXTRA_TOKEN_START = 10
ENCODE_MAX_WHITESPACE_SPAN = 10


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the data library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the data library)."}
    )
    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The path of user-defined data. Necessary for PyTorrent and CodeParrot."}
    )
    skip_data_files: int = field(
        default=0, metadata={"help": "The number of data files to skip."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization."
                    "The training dataset will be truncated in block of this size for training."
                    "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    encode_whitespaces: bool = field(
        default=True,
        metadata={"help": "Whether to encode whitespace runs of different lengths to special tokens."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
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
    datasets_utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the data: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public data available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the data Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    logger.info('Loading dataset ...')

    if data_args.dataset_name == 'codeparrot':
        assert data_args.dataset_path is not None, 'Should give dataset path'
        raw_datasets = load_tokenized_codeparrot(data_args.dataset_path)
    elif data_args.dataset_name == 'code_data':
        assert data_args.dataset_path is not None, 'Should give dataset path'
        raw_datasets = load_tokenized_code_data(data_args.dataset_path, num_skip_files=data_args.skip_data_files)
    elif data_args.dataset_name == 'coq_data':
        assert data_args.dataset_path is not None, 'Should give dataset path'
        raw_datasets = load_tokenized_coq_data(data_args.dataset_path, num_skip_files=data_args.skip_data_files)
    elif data_args.dataset_name == 'fused_code':
        assert data_args.dataset_path is not None, 'Should give dataset path'
        raw_datasets = load_fused_code(data_args.dataset_path, num_skip_files=data_args.skip_data_files)
    else:
        raise RuntimeError(f'Un-recoginized dataset name: {data_args.dataset_name}')

    # if data_args.dataset_name is not None:
    #     # Downloading and loading a dataset from the hub.
    #     if data_args.dataset_name == 'pytorrent':
    #         assert data_args.dataset_path is not None, 'Should give dataset path'
    #         raw_datasets = load_pytorrent(data_args.dataset_path)
    #         # renaming rows
    #         raw_datasets['train'] = concatenate_datasets([datasets['train'], datasets.pop('test')])
    #         raw_datasets['validation'] = datasets.pop('valid')
    #     elif data_args.dataset_name == 'codeparrot':
    #         assert data_args.dataset_path is not None, 'Should give dataset path'
    #         raw_datasets = load_codeparrot(data_args.dataset_path)
    #     else:
    #         raw_datasets = load_dataset(data_args.dataset_name,
    #                                     data_args.dataset_config_name)
    #
    #     # if "validation" not in data.keys():
    #     #     data["validation"] = load_dataset(
    #     #         data_args.dataset_name,
    #     #         data_args.dataset_config_name,
    #     #         split=f"train[:{data_args.validation_split_percentage}%]",
    #     #     )
    #     #     data["train"] = load_dataset(
    #     #         data_args.dataset_name,
    #     #         data_args.dataset_config_name,
    #     #         split=f"train[{data_args.validation_split_percentage}%:]",
    #     #     )
    # else:
    #     data_files = {}
    #     if data_args.train_file is not None:
    #         data_files["train"] = data_args.train_file
    #     if data_args.validation_file is not None:
    #         data_files["validation"] = data_args.validation_file
    #     extension = (
    #         data_args.train_file.split(".")[-1]
    #         if data_args.train_file is not None
    #         else data_args.validation_file.split(".")[-1]
    #     )
    #     if extension == "txt":
    #         extension = "text"
    #
    #     raw_datasets = load_dataset(
    #         extension, data_files=data_files)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")

    # Gradient checkpointing and caching
    # config.gradient_checkpointing = True
    config.use_cache = False

    logger.info('loading tokenizer')
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs)

    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    logger.info('loading model')
    if model_args.model_name_or_path:
        model = GPTJForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = GPTJForCausalLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params / 2 ** 20:.2f}M params")

    # Do not resize the embeddings for GPT-J models
    # if model_args.model_name_or_path != "EleutherAI/gpt-j-6B":
    #     model.resize_token_embeddings(len(tokenizer))

    # def get_column_name(dataset):
    #     if isinstance(dataset, IterableDataset):
    #         for sample in dataset:
    #             return list(sample.keys())

    logger.info('enabling gradient checkpointing')
    model.gradient_checkpointing_enable()

    # Preprocessing the data.
    # First we tokenize all the texts.
    # if training_args.do_train:
    #     column_names = raw_datasets["train"].column_names
    # else:
    #     column_names = raw_datasets["validation"].column_names
    # text_column_name = "text" if "text" in column_names else column_names[0]
    # if data_args.dataset_name == 'pytorrent':
    #     text_column_name = 'func_code'
    # elif data_args.dataset_name == 'codeparrot':
    #     text_column_name = 'content'
    # elif data_args.dataset_name == 'code_data':
    #     text_column_name = 'code'
    # else:
    #     assert False, f'Unknown text column for dataset {data_args.dataset_name}'

    # def encode_whitespaces(text):
    #     def push_acc_space(acc_len: int, text: str):
    #         if acc_len == 0:
    #             return text
    #         if acc_len == 1:
    #             return text + ' '
    #         assert acc_len <= ENCODE_MAX_WHITESPACE_SPAN, \
    #             f'Max whitespace run length {ENCODE_MAX_WHITESPACE_SPAN}, but found {acc_len}'
    #         extra_id = EXTRA_TOKEN_START - 2 + acc_len
    #         extra_token = f'<|extratoken_{extra_id}|>'
    #         return text + extra_token
    #
    #     acc_len = 0
    #     res = ''
    #     for ch in text:
    #         if ch == ' ':
    #             acc_len += 1
    #             if acc_len == ENCODE_MAX_WHITESPACE_SPAN:
    #                 res = push_acc_space(acc_len, res)
    #                 acc_len = 0
    #         else:
    #             res = push_acc_space(acc_len, res)
    #             acc_len = 0
    #             res = res + ch
    #
    #     res = push_acc_space(acc_len, res)
    #
    #     return res
    #
    # def tokenize_function(examples):
    #     texts = examples[text_column_name]
    #     tic = perf_counter()
    #     texts = [encode_whitespaces(text) for text in texts]
    #     toc = perf_counter()
    #     logger.debug(f'(TIMING) took {toc - tic} sec in whitespace processing, {len(texts)} samples')
    #     tic = perf_counter()
    #     tokenized = tokenizer(texts)
    #     toc = perf_counter()
    #     logger.debug(f'(TIMING) took {toc - tic} sec in tokenization')
    #     return tokenized
    #
    # logger.info('tokenizing dataset')
    # with training_args.main_process_first(desc="dataset map tokenization"):
    #     tokenized_datasets = {
    #         k: raw_datasets[k].map(
    #             tokenize_function,
    #             batched=True
    #         )
    #         for k in raw_datasets.keys()
    #     }

    # if data_args.block_size is None:
    #     block_size = tokenizer.model_max_length
    #     if block_size > 1024:
    #         logger.warning(
    #             f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
    #             "Picking 1024 instead. You can change that default value by passing --block_size xxx."
    #         )
    #     block_size = 1024
    # else:
    #     if data_args.block_size > tokenizer.model_max_length:
    #         logger.warning(
    #             f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
    #             f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
    #         )
    #     block_size = min(data_args.block_size, tokenizer.model_max_length)
    #
    # logger.info(f'extracted block size: {block_size}')

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    # def group_texts(examples):
    #     # Concatenate all texts.
    #     tic = perf_counter()
    #     concatenated_examples = {
    #         k: sum(examples[k], []) for k in examples.keys()}
    #     total_length = len(concatenated_examples[list(examples.keys())[0]])
    #     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    #     # customize this part to your needs.
    #     total_length = (total_length // block_size) * block_size
    #     # Split by chunks of max_len.
    #     result = {
    #         k: [t[i: i + block_size]
    #             for i in range(0, total_length, block_size)]
    #         for k, t in concatenated_examples.items()
    #     }
    #     result["labels"] = result["input_ids"].copy()
    #     toc = perf_counter()
    #
    #     num_input_samples = len(examples['input_ids'])
    #     num_output_samples = len(result['input_ids'])
    #
    #     logger.debug(f'(TIMING) took {toc - tic} sec in grouping, '
    #                  f'input {num_input_samples} samples, '
    #                  f'output {num_output_samples} samples')
    #     return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    # logger.info('grouping dataset')
    # with training_args.main_process_first(desc="grouping texts together"):
    #     lm_datasets = {
    #         k: tokenized_datasets[k].map(
    #             group_texts,
    #             batched=True
    #         )
    #         for k in tokenized_datasets.keys()
    #     }

    # No processing is needed. It is done in the pre-processing step.
    lm_datasets = raw_datasets

    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]

        # Wrap training dataset as a PyTorch IterableDataset
        # if isinstance(train_dataset, datasets.IterableDataset):
        #     train_dataset = iter_wrapper(train_dataset)

        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(
                range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]

        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(
                range(data_args.max_val_samples))

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(
            eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
