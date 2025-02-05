#!/usr/bin/env python
# coding=utf-8

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import webdataset as wds

import transformers
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from model.modeling_zrigf_2 import ZRIGF2ForConditionalGeneration, concat_text_input_output

logger = logging.getLogger(__name__)


I2T_PROMPTS = [
    "A short image caption:",
    "A short image description:",
    "An image that shows",
    "Write a short description for the image.",
    "Write a description for the photo.",
    "Provide a description of what is presented in the photo.",
    "Briefly describe the content of the image.",
    "Can you briefly explain what you see in the image?",
    "Could you use a few words to describe what you perceive in the photo?",
    "Please provide a short depiction of the picture.",
    "Use a few words to illustrate what is happening in the picture.",
]

T2I_PROMPTS = [
    "Draw an image based on this description:",
    "Create an image that reflects the following description:",
    "Generate a picture that matches this text:",
    "Illustrate the following scene:",
    "Construct a visual interpretation of the following text:",
    "Design an image according to this description:",
    "Can you visually represent the text below?",
    "Please generate an image that embodies the following narrative:",
    "Produce a visual representation of the following description:",
    "Depict the text in a picture:",
    "Render an image following the provided textual cue:",
]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    clip_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained clip model or model identifier from huggingface.co/models"},
    )
    qformer_name_or_path: str = field(
        metadata={"help": "Path to pretrained qformer model or model identifier from huggingface.co/models"},
    )
    language_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained large language model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    freeze_clip_model: bool = field(
        default=True, metadata={"help": "Whether to freeze the clip model parameters or not."}
    )
    freeze_language_model: bool = field(
        default=True, metadata={"help": "Whether to freeze the language model parameters or not."}
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the "
                "pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(default=None, metadata={"help": "The name of the dataset to use"})
    mask_patch_size: int = field(default=28, metadata={"help": "The size of the square patches to use for masking."})
    mask_ratio: float = field(
        default=0.6,
        metadata={"help": "Percentage of patches to mask."},
    )
    unchanged_ratio: float = field(
        default=0.2,
        metadata={"help": "Percentage of unchanged patches to mask."},
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Percentage of text to mask."},
    )
    max_seq_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


dataset_name_mapping = {
    "coco": {
        "train": "data/web_coco/train-{000000..000011}.tar",
        "valid": "data/web_coco/val-000000.tar",
        "train_len": 594144,
    },
    "ccs": {
        # "train": "data/cc_sbu_dataset/{00000..01250}.tar",
        "train": "data/cc_sbu_dataset/{00000..00069}.tar",
        "valid": "data/cc_sbu_dataset/{01251..01255}.tar",
        "train_len": 10224679,
    },
}


class MaskGenerator:
    """
    A class to generate boolean masks for the pretraining task.

    A mask is a 1D tensor of shape (model_patch_size**2,) where the value is either 0 or 1,
    where 1 indicates "masked".
    """

    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        if self.input_size % self.mask_patch_size != 0:
            raise ValueError("Input size must be divisible by mask patch size")
        if self.mask_patch_size % self.model_patch_size != 0:
            raise ValueError("Mask patch size must be divisible by model patch size")

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return torch.tensor(mask.flatten())


def main():
    # 1. Parse input arguments
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

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3. Detecting last checkpoint and eventualy continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 4. Load pretrained model, tokenizer, and image processor
    clip_processor = AutoProcessor.from_pretrained(model_args.clip_model_name_or_path)
    image_processor = clip_processor.image_processor
    text_tokenizer = clip_processor.tokenizer
    llm_tokenizer = AutoTokenizer.from_pretrained(model_args.language_model_name_or_path)

    model = ZRIGF2ForConditionalGeneration.from_clip_qformer_llm_pretrained(
        model_args.clip_model_name_or_path,
        model_args.qformer_name_or_path,
        model_args.language_model_name_or_path,
        cache_dir=model_args.cache_dir,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        torch_dtype=torch.bfloat16,
    )

    model.unchanged_ratio = data_args.unchanged_ratio
    config = model.config
    vision_config = config.clip_config.vision_config

    max_seq_length = min(data_args.max_seq_length, config.clip_config.text_config.max_position_embeddings)

    # llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # model.resize_token_embeddings(len(llm_tokenizer))
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_clip_model:
        _freeze_params(model.clip_text_model)
        _freeze_params(model.clip_text_projection)
        _freeze_params(model.clip_vision_model)
        _freeze_params(model.clip_visual_projection)

    if model_args.freeze_language_model:
        _freeze_params(model.language_model)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # 4. Load and preprocess the dataset
    def coco_compose(src):
        for sample in src:
            image, labels = sample
            for label in labels.split('\n'):
                yield image, label

    def to_dict(sample):
        return {
            "image": sample[0],
            "caption": sample[1] if sample[1].endswith('.') else sample[1] + '.',
            "t2i_prompt": random.choice(T2I_PROMPTS),
            "i2t_prompt": random.choice(I2T_PROMPTS),
        }

    dataset_names = data_args.dataset_name.split(',')

    def create_dataset_pipeline(data_name, dataset_type):
        if dataset_type == 'train':
            common_pipeline = [wds.ResampledShards(dataset_name_mapping[data_name][dataset_type])]
        else:
            common_pipeline = [
                wds.SimpleShardList(dataset_name_mapping[data_name][dataset_type]),
                wds.split_by_worker,
            ]

        common_pipeline.extend([
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pil", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "txt", handler=wds.warn_and_continue)
        ])

        if data_name == 'coco':
            common_pipeline.extend([
                coco_compose,
                wds.shuffle(1000, handler=wds.warn_and_continue)
            ])

        common_pipeline.append(wds.map(to_dict, handler=wds.warn_and_continue))

        return wds.DataPipeline(*common_pipeline)

    datasets = {}

    if 'coco' in dataset_names:
        datasets['coco_train'] = (
            create_dataset_pipeline('coco', 'train')
            .repeat(2).with_epoch(dataset_name_mapping['coco']['train_len'] * 4)
        )
        datasets['coco_valid'] = create_dataset_pipeline('coco', 'valid')

    if 'ccs' in dataset_names:
        datasets['ccs_train'] = (
            create_dataset_pipeline('ccs', 'train')
            .with_epoch(dataset_name_mapping['ccs']['train_len'])
        )
        datasets['ccs_valid'] = create_dataset_pipeline('ccs', 'valid')

    # If both datasets are present, use a random mix
    if 'coco' in dataset_names and 'ccs' in dataset_names:
        # train_dataset = wds.RoundRobin(
        #     [datasets['coco_train'], datasets['ccs_train']], longest=True
        # )
        train_dataset = wds.RandomMix(
            [datasets['coco_train'], datasets['ccs_train']], probs=[0.2, 0.8], longest=True
        )
        eval_dataset = wds.RoundRobin([datasets['coco_valid'], datasets['ccs_valid']], longest=True)
    else:
        # Defaulting to single datasets
        train_dataset = datasets.get('coco_train', datasets.get('ccs_train'))
        eval_dataset = datasets.get('coco_valid', datasets.get('ccs_valid'))

    # create mask generator
    mask_generator = MaskGenerator(
        input_size=vision_config.image_size,
        mask_patch_size=data_args.mask_patch_size,
        model_patch_size=vision_config.patch_size,
        mask_ratio=data_args.mask_ratio,
    )

    def collate_fn(examples):
        captions = list([example["caption"] for example in examples])
        i2t_prompts = list(["Image: " + example["i2t_prompt"] for example in examples])
        t2i_prompts = list([example["t2i_prompt"] + " " + example["caption"] + " Image:" for example in examples])

        text_inputs = text_tokenizer(
            captions,
            max_length=max_seq_length,
            padding="longest",
            return_tensors="pt",
            truncation=True,
        )

        t2i_llm_inputs = llm_tokenizer(
            t2i_prompts,
            max_length=data_args.max_seq_length,
            padding="longest",
            return_tensors="pt",
            truncation=True,
        )
        i2t_prompt_inputs = llm_tokenizer(
            i2t_prompts,
            max_length=data_args.max_seq_length,
            padding="longest",
            return_tensors="pt",
            truncation=True,
        )
        caption_inputs = llm_tokenizer(
            [t + llm_tokenizer.eos_token for t in captions],
            max_length=data_args.max_seq_length,
            padding="longest",
            return_tensors="pt",
            truncation=True,
        )

        i2t_llm_inputs, input_part_targets_len = concat_text_input_output(
            i2t_prompt_inputs.input_ids,
            i2t_prompt_inputs.attention_mask,
            caption_inputs.input_ids,
            caption_inputs.attention_mask,
        )

        # do not apply loss to the padding
        targets = i2t_llm_inputs['input_ids'].masked_fill(
            i2t_llm_inputs['input_ids'] == llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        images = list([example['image'] for example in examples])
        pixel_values = image_processor(images, return_tensors="pt")['pixel_values']
        bool_masked_pos = torch.stack([mask_generator() for _ in range(len(images))])

        masked_text_labels = text_inputs.input_ids.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `data_args.mlm_probability`)
        probability_matrix = torch.full(masked_text_labels.shape, data_args.mlm_probability)
        special_tokens_mask = [
            text_tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in masked_text_labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_text_labels[~masked_indices] = -100  # We only compute loss on masked tokens

        return {
            "text_input_ids": text_inputs.input_ids,
            "text_attention_mask": text_inputs.attention_mask,
            "i2t_llm_input_ids": i2t_llm_inputs["input_ids"],
            "i2t_llm_attention_mask": i2t_llm_inputs["attention_mask"],
            "t2i_llm_input_ids": t2i_llm_inputs["input_ids"],
            "t2i_llm_attention_mask": t2i_llm_inputs["attention_mask"],
            "pixel_values": pixel_values,
            "bool_masked_pos": bool_masked_pos,
            "masked_indices": masked_indices,
            "masked_text_labels": masked_text_labels,
            "labels": targets,
        }

    # 8. Initalize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
    )

    # 9. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        text_tokenizer.save_pretrained(os.path.join(training_args.output_dir, "text_tokenizer"))
        llm_tokenizer.save_pretrained(os.path.join(training_args.output_dir, "llm_tokenizer"))
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 10. Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
