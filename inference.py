#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import logging
import math
import os
import re
from collections import Counter

import datasets
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from nlgeval import NLGEval
from nltk import ngrams
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoTokenizer,
    SchedulerType,
    GenerationConfig,
    get_scheduler,
)

from model.configuration_zrigf_2 import ZRIGF2Config
from model.modeling_zrigf_2 import ZRIGF2ForConditionalGeneration, concat_text_input_output


torch.backends.cuda.matmul.allow_tf32 = True

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-resource inference")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        required=True,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to pretrained model",
        required=True,
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=32,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--train_data_size", type=str, default="1", choices=["0", "1/4", "1/8","1"]
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_kwargs = {}
    if args.with_tracking:
        accelerator_kwargs["log_with"] = args.report_to
        accelerator_kwargs["project_dir"] = args.output_dir
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    # datasets.utils.logging.set_verbosity_error()
    # transformers.utils.logging.set_verbosity_error()
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'commandline_args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    accelerator.wait_for_everyone()

    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(args.dataset_name)

    # Load pretrained model and tokenizer

    config = ZRIGF2Config.from_pretrained(args.model_path)
    config.pretraining = False
    model = ZRIGF2ForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, config=config,
    )
    text_tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_path, "text_tokenizer"))
    llm_tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_path, "llm_tokenizer"))

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    _freeze_params(model.clip_text_model)
    _freeze_params(model.clip_text_projection)
    _freeze_params(model.language_model)

    generation_config = GenerationConfig.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    generation_config.max_new_tokens = args.val_max_target_length if args.val_max_target_length is not None else args.max_target_length
    generation_config.do_sample = False
    generation_config.num_beams = 3
    generation_config.no_repeat_ngram_size = 3

    # Temporarily set max_target_length for training.
    max_source_length = min(args.max_source_length, config.clip_config.text_config.max_position_embeddings)

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    if args.train_data_size == "1":
        max_train_samples = len(train_dataset)
    elif args.train_data_size == "1/4":
        max_train_samples = len(train_dataset) // 4
    else:
        max_train_samples = len(train_dataset) // 8

    train_dataset = train_dataset.select(range(max_train_samples))

    def normalize_answer(s):
        """
        Lower text and remove punctuation, articles and extra whitespace.
        """

        s = s.lower()
        re_art = re.compile(r'\b(a|an|the)\b')
        re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
        s = re_punc.sub(' ', s)
        s = re_art.sub(' ', s)
        # TODO: this could almost certainly be faster with a regex \s+ -> ' '
        s = ' '.join(s.split())
        return s

    def remove_non_utf8_and_emojis(text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)

        # Remove specified Unicode characters
        specific_char_pattern = "[\uD83D\uFFFD\uFE0F\u203C\u3010\u3011\u300A\u166D\u200C\u202A\u202C\u2049\u20E3\u300B\u300C\u3030\u065F\u0099\u0F3A\u0F3B\uF610\uFFFC]"
        text = re.sub(specific_char_pattern, "", text)

        return text

    def postprocess_text(preds, labels):
        preds = [remove_non_utf8_and_emojis(pred) for pred in preds]
        preds = [pred.encode('utf-8', 'ignore').decode('utf-8') for pred in preds]
        labels = [label.strip() for label in labels]

        preds = [' '.join(word_tokenize(pred)) for pred in preds]
        labels = [' '.join(word_tokenize(label)) for label in labels]

        preds = [re.sub(r'\[.*$', '', pred.lower()) for pred in preds]
        preds = [re.sub(u'\uFFFD', '', pred) for pred in preds]

        return preds, labels

    def compute_distinct(preds):
        unigram_counter, bigram_counter = Counter([]), Counter([])
        for pred in preds:
            pred_for_cal = pred.split()
            unigram_counter.update(pred_for_cal)
            bigram_counter.update(ngrams(pred_for_cal, 2))

        try:
            distinct_1 = len(unigram_counter) / (sum(unigram_counter.values()))
        except ZeroDivisionError:
            distinct_1 = 0
        try:
            distinct_2 = len(bigram_counter) / (sum(bigram_counter.values()))
        except ZeroDivisionError:
            distinct_2 = 0

        return distinct_1, distinct_2

    def collate_fn(examples):
        contexts = list([example["context"] for example in examples])
        conversations = list([example["conversation"] for example in examples])
        responses = list([example["response"] for example in examples])

        text_inputs = text_tokenizer(
            contexts,
            max_length=max_source_length,
            padding="longest",
            return_tensors="pt",
            truncation=True,
        )

        llm_inputs = llm_tokenizer(
            conversations,
            max_length=args.max_source_length,
            padding="longest",
            return_tensors="pt",
            truncation=True,
        )
        llm_outputs = llm_tokenizer(
            text_target=[r + llm_tokenizer.eos_token for r in responses],
            max_length=args.max_target_length,
            padding="longest",
            return_tensors="pt",
            truncation=True,
        )

        i2t_llm_inputs, input_part_targets_len = concat_text_input_output(
            llm_inputs.input_ids,
            llm_inputs.attention_mask,
            llm_outputs.input_ids,
            llm_outputs.attention_mask,
        )

        mask = i2t_llm_inputs['input_ids'] == llm_tokenizer.pad_token_id
        mask = mask & (torch.cumsum(mask, dim=1) > 2)

        targets = i2t_llm_inputs['input_ids'].masked_fill(mask, -100)

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        return {
            "text_input_ids": text_inputs.input_ids,
            "text_attention_mask": text_inputs.attention_mask,
            "llm_input_ids": llm_inputs.input_ids,
            "llm_attention_mask": llm_inputs.attention_mask,
            "i2t_llm_input_ids": i2t_llm_inputs['input_ids'],
            "i2t_llm_attention_mask": i2t_llm_inputs['attention_mask'],
            "labels": targets,
        }

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size,
        num_workers=args.preprocessing_num_workers, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size,
        num_workers=args.preprocessing_num_workers, pin_memory=True
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("runs", experiment_config)

    # Metric
    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=["SPICE"])

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

        completed_steps = starting_epoch * num_update_steps_per_epoch
        progress_bar.update(completed_steps)

    best_score = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        if args.train_data_size != "0":
            model.train()
            if args.with_tracking:
                total_loss = 0
            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        progress_bar.update(1)
                        completed_steps += 1
                        continue

                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        outputs = model(
                            text_input_ids=batch['text_input_ids'],
                            i2t_llm_input_ids=batch['i2t_llm_input_ids'],
                            text_attention_mask=batch['text_attention_mask'],
                            i2t_llm_attention_mask=batch['i2t_llm_attention_mask'],
                            labels=batch['labels'],
                            bias=4,
                        )
                        loss = outputs.loss
                    # We keep track of the loss at each epoch
                    if args.with_tracking:
                        total_loss += loss.detach().float()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

                if completed_steps >= args.max_train_steps:
                    break

        model.eval()

        hyp_list = []
        ref_list = []
        losses = []
        a = len(eval_dataloader)
        for step, batch in enumerate(eval_dataloader):
            print(f'{step}/{a}', end='\r')
            with torch.no_grad():
                outputs = model(
                    text_input_ids=batch['text_input_ids'],
                    i2t_llm_input_ids=batch['i2t_llm_input_ids'],
                    text_attention_mask=batch['text_attention_mask'],
                    i2t_llm_attention_mask=batch['i2t_llm_attention_mask'],
                    labels=batch['labels'],
                    bias=4,
                )
                loss = outputs.loss
                labels = batch["labels"]
                losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

                generated_tokens = accelerator.unwrap_model(model).generate(
                    text_input_ids=batch['text_input_ids'],
                    llm_input_ids=batch['llm_input_ids'],
                    text_attention_mask=batch['text_attention_mask'],
                    llm_attention_mask=batch['llm_attention_mask'],
                    generation_config=generation_config,
                    bias=4,
                )
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=llm_tokenizer.pad_token_id
                )
                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=llm_tokenizer.pad_token_id)

                generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, llm_tokenizer.pad_token_id)

                decoded_preds = llm_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = llm_tokenizer.batch_decode(labels, skip_special_tokens=True)

                hyp_list.extend(decoded_preds)
                ref_list.extend(decoded_labels)

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        distinct_1, distinct_2 = compute_distinct([normalize_answer(s) for s in hyp_list])
        hyp_list, ref_list = postprocess_text(hyp_list, ref_list)
        result = nlgeval.compute_metrics([ref_list], hyp_list)
        result['distinct_1'] = distinct_1
        result['distinct_2'] = distinct_2
        result = {k: round(v * 100, 4) for k, v in result.items()}
        logger.info(result)
        result['perplexity'] = perplexity

        if args.with_tracking:
            # result["train_loss"] = total_loss.item() / len(train_dataloader)
            result["eval_loss"] = eval_loss.item()
            result["epoch"] = epoch
            result["step"] = completed_steps
            accelerator.log(result, step=completed_steps)

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            with open(os.path.join(output_dir, "results_hyp.txt"), "w") as f:
                f.writelines([p + '\n' for p in hyp_list])
            with open(os.path.join(output_dir, "results_ref.txt"), "w") as f:
                f.writelines([p + '\n' for p in ref_list])

        if result['Bleu_1'] > best_score:
            best_score = result['Bleu_1']
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                text_tokenizer.save_pretrained(os.path.join(args.output_dir, "text_tokenizer"))
                llm_tokenizer.save_pretrained(os.path.join(args.output_dir, "llm_tokenizer"))
                with open(os.path.join(args.output_dir, "results_hyp.txt"), "w") as f:
                    f.writelines([p + '\n' for p in hyp_list])
                with open(os.path.join(args.output_dir, "results_ref.txt"), "w") as f:
                    f.writelines([p + '\n' for p in ref_list])
                with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                    json.dump(result, f)


if __name__ == "__main__":
    main()
