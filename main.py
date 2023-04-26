import logging
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union
import datasets
import numpy as np
import torch
from packaging import version
from torch import nn
import librosa
from lang_trans import arabic
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    is_apex_available,
    trainer_utils,
)
from model import Wav2Vec2ForWCTC
import random
from classes import *

random.seed(42)
torch.cuda.set_device(6)



class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_cuda_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            if model.module.config.ctc_loss_reduction == "mean":
                loss = loss.mean()
            elif model.module.config.ctc_loss_reduction == "sum":
                loss = loss.sum() / (inputs["labels"] >= 0).sum()
            else:
                raise ValueError(f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_cuda_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    configure_logger(model_args, training_args)

    orthography = Orthography.from_name(data_args.orthography.lower())
    processor = orthography.create_processor(model_args)
    model = Wav2Vec2ForWCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        gradient_checkpointing=training_args.gradient_checkpointing,
        vocab_size=len(processor.tokenizer),
        wctc=model_args.wctc,
    )

    train_dataset = datasets.load_dataset(
        data_args.dataset_name, data_args.dataset_config_name, split=data_args.train_split_name, data_dir = "./timit/"
    )
    train_dataset = train_dataset.filter(lambda example, indice: np.random.rand() < 0.1, with_indices=True)

    val_dataset = datasets.load_dataset(
        data_args.dataset_name, data_args.dataset_config_name, split=data_args.validation_split_name, data_dir = "./timit/"
    )
    val_dataset = val_dataset.filter(lambda example, indice: np.random.rand() < 0.1, with_indices=True)


    wer_metric = datasets.load_metric("wer")
    target_sr = processor.feature_extractor.sampling_rate if data_args.target_feature_extractor_sampling_rate else None
    vocabulary_chars_str = "".join(t for t in processor.tokenizer.get_vocab().keys() if len(t) == 1)
    vocabulary_text_cleaner = re.compile(  # remove characters not in vocabulary
        f"[^\s{re.escape(vocabulary_chars_str)}]",  # allow space in addition to chars in vocabulary
        flags=re.IGNORECASE if processor.tokenizer.do_lower_case else 0,
    )
    text_updates = []

    def prepare_example(example, mask_ratio=0.0):  
        example["speech"], example["sampling_rate"] = librosa.load(example[data_args.speech_file_column], sr=target_sr)
        if data_args.max_duration_in_seconds is not None:
            example["duration_in_seconds"] = len(example["speech"]) / example["sampling_rate"]
        # Normalize and clean up text; order matters!
        updated_text = orthography.preprocess_for_training(example[data_args.target_text_column])
        updated_text = vocabulary_text_cleaner.sub("", updated_text)
        if updated_text != example[data_args.target_text_column]:
            text_updates.append((example[data_args.target_text_column], updated_text))
            example[data_args.target_text_column] = updated_text

        # randomly mask labels based on mask_ratio
        if mask_ratio > 0:
            text_len = len(example[data_args.target_text_column])
            keep_len = round(text_len * (1-mask_ratio))
            start = random.randint(0, text_len-keep_len)
            example[data_args.target_text_column] = example[data_args.target_text_column][start:start+keep_len]

        return example

    train_dataset = train_dataset.map(prepare_example, fn_kwargs={"mask_ratio": data_args.mask_ratio}, keep_in_memory=True, load_from_cache_file=False, remove_columns=[data_args.speech_file_column])
    val_dataset = val_dataset.map(prepare_example, remove_columns=[data_args.speech_file_column])

    text_updates = None

    def prepare_dataset(batch):
        # check that all files have the correct sampling rate
        assert (
            len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

        batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
        with processor.as_target_processor():
            batch["labels"] = processor(batch[data_args.target_text_column]).input_ids
        return batch

    train_dataset = train_dataset.map(
        prepare_dataset,
        batch_size=training_args.per_device_train_batch_size,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        keep_in_memory=True,
        load_from_cache_file=False
    )
    val_dataset = val_dataset.map(
        prepare_dataset,
        batch_size=training_args.per_device_train_batch_size,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        if logger.isEnabledFor(logging.DEBUG):
            for reference, predicted in zip(label_str, pred_str):
                logger.debug(f'reference: "{reference}"')
                logger.debug(f'predicted: "{predicted}"')

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    if model_args.freeze_feature_extractor:
        model.freeze_feature_extractor()

    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()
    
    eval_results = trainer.evaluate()
    trainer.save_model("./ASR_hacker_model")

if __name__ == "__main__":
    main()
    