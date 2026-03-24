import os
import torch
from functools import partial
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_dataset
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from accelerate import Accelerator

# ── your constants ────────────────────────────────────────────────
CHECKPOINT = "microsoft/speecht5_tts"
REPO_NAME  = "speecht5_finetuned_hindi_devanagari_01"

# ── processor & model ─────────────────────────────────────────────
processor = SpeechT5Processor.from_pretrained(CHECKPOINT)
model     = SpeechT5ForTextToSpeech.from_pretrained(CHECKPOINT)
model.config.use_cache = False
model.generate = partial(model.generate, use_cache=True)

# ── collator ──────────────────────────────────────────────────────
@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        input_ids        = [{"input_ids": f["input_ids"]} for f in features]
        label_features   = [{"input_values": f["labels"]} for f in features]
        speaker_features = [f["speaker_embeddings"] for f in features]

        batch = self.processor.pad(
            input_ids=input_ids,
            labels=label_features,
            return_tensors="pt",
        )

        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )
        del batch["decoder_attention_mask"]

        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor(
                [len(f["input_values"]) for f in label_features]
            )
            target_lengths = target_lengths.new(
                [l - l % model.config.reduction_factor for l in target_lengths]
            )
            max_len = max(target_lengths)
            max_len = max_len - (max_len % model.config.reduction_factor)
            batch["labels"] = batch["labels"][:, :max_len]

        batch["speaker_embeddings"] = torch.tensor(speaker_features)
        return batch

# ── dataset (replace with your own loading logic) ─────────────────
dataset = load_dataset(...)   # ← your dataset here

# ── training args ─────────────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir=REPO_NAME,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    fp16=True,

    learning_rate=1e-4,
    warmup_steps=500,
    max_steps=1500,

    eval_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=50,

    load_best_model_at_end=True,
    greater_is_better=False,

    dataloader_num_workers=2,
    ddp_find_unused_parameters=False,   # ← important for SpeechT5

    label_names=["labels"],
    report_to=["tensorboard"],
    push_to_hub=False,
)

# ── trainer ───────────────────────────────────────────────────────
data_collator = TTSDataCollatorWithPadding(processor=processor)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    processing_class=processor,
)

trainer.train()
