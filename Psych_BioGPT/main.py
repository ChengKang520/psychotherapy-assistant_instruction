


#
# from peft import get_peft_model
# import torch
# import transformers
# from peft import LoraConfig
# from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
# from transformers import AutoTokenizer
# import torch
# import transformers
# from peft import LoraConfig
# from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
# from trl import SFTTrainer
# from huggingface_hub import notebook_login
# import matplotlib.pyplot as plt
# from datasets import load_dataset
#
#
#
# def plot_sequence_lengths(dataset_obj):
#
#     # Initialize a list to store the sequence lengths
#     sequence_lengths = []
#
#     # list of indices that are too long
#     too_long = []
#
#     # Loop over the dataset and get the lengths of text sequences
#     for idx, example in enumerate(dataset_obj["train"]):
#         sequence_lengths.append(len(example['instruction']) + len(example["context"]) + len(example["response"]))
#         if sequence_lengths[idx] > 2200:
#           too_long.append(idx)
#
#     # Plot the histogram
#     plt.hist(sequence_lengths, bins=30)
#     plt.xlabel('Sequence Length')
#     plt.ylabel('Count')
#     plt.title('Distribution of Text Sequence Lengths')
#     plt.show()
#
#     return too_long
#
# def formatting_func(example):
#     if example.get("context", "") != "":
#         input_prompt = (
#             f"Below is an instruction that describes a task, paired with an input that provides further context. "
#             "Write a response that appropriately completes the request.\n\n"
#             "### Instruction:\n"
#             f"{example['instruction']}\n\n"
#             f"### Input: \n"
#             f"{example['context']}\n\n"
#             f"### Response: \n"
#             f"{example['response']}")
#
#     else:
#         input_prompt = (f"Below is an instruction that describes a task. "
#                         "Write a response that appropriately completes the request.\n\n"
#                         "### Instruction:\n"
#                         f"{example['instruction']}\n\n"
#                         f"### Response:\n"
#                         f"{example['response']}")
#
#     return {"text": input_prompt}
#
#
#
# dbricks_15k_dataset_base = load_dataset("databricks/databricks-dolly-15k")
# indexes_to_drop = plot_sequence_lengths(dbricks_15k_dataset_base)
# len(indexes_to_drop)
#
#
# dbricks_15k_dataset_reduced = dbricks_15k_dataset_base["train"].select(
#     i for i in range(len(dbricks_15k_dataset_base["train"])) if i not in set(indexes_to_drop)
# )
# print(dbricks_15k_dataset_reduced)
#
#
# dbricks_15k_dataset_prepared = dbricks_15k_dataset_reduced.train_test_split(test_size=0.1)
# formatted_dataset = dbricks_15k_dataset_prepared.map(formatting_func)
# print(formatted_dataset["train"][2]["text"])
#





# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template



# Need to call this before importing transformers.
from fastchat.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)
replace_llama_attn_with_flash_attn()


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

            cur_len += round_len
        target[cur_len:] = IGNORE_TOKEN_ID

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")
    raw_data = json.load(open(data_args.data_path, "r"))

    # Split train/test
    np.random.seed(0)
    perm = np.random.permutation(len(raw_data))
    split = int(len(perm) * 0.98)
    train_indices = perm[:split]
    eval_indices = perm[split:]
    train_raw_data = [raw_data[i] for i in train_indices]
    eval_raw_data = [raw_data[i] for i in eval_indices]
    rank0_print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")

    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer)
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    print('*********************** Model Loading! ***********************')
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )


    print('*********************** Model Loaded Successfully! ***********************')
    model.config.use_cache = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    print('*********************** Data Pre-processed Successfully! ***********************')


    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()




#
#
#
#
# model_id = "openlm-research/open_llama_7b_700bt_preview"
#
# qlora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM"
# )
#
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )
#
# base_model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     quantization_config=bnb_config,
# )
#
#
# from transformers import LlamaTokenizerFast
#
# tokenizer = LlamaTokenizerFast.from_pretrained(model_id)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#
# print(base_model)
#
#
#
#
# supervised_finetuning_trainer = SFTTrainer(
#     base_model,
#     train_dataset=formatted_dataset["train"],
#     eval_dataset=formatted_dataset["test"],
#     args=transformers.TrainingArguments(
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=4,
#         learning_rate=2e-4,
#         max_steps=5000,
#         output_dir="./SFTOpenLM-Dolly15k",
#         optim="paged_adamw_8bit",
#         fp16=True,
#     ),
#     tokenizer=tokenizer,
#     peft_config=qlora_config,
#     dataset_text_field="text",
#     max_seq_length=512
# )
#
#
# supervised_finetuning_trainer.train()
#
# notebook_login()
#
#
# # base_model.push_to_hub("FourthBrainGenAI/FB-DLAI-Instruct-tune-v3", private=True)
# # tokenizer.push_to_hub("FourthBrainGenAI/FB-DLAI-Instruct-tune-v3")
# # lora_config = LoraConfig.from_pretrained("FourthBrainGenAI/FB-DLAI-Instruct-tune-v3")
# # bnb_config = BitsAndBytesConfig(
# #     load_in_4bit=True,
# #     bnb_4bit_use_double_quant=True,
# #     bnb_4bit_quant_type="nf4",
# #     bnb_4bit_compute_dtype=torch.bfloat16
# # )
# # tokenizer = AutoTokenizer.from_pretrained("FourthBrainGenAI/FB-DLAI-Instruct-tune-v3")
# # model = AutoModelForCausalLM.from_pretrained(
# #     lora_config.base_model_name_or_path,
# #     quantization_config=bnb_config,
# #     device_map={"":0})
# # model = get_peft_model(model, lora_config)
#
#
#
# def make_inference(instruction, context = None):
#   if context:
#     prompt = f"Below is an instruction that describes a task, paired with an input that provides further context.\n\n### Instruction: \n{instruction}\n\n### Input: \n{context}\n\n### Response: \n"
#   else:
#     prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction: \n{instruction}\n\n### Response: \n"
#   inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to("cuda:0")
#   outputs = base_model.generate(**inputs, max_new_tokens=100)
#   print((tokenizer.decode(outputs[0], skip_special_tokens=True)))
#   outputs = model.generate(**inputs, max_new_tokens=50)
#   print("---- NON-INSTRUCT-TUNED-MODEL ----")
#   print((tokenizer.decode(outputs[0], skip_special_tokens=True)))
#
#
#
# make_inference("Convert the text into a dialogue between two characters.", "Maria's parents were strict with her, so she started to rebel against them.")
# make_inference("Explain in simple terms how the attention mechanism of a transformer model works")
# make_inference("Identify the odd one out and explain your choice.", "Orange, Green, Airplane.")
#
#
#
#














