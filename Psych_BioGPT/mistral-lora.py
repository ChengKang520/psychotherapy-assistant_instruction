

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model
import transformers
from datetime import datetime
import os
from peft import prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import argparse
import numpy as np
import evaluate
from transformers import set_seed
import torch.distributed as dist


fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)



def check_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = local_rank = world_size = -1
    return rank, local_rank, world_size


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


def create_prompt(context, question, answer):
    if len(answer["text"]) < 1:
        answer = "Cannot Find Answer"
    else:
        answer = answer["text"][0]
    prompt_template = f"### CONTEXT\n{context}\n\n### QUESTION\n{question}\n\n### ANSWER\n{answer}</s>"
    return prompt_template


def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
    This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
    The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']

    ### Target sentence:
    {data_point["target"]}

    ### Meaning representation:
    {data_point["meaning_representation"]}
    """

    return tokenize(full_prompt)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class EvalPrediction:
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*):
    """

    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs

    def __iter__(self):
        if self.inputs is not None:
            return iter((self.predictions, self.label_ids, self.inputs))
        else:
            return iter((self.predictions, self.label_ids))

    def __getitem__(self, idx):
        if idx < 0 or idx > 2:
            raise IndexError("tuple index out of range")
        if idx == 2 and self.inputs is None:
            raise IndexError("tuple index out of range")
        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.label_ids
        elif idx == 2:
            return self.inputs


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--FilePath", type=str, default='CTV_data/CTV_data1/')
    parser.add_argument("--base_model_id", type=str, required=True, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument("--data_id", type=str, required=True, default='squad_v1')
    args = parser.parse_args()



    # setting torch distributed
    set_seed(888)
    rank, local_rank, world_size = check_distributed()
    is_main_process = local_rank in [-1, 0]
    is_distributed = world_size != -1
    print(rank, local_rank, world_size, is_distributed)

    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl")
        device_map = {"": int(local_rank or 0)}
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_map = "auto"

    # base_model_id = "mistralai/Mistral-7B-v0.1"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True, quantization_config=bnb_config)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        trust_remote_code=True,
        load_in_8bit=True,
        device_map=device_map
    )
    tokenizer.pad_token = tokenizer.eos_token

    if args.data_id == 'squad_v1':
        qa_dataset = load_dataset("squad")
        mapped_qa_dataset = qa_dataset.map(
            lambda samples: tokenizer(create_prompt(samples['context'], samples['question'], samples['answers'])))
        mapped_train_dataset = mapped_qa_dataset['train']
        mapped_validation_dataset = mapped_qa_dataset['validation']

        metric = evaluate.load("squad")

    elif args.data_id == 'squad_v2':
        qa_dataset = load_dataset("squad_v2")
        mapped_qa_dataset = qa_dataset.map(
            lambda samples: tokenizer(create_prompt(samples['context'], samples['question'], samples['answers'])))
        mapped_train_dataset = mapped_qa_dataset['train']
        mapped_validation_dataset = mapped_qa_dataset['validation']

        metric = evaluate.load("squad_v2")

    elif args.data_id == 'viggo':
        train_dataset = load_dataset('gem/viggo', split='train')
        eval_dataset = load_dataset('gem/viggo', split='validation')
        test_dataset = load_dataset('gem/viggo', split='test')
        mapped_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
        mapped_validation_dataset = eval_dataset.map(generate_and_tokenize_prompt)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    # Apply the accelerator. You can comment this out to remove the accelerator.
    model = accelerator.prepare_model(model)

    if torch.cuda.device_count() > 1: # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True
    #%%
    project = "viggo-finetune"
    base_model_name = "mistral"
    run_name = base_model_name + "-" + project
    output_dir = "./" + run_name

    tokenizer.pad_token = tokenizer.eos_token



    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=mapped_train_dataset,
        eval_dataset=mapped_validation_dataset,
        compute_metrics=compute_metrics,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            optim="paged_adamw_8bit",
            save_steps=25,
            logging_steps=25,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=False,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant",
            do_eval=True,
            evaluation_strategy="steps",
            eval_steps=10,

            # warmup_steps=5,
            # per_device_train_batch_size=4,
            # gradient_accumulation_steps=1,
            # num_train_epochs=5,
            # # evaluation_strategy='epoch',
            # learning_rate=2.5e-5, # Want about 10x smaller than the Mistral learning rate
            # logging_steps=10,
            # fp16=False,
            # bf16=False,
            # optim="paged_adamw_8bit",
            # logging_dir="./logs",        # Directory for storing logs
            # save_strategy="steps",       # Save the model checkpoint every logging step
            # save_steps=10,                # Save checkpoints every 50 steps
            # evaluation_strategy="steps", # Evaluate the model every logging step
            # eval_steps=10,               # Evaluate and save checkpoints every 50 steps
            # do_eval=True,                # Perform evaluation at the end of training
            # report_to="wandb",           # Comment this out if you don't want to use weights & baises
            # run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()


