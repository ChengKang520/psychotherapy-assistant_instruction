import sys

import click
import torch
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
)
from utils import generate_prompt
from collections import namedtuple
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


def decide_model(finetuned_flag, args, device_map, finetuned_weights):
    ModelClass = namedtuple("ModelClass", ('tokenizer', 'model'))
    _MODEL_CLASSES = {
        "llama": ModelClass(**{
            "tokenizer": LlamaTokenizer,
            "model": LlamaForCausalLM,
        }),
        "chatglm": ModelClass(**{
            "tokenizer": AutoTokenizer,  # ChatGLMTokenizer,
            "model": AutoModel,  # ChatGLMForConditionalGeneration,
        }),
        "bloom": ModelClass(**{
            "tokenizer": BloomTokenizerFast,
            "model": BloomForCausalLM,
        }),
        "Auto": ModelClass(**{
            "tokenizer": AutoTokenizer,
            "model": AutoModelForCausalLM,
        })
    }
    model_type = "Auto" if args.model_type not in ["llama", "bloom", "chatglm"] else args.model_type

    if model_type == "chatglm":
        tokenizer = _MODEL_CLASSES[model_type].tokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=True
        )
        # todo: ChatGLMForConditionalGeneration revision
        model = _MODEL_CLASSES[model_type].model.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            device_map=device_map
        )
    else:
        tokenizer = _MODEL_CLASSES[model_type].tokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        model = _MODEL_CLASSES[model_type].model.from_pretrained(
            args.base_model,
            load_in_8bit=True,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=device_map
        )

    if model_type == "llama":
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"  # Allow batched inference
    elif (model_type == "mpt") or (model_type == "falcon") or (model_type == "wizard"):
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"  # Allow batched inference


    if finetuned_flag == True:
        if device_map == "auto":
            model = PeftModel.from_pretrained(
                model,
                finetuned_weights,
                torch_dtype=torch.float16,
            )
            print("********************************************")
            print('PEFT Auto')
            print("********************************************")
        else:
            model = PeftModel.from_pretrained(
                model,
                finetuned_weights,
                device_map=device_map
            )
            print("********************************************")
            print('PEFT Device_Map')
            print("********************************************")
    else:
        print("********************************************")
        print('ORIGINAL Device_Map')
        print("********************************************")

    return tokenizer, model


class ModelServe1:
    def __init__(
            self,
            load_8bit: bool = True,
            model_type: str = "falcon",
            base_model: str = "/home/kangchen/Chatbot/Psych_BioGPT/models/input/falcon/falcon-7b-instruct/",
            finetuned_weights: str = "/home/kangchen/Chatbot/chatbot-chinese/finetuned/falcon-7b-instruct_psychtherapy_10Epochs/checkpoint-1200/",
    ):

        # /home/kangchen/Chatbot/chatbot-chinese/finetuned/vicuna-7b-delta-v1.1_alpaca-en-zh/
        # /home/kangchen/Chatbot/chatbot-chinese/finetuned/falcon-7b-instruct_psychtherapy_all/checkpoint-800/
        args = locals()
        namedtupler = namedtuple("args", tuple(list(args.keys())))
        local_args = namedtupler(**args)

        if torch.cuda.is_available():
            self.device = "cuda"
            self.device_map = "auto"
            # self.max_memory = {i: "15GIB" for i in range(torch.cuda.device_count())}
            # self.max_memory.update({"cpu": "30GB"})
        else:
            self.device = "cpu"
            self.device_map = {"": self.device}

        self.tokenizer, self.model = decide_model(finetuned_flag=False, args=local_args, device_map=self.device_map,
                                                  finetuned_weights=finetuned_weights)

        # unwind broken decapoda-research config
        # self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        # if not load_8bit:
        #     self.model.half()  # seems to fix bugs for some users.

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

    def generate(
            self,
            instruction: str,
            input: str,
            temperature: float = 0.7,
            top_p: float = 0.75,
            top_k: int = 40,
            num_beams: int = 4,
            max_new_tokens: int = 1024,
            **kwargs
    ):
        prompt = generate_prompt(instruction, input)
        print(f"Prompt: {prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        print("generating...")
        with torch.no_grad():
            generation_output = self.model.generate(
                inputs=input_ids,
                temperature=0.7,
                pad_token_id=50256,
                top_p=0.8,
                repetition_penalty=1.02,
                max_new_tokens=1024
            )

            #  Can you give me some suggestions about how to fight against depression

            # generation_output = self.model.generate(
            #     input_ids=input_ids,
            #     generation_config=generation_config,
            #     return_dict_in_generate=True,
            #     output_scores=True,
            #     max_new_tokens=max_new_tokens,
            # )

            # generation_output = self.model.generate(inputs=input_ids, max_length=128, pad_token_id=50256,
            #                                         num_return_sequences=1)
        s = generation_output[0]
        output = self.tokenizer.decode(s)
        # print(f"Output: {output}")
        # return output

        print("********************************************")
        print(output.split("Response：")[1].split("<|endoftext|>")[0])
        print("********************************************")
        # Response_end = output.split("Response：")[1].split("<|endoftext|>")[0]
        Response_end = output.split("Response：")[1]
        return Response_end


class ModelServe2:
    def __init__(
            self,
            load_8bit: bool = True,
            model_type: str = "falcon",
            base_model: str = "/home/kangchen/Chatbot/Psych_BioGPT/models/input/falcon/falcon-7b-instruct/",
            finetuned_weights: str = "/home/kangchen/Chatbot/chatbot-chinese/finetuned/falcon-7b-instruct_psychtherapy_10Epochs/checkpoint-1200/",
    ):

        # /home/kangchen/Chatbot/chatbot-chinese/finetuned/vicuna-7b-delta-v1.1_alpaca-en-zh/
        # /home/kangchen/Chatbot/chatbot-chinese/finetuned/falcon-7b-instruct_psychtherapy_all/checkpoint-800/
        args = locals()
        namedtupler = namedtuple("args", tuple(list(args.keys())))
        local_args = namedtupler(**args)

        if torch.cuda.is_available():
            self.device = "cuda"
            self.device_map = "auto"
            # self.max_memory = {i: "15GIB" for i in range(torch.cuda.device_count())}
            # self.max_memory.update({"cpu": "30GB"})
        else:
            self.device = "cpu"
            self.device_map = {"": self.device}

        self.tokenizer, self.model = decide_model(finetuned_flag=True, args=local_args, device_map=self.device_map,
                                                  finetuned_weights=finetuned_weights)

        # unwind broken decapoda-research config
        # self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        # if not load_8bit:
        #     self.model.half()  # seems to fix bugs for some users.

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

    def generate(
            self,
            instruction: str,
            input: str,
            temperature: float = 0.7,
            top_p: float = 0.75,
            top_k: int = 40,
            num_beams: int = 4,
            max_new_tokens: int = 1024,
            **kwargs
    ):
        prompt = generate_prompt(instruction, input)
        print(f"Prompt: {prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        print("generating...")
        with torch.no_grad():
            generation_output = self.model.generate(
                inputs=input_ids,
                temperature=0.7,
                pad_token_id=50256,
                top_p=0.8,
                repetition_penalty=1.02,
                max_new_tokens=1024
            )

            #  Can you give me some suggestions about how to fight against depression

            # generation_output = self.model.generate(
            #     input_ids=input_ids,
            #     generation_config=generation_config,
            #     return_dict_in_generate=True,
            #     output_scores=True,
            #     max_new_tokens=max_new_tokens,
            # )

            # generation_output = self.model.generate(inputs=input_ids, max_length=128, pad_token_id=50256,
            #                                         num_return_sequences=1)
        s = generation_output[0]
        output = self.tokenizer.decode(s)
        # print(f"Output: {output}")
        # return output

        print("********************************************")
        print(output.split("Response：")[1].split("<|endoftext|>")[0])
        print("********************************************")
        # Response_end = output.split("Response：")[1].split("<|endoftext|>")[0]
        Response_end = output.split("Response：")[1]
        return Response_end
