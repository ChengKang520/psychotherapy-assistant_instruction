# # imports
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
from typing import List, Dict
import math
import time
import json
import os
import argparse


def formatting_func(example):
    if example.get("context", "") != "":
        input_prompt = (
            f"Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{example['instruction']}\n\n"
            f"### Input: \n"
            f"{example['input']}\n\n"
            f"### Output: \n"
            f"{example['output']}")

    else:
        input_prompt = (f"Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\n"
                        f"{example['instruction']}\n\n"
                        f"### Output:\n"
                        f"{example['output']}")

    return {"text": input_prompt}


if __name__ == "__main__":


    json_path = 'CTV_json/'
    json_chatgpt_path = 'CTV_json_chatgpt/'
    file_path = ''

    print('Hello')

    files = os.listdir(json_chatgpt_path)
    instructions_train = []
    for file in files:

        conversation_sample = [json.loads(line) for line in open(os.path.join(json_chatgpt_path, file), 'r', encoding='utf-8')][0]
        for line_i in range(len(conversation_sample)):
            instructions_train.append(conversation_sample[line_i])

    with open(os.path.join(file_path, 'psychtherapy_data.json'), "a") as outfile:
        json.dump(instructions_train, outfile)
