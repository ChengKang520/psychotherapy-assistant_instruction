
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaForCausalLM,
    LlamaTokenizer,
    set_seed,
)


def perplexity(sentences, tokenizer, model, device='cuda'):
    # calculate perplexity
    with torch.no_grad():
        ppl = []
        sos_token = tokenizer.decode([0])
        for sentence in tqdm(sentences, total=len(sentences)):
            full_tensor_input = tokenizer.encode(sos_token + sentence.replace(EOT_TOKEN, ' ').strip(), return_tensors='pt').to(device)
            full_loss = model(full_tensor_input, labels=full_tensor_input)[0].mean()
            ppl.append(torch.exp(full_loss).flatten().cpu().item())
    return np.mean(ppl)/len(sentences), np.std(ppl)/len(sentences)


def grammaticality(sentences, tokenizer, model, device='cuda'):
    with torch.no_grad():
        total_good = 0
        for sent in tqdm(sentences, total=len(sentences)):
            good_prob = F.softmax(model(tokenizer.encode(sent, return_tensors='pt').to(device))[0].flatten(), dim=0)[1]
            total_good += good_prob
        return total_good / len(sentences) # avg probability of grammaticality according to model



def make_inference(instruction, context = None):
    if context:
        prompt = f"Below is an instruction that describes a task, paired with an input that provides further context.\n\n### Instruction: \n{instruction}\n\n### Input: \n{context}\n\n### Response: \n"
    else:
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction: \n{instruction}\n\n### Response: \n"
        inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to("cuda:0")
        outputs = base_model.generate(**inputs, max_new_tokens=100)
        print((tokenizer.decode(outputs[0], skip_special_tokens=True)))
        outputs = model.generate(**inputs, max_new_tokens=50)
        print("---- NON-INSTRUCT-TUNED-MODEL ----")
        print((tokenizer.decode(outputs[0], skip_special_tokens=True)))



    make_inference("Convert the text into a dialogue between two characters.", "Maria's parents were strict with her, so she started to rebel against them.")
    make_inference("Explain in simple terms how the attention mechanism of a transformer model works")
    make_inference("Identify the odd one out and explain your choice.", "Orange, Green, Airplane.")







