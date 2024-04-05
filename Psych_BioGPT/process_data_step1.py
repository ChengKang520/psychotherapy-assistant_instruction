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
import copy


TASK_DATA = ["concept_explanation", "question_answering", "mental_status_assessment", "psychological_counseling", "information_extraction", "dialogue_generation", "sentiment_analysis", "event_ordering"]

TASK_INST = {
            "concept_explanation": "Can the following input be regarded as a concept explanation task with finite output ",
            "question_answering": "Can the following input be regarded as a question answering task with finite output ",
            "mental_status_assessment": "Can the following input be regarded as a mental status assessment task with finite ",
            "psychological_counseling": "Can the following input be regarded as a psychological counseling task with finite ",
            "information_extraction": "Can the following input be regarded as a information extraction task with finite ",
            "dialogue_generation": "Can the following input be regarded as a concept explanation task with finite ",
            "sentiment_analysis": "Can the following input be regarded as a concept explanation task with finite ",
            "event_ordering": "Can the following input be regarded as a concept explanation task with finite ",
    }


PROMPT_DICT = {

    "rewrite_prompt": (
        "Make a more professional instruction and output based on given context of conversation in the domain. Remove people’s names and UNKNOWN. Then, improve them all based on your knowledge. If you cannot do that, output nothing."
    ),
    "success_prompt": (
        "Given an instruction and an output in the domain, rate whether the response appears to be a helpful and "
        "informative answer to the query, from 1 (lowest) - 5 (highest). The detailed criterion is as follows: "
        "5: The response provides a complete, highly detailed, and informative response to the query, fully satisfying the information needs. "
        "4: The response mostly fulfills the need in the query, while there can be some minor improvements such as discussing more detailed information, having better structure of the response, or improving coherence. "
        "3: The response is acceptable, but some major additions or improvements are needed to satisfy users’ needs. "
        "2: The response still addresses the main request, but it is not complete or not relevant to the query. "
        "1: The response is barely on-topic or completely irrelevant."
    ),
    "ground_instruction": (
        "You will be given an task instruction, evidence, and output. Your objective is to assess the extent to which the output is supported by the information presented in the evidence.\n"
        "Rate the level of support on a scale from 1 ( Ignore / Contradictory), 2 (Little support), 3 (Partially supported), 4 (Mostly supported), 5 (Fully supported)."
    ),
    "concept_explanation": (
        "Summarize the bellow description and explain the below concept on the psychotherapy domain. Keep the knowledge of original input. Add more knowledge according to your understanding."
    ),
    "question_answering": (
        "Answer the bellow question on the psychotherapy domain. Keep the knowledge of original input. Add more knowledge according to your understanding."
    ),
    "mental_status_assessment": (
        "Evaluate the mental status according to the following content. Keep the knowledge of original input. Add more knowledge according to your understanding."
    ),
    "psychological_counseling": (
        "Rewrite the psychological counseling content on the psychotherapy domain. Keep the knowledge of original input. Add more knowledge according to your understanding."
    ),
    "information_extraction": (
        "Extract the main information. Keep the knowledge of original input. Add more knowledge according to your understanding."
    ),
    "dialogue_generation": (
        "Generate the dialogue on the psychotherapy domain. Keep the knowledge of original input. Add more knowledge according to your understanding."
    ),
    "sentiment_analysis": (
        "Analyze the sentiment result according to the input content on the psychotherapy domain. Keep the knowledge of original input. Add more knowledge according to your understanding."
    ),
    "event_ordering": (
        "Extract the address and time of the event ordering task on the psychotherapy domain."
    ),
}


def process_data(conversation_sample, topic, save_sample):
    start_flag = False
    content_flag = False
    topic_fuse_flag = False
    conversation_temp = ''
    instructions_temp = ''
    suggestions_temp = ''
    time_bar = []
    i = 0

    instructions = {
        "instruction": "",
        "input": "",
        "output": "",
        "task": "",
        "domain": "",
    }

    # Instructions_train = instructions.copy()
    # Instructions_chatgpt_train = instructions.copy()

    Instructions_train = []
    Instructions_chatgpt_train = []
    while (i < len(conversation_sample)):

        # if i >= 120:
        #     break

        if (conversation_sample[i] == 'BEGIN TRANSCRIPT: ') & (start_flag == False):
            start_flag = True
            i += 1
            continue

        elif (len(conversation_sample[i]) == 8) & (str(conversation_sample[i]) != str('00:00:00')) & (
                content_flag == False):
            content_flag = True
            if (time_bar == conversation_sample[i]):
                topic_fuse_flag = True
            #         elif (time_bar != conversation_sample[i]):
            #             topic_fuse_flag = False
            time_bar = conversation_sample[i]
            #         print('==============' + time_bar + '===============')
            i += 1
            continue

        if (start_flag == True) & (content_flag == True) & (len(conversation_sample[i]) >= 200):
            suggestions_temp = conversation_sample[i]

            if len(suggestions_temp) >= 1000:
                suggestions_temp_seg = []
                seg_num = math.ceil(len(suggestions_temp) / 1000)
                suggestions_temp1 = suggestions_temp.split('.')
                seg_length = math.ceil(len(suggestions_temp1) / seg_num)
                for i_seg1 in range(seg_num):
                    if i_seg1 != (seg_num - 1):
                        suggestions_temp_seg.append(
                            ' '.join(suggestions_temp1[(0 + i_seg1 * seg_length): (i_seg1 + 1) * seg_length]))
                    else:
                        suggestions_temp_seg.append(
                            ' '.join(suggestions_temp1[(0 + i_seg1 * seg_length):]))

                for i_segment in range(len(suggestions_temp_seg)):

                    # ***********************************************
                    for i_task in range(len(TASK_INST)):
#######################################################################################################################
                        prompts = TASK_INST[TASK_DATA[i_task]] + "on " + topic + " domian ? " + "The answer is YES or NO."

                        # *********************************************** ChatGPT Instructions
                        query = f"""
                                Rewrite the following content according to the topic of {topic}. Remove people's name and UNKNOWN.
                                Below is an instruction that describes a task, paired with an input that provides further context.
                                Write a response that appropriately completes the request.\n\n
                                
                                   Prompt:
                                   \"\"\"
                                   {prompts}
                                   \"\"
    
                                   Input:
                                   \"\"\"
                                   {instructions_temp}
                                   \"\"\"
    
                                   Output:
                                   \"\"\"
                                   {suggestions_temp_seg[i_segment]}
                                   \"\"\"
                                   """
                        # print(query)
                        try:
                            response = openai.ChatCompletion.create(
                                engine="ft-gpt4-psytherapy",  # engine = "deployment_name".
                                messages=[
                                    {'role': 'system', 'content': 'Formate the below content.'},
                                    {'role': 'user', 'content': query},
                                ],
                            )
                            # print(response['choices'][0]['message']['content'])
                            req_response_num = 0
                            content_length = len(response['choices'][0]['message']['content'].split('\n\n')[0])

                            while (len(response['choices'][0]['message']['content'].split('\n\n')) < 3) or (
                                    "Instruction:" not in response['choices'][0]['message']['content']):
                                response = openai.ChatCompletion.create(
                                    engine="ft-gpt4-psytherapy",  # engine = "deployment_name".
                                    messages=[
                                        {'role': 'system', 'content': 'You formate the below content again.'},
                                        {'role': 'user', 'content': query},
                                    ],
                                )

                                time.sleep(1)
                                req_response_num += 1
                                print('%%%%%%% Unsuccessful Time ' + str(req_response_num) + ' %%%%%%')
                                content_length = len(response['choices'][0]['message']['content'].split('\n\n')[0])
                                if (content_length > 20) and (req_response_num >= 10):
                                    break

                            instructions_task = response['choices'][0]['message']['content'].split('\n\n')

                            if "YES" in instructions_task:
#######################################################################################################################

                                # ***********************************************
                                instructions[
                                    'instruction'] = PROMPT_DICT[TASK_INST[i_task]]
                                instructions['input'] = instructions_temp
                                instructions['output'] = suggestions_temp_seg[i_segment]
                                instructions['domain'] = topic
                                instructions['task'] = TASK_INST[i_task]

                                # *********************************************** ChatGPT Instructions
                                query = f"""
                                    {PROMPT_DICT["rewrite_prompt"]}

                                    Instruction:
                                    \"\"\"
                                    {instructions['instruction']}
                                    \"\"
                                    
                                    Input:
                                    \"\"\"
                                    {instructions['input']}
                                    \"\"\"
                                    
                                    Output:
                                    \"\"\"
                                    {instructions['output']}
                                    \"\"\"
                                    
                                    Domain:
                                    \"\"\"
                                    instructions['domain']
                                    \"\"\"
                                    
                                    """
                                # print(query)
                                try:
                                    response = openai.ChatCompletion.create(
                                        engine="ft-gpt4-psytherapy",  # engine = "deployment_name".
                                        messages=[
                                            {'role': 'system', 'content': 'Formate the below content.'},
                                            {'role': 'user', 'content': query},
                                        ],
                                    )
                                    # print(response['choices'][0]['message']['content'])
                                    req_response_num = 0
                                    content_length = len(response['choices'][0]['message']['content'].split('\n\n')[0])

                                    while (len(response['choices'][0]['message']['content'].split('\n\n')) < 3) or (
                                            "Instruction:" not in response['choices'][0]['message']['content']):
                                        response = openai.ChatCompletion.create(
                                            engine="ft-gpt4-psytherapy",  # engine = "deployment_name".
                                            messages=[
                                                {'role': 'system', 'content': 'You formate the below content again.'},
                                                {'role': 'user', 'content': query},
                                            ],
                                        )

                                        time.sleep(1)
                                        req_response_num += 1
                                        print('%%%%%%% Unsuccessful Time ' + str(req_response_num) + ' %%%%%%')
                                        content_length = len(
                                            response['choices'][0]['message']['content'].split('\n\n')[0])
                                        if (content_length > 20) and (req_response_num >= 10):
                                            break

                                    instructions_chatgpt = response['choices'][0]['message']['content'].split('\n\n')

                                    if len(instructions_chatgpt) == 6:
                                        for i_response in range(len(instructions_chatgpt)):
                                            if "Instruction:" in instructions_chatgpt[i_response]:
                                                instructions['instruction'] = instructions_chatgpt[i_response + 1]
                                            if "Input:" in instructions_chatgpt[i_response]:
                                                instructions['input'] = instructions_chatgpt[i_response + 1]
                                            if "Output:" in instructions_chatgpt[i_response]:
                                                instructions['output'] = instructions_chatgpt[i_response + 1]
                                    elif (len(instructions_chatgpt) >= 3) and (len(instructions_chatgpt) != 6):
                                        for i_response in range(len(instructions_chatgpt)):
                                            if "Instruction:" in instructions_chatgpt[i_response]:
                                                instructions['instruction'] = instructions_chatgpt[i_response].replace(
                                                    'Instruction:',
                                                    '')
                                            if "Input:" in instructions_chatgpt[i_response]:
                                                instructions['input'] = instructions_chatgpt[i_response].replace(
                                                    'Input:', '')
                                            if "Output:" in instructions_chatgpt[i_response]:
                                                instructions['output'] = instructions_chatgpt[i_response].replace(
                                                    'Output:', '')
                                    else:
                                        print('%%%%%%% Unsuccessful, Short! %%%%%%')
                                        instructions[
                                            'instruction'] = 'What suggestions or comments you can provide to solve or relieve ' + topic + '?'
                                        instructions['input'] = instructions_temp
                                        instructions['output'] = suggestions_temp_seg[i_segment]
                                except:
                                    print('Server taking too long. Try again later')
                                else:
                                    instructions[
                                        'instruction'] = 'What suggestions or comments you can provide to solve or relieve ' + topic + '?'
                                    instructions['input'] = instructions_temp
                                    instructions['output'] = suggestions_temp




#######################################################################################################################
                                prompts = TASK_INST[i_task] + "on " + topic + " domian ? " + "The answer is YES or NO."

                                # *********************************************** ChatGPT Instructions
                                query = f"""

                                    {PROMPT_DICT["rewrite_prompt"]}

                                    Instruction:
                                    \"\"\"
                                    {instructions['instruction']}
                                    \"\"
                                    
                                    Input:
                                    \"\"\"
                                    {instructions['input']}
                                    \"\"\"
                                    
                                    Output:
                                    \"\"\"
                                    {instructions['output']}
                                    \"\"\"
                                    
                                    Domain:
                                    \"\"\"
                                    {instructions['domain']}
                                    \"\"\"
                                                                        
                                    Task:
                                    \"\"\"
                                    {instructions['task']}
                                    \"\"\"
                                        """
                                # print(query)
                                try:
                                    response = openai.ChatCompletion.create(
                                        engine="ft-gpt4-psytherapy",  # engine = "deployment_name".
                                        messages=[
                                            {'role': 'system', 'content': 'Formate the below content.'},
                                            {'role': 'user', 'content': query},
                                        ],
                                    )
                                    # print(response['choices'][0]['message']['content'])
                                    req_response_num = 0
                                    content_length = len(response['choices'][0]['message']['content'].split('\n\n')[0])

                                    while (len(response['choices'][0]['message']['content'].split('\n\n')) < 3) or (
                                            "Instruction:" not in response['choices'][0]['message']['content']):
                                        response = openai.ChatCompletion.create(
                                            engine="ft-gpt4-psytherapy",  # engine = "deployment_name".
                                            messages=[
                                                {'role': 'system', 'content': 'You formate the below content again.'},
                                                {'role': 'user', 'content': query},
                                            ],
                                        )

                                        time.sleep(1)
                                        req_response_num += 1
                                        print('%%%%%%% Unsuccessful Time ' + str(req_response_num) + ' %%%%%%')
                                        content_length = len(response['choices'][0]['message']['content'].split('\n\n')[0])
                                        if (content_length > 20) and (req_response_num >= 10):
                                            break

                                    instructions_success = response['choices'][0]['message']['content'].split('\n\n')

                                    if instructions_success > 3:

                                        Instructions_chatgpt_train.append(instructions.copy())
                                        # print('********************Query*****************')
                                        # print(query)
                                        print('********************From ChatGPT*****************')
                                        print(instructions)
                                        instructions.clear()
                                        time.sleep(1)

                                except:
                                    print('Server taking too long. Try again later')

                        except:
                            instructions['instruction'] = 'What suggestions or comments you can provide to solve or relieve ' + topic + '?'
                            instructions['input'] = instructions_temp
                            instructions['output'] = suggestions_temp
                            Instructions_chatgpt_train.append(instructions.copy())



        elif (start_flag == True) & (content_flag == True) & (len(conversation_sample[i]) <= 100):

            if topic_fuse_flag == True:
                instructions_temp = instructions_temp + ',' + conversation_sample[i]
                content_flag = False
                i += 1
                continue
            else:
                instructions_temp = conversation_sample[i]
                content_flag = False
                #             topic_fuse_flag = True
                i += 1
                continue

        i += 1
    return Instructions_train, Instructions_chatgpt_train


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--FilePath", type=str, default='CTV_data/CTV_data1/')
    # parser.add_argument("--target-model-path", type=str, required=True)
    args = parser.parse_args()

    # ********************************************************************************************
    # {"instruction": "When did Virgin Australia start operating?", "context": "Virgin Australia, the trading name of Virgin Australia Airlines Pty Ltd, is an Australian-based airline. It is the largest airline by fleet size to use the Virgin brand. It commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route. It suddenly found itself as a major airline in Australia's domestic market after the collapse of Ansett Australia in September 2001. The airline has since grown to directly serve 32 cities in Australia, from hubs in Brisbane, Melbourne and Sydney.", "response": "Virgin Australia commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route.", "category": "closed_qa"}
    # {"instruction": "Which is a species of fish? Tope or Rope", "context": "", "response": "Tope", "category": "classification"}
    # *********************************************** models
    # EMBEDDING_MODEL = "text-embedding-ada-002"
    # GPT_MODEL = "gpt-3.5-turbo"

    openai.api_type = "azure"
    openai.api_version = "2023-05-15"
    openai.api_base = 'https://czechia.openai.azure.com'  # Your Azure OpenAI resource's endpoint value.
    openai.api_key = '919a7b563ead4f26adb45216aa98ab87'

    # *********************************************** data
    text_path = args.FilePath  #  args.FilePath   'CTV_data/CTV_data4'
    json_path = 'CTV_json/'
    json_base_path = 'CTV_json_base/'
    json_chatgpt_path = 'CTV_json_chatgpt/'
    # file_path = ''

    files = os.listdir(text_path)
    text_list = []
    for file in files:
        with open(os.path.join(text_path, file), "r", encoding="UTF-8") as f:
            conversation_sample = f.read().split('\n')
            Instructions_train, Instructions_chatgpt_train = process_data(conversation_sample, file[:-4], os.path.join(json_path, file[:-4] + '.json'))

        with open(os.path.join(json_base_path, file[:-4] + '.json'), "w") as outfile:
            json.dump(Instructions_train, outfile)

        with open(os.path.join(json_chatgpt_path, file[:-4] + '.json'), "w") as outfile:
            json.dump(Instructions_chatgpt_train, outfile)
