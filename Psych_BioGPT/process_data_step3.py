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
            "concept_explanation": "Can the following input be regarded as a concept explanation task with finite output ? The answer is YES or NO.",
            "question_answering": "Can the following input be regarded as a question answering task with finite output ? The answer is YES or NO.",
            "mental_status_assessment": "Can the following input be regarded as a mental status assessment task with finite ? The answer is YES or NO.",
            "psychological_counseling": "Can the following input be regarded as a psychological counseling task with finite ? The answer is YES or NO.",
            "information_extraction": "Can the following input be regarded as a information extraction task with finite ? The answer is YES or NO.",
            "dialogue_generation": "Can the following input be regarded as a concept explanation task with finite ? The answer is YES or NO.",
            "sentiment_analysis": "Can the following input be regarded as a concept explanation task with finite ? The answer is YES or NO.",
            "event_ordering": "Can the following input be regarded as a concept explanation task with finite ? The answer is YES or NO.",
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
                    instructions[
                        'instruction'] = 'What suggestions or comments you can provide to address or alleviate ' + topic + '?'
                    instructions['context'] = instructions_temp
                    instructions['response'] = suggestions_temp_seg[i_segment]

                    # with open(save_sample, "a") as outfile:
                    #     json.dump(instructions, outfile)
                    #     json.dump('\n', outfile)
                    Instructions_train.append(instructions.copy())
                    print('********************Instructions*****************')
                    print(instructions)

                    # *********************************************** ChatGPT Instructions
                    query = f"""  
                               Rewrite the instruction according to the topic of {topic} and the response. Remove people's name and UNKNOWN. Then, formate them all. If you cannot do that, output nothing."

                               Instruction:
                               \"\"\"
                               {instructions['instruction']}
                               \"\"

                               Context:
                               \"\"\"
                               {instructions['context']}
                               \"\"\"

                               Response:
                               \"\"\"
                               {suggestions_temp_seg[i_segment]}
                               \"\"\"
                               """
                    print(query)
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

                        instructions_chatgpt = response['choices'][0]['message']['content'].split('\n\n')

                        if len(instructions_chatgpt) == 6:
                            for i_response in range(len(instructions_chatgpt)):
                                if "Instruction:" in instructions_chatgpt[i_response]:
                                    instructions['instruction'] = instructions_chatgpt[i_response + 1]
                                if "Context:" in instructions_chatgpt[i_response]:
                                    instructions['context'] = instructions_chatgpt[i_response + 1]
                                if "Response:" in instructions_chatgpt[i_response]:
                                    instructions['response'] = instructions_chatgpt[i_response + 1]
                        elif (len(instructions_chatgpt) >= 3) and (len(instructions_chatgpt) != 6):
                            for i_response in range(len(instructions_chatgpt)):
                                if "Instruction:" in instructions_chatgpt[i_response]:
                                    instructions['instruction'] = instructions_chatgpt[i_response].replace('Instruction:',
                                                                                                           '')
                                if "Context:" in instructions_chatgpt[i_response]:
                                    instructions['context'] = instructions_chatgpt[i_response].replace('Context:', '')
                                if "Response:" in instructions_chatgpt[i_response]:
                                    instructions['response'] = instructions_chatgpt[i_response].replace('Response:', '')
                        else:
                            print('%%%%%%% Unsuccessful, Short! %%%%%%')
                            instructions[
                                'instruction'] = 'What suggestions or comments you can provide to solve or relieve ' + topic + '?'
                            instructions['context'] = instructions_temp
                            instructions['response'] = suggestions_temp_seg[i_segment]
                    except:
                        print('Server taking too long. Try again later')
                    else:
                        instructions['instruction'] = 'What suggestions or comments you can provide to solve or relieve ' + topic + '?'
                        instructions['context'] = instructions_temp
                        instructions['response'] = suggestions_temp

                    # with open(save_sample, "a") as outfile:
                    #     json.dump(instructions, outfile)
                    #     json.dump('\n', outfile)
                    Instructions_chatgpt_train.append(instructions.copy())
                    # print('********************Query*****************')
                    # print(query)
                    print('********************From ChatGPT*****************')
                    print(instructions)
                    instructions.clear()
                    time.sleep(1)

            else:

                # ***********************************************
                instructions[
                    'instruction'] = 'What suggestions or comments you can provide to solve or relieve ' + topic + '?'
                instructions['context'] = instructions_temp
                instructions['response'] = suggestions_temp

                # with open(save_sample, "a") as outfile:
                #     json.dump(instructions, outfile)
                #     json.dump('\n', outfile)
                Instructions_train.append(instructions.copy())
                print('********************Instructions*****************')
                print(instructions)


                # *********************************************** ChatGPT Instructions
                query = f"""  
                Formate and rewrite the below training data. Remove ethical information, people's name and UNKNOWN."

                Instruction:
                \"\"\"
                {instructions['instruction']}
                \"\"

                Context:
                \"\"\"
                {instructions['context']}
                \"\"\"

                Response:
                \"\"\"
                {suggestions_temp}
                \"\"\"
                """
                try:
                    response = openai.ChatCompletion.create(
                        engine="ft-gpt4-psytherapy",  # engine = "deployment_name".
                        messages=[
                            {'role': 'system', 'content': 'You formate the below content again.'},
                            {'role': 'user', 'content': query},
                        ],
                    )

                    instructions_chatgpt = response['choices'][0]['message']['content'].split('\n\n')
                    for i_response in range(len(instructions_chatgpt)):
                        if "Instruction:" in instructions_chatgpt[i_response]:
                            instructions['instruction'] = instructions_chatgpt[i_response].replace('Instruction:', '')
                        if "Context:" in instructions_chatgpt[i_response]:
                            instructions['context'] = instructions_chatgpt[i_response].replace('Context:', '')
                        if "Response:" in instructions_chatgpt[i_response]:
                            instructions['response'] = instructions_chatgpt[i_response].replace('Response:', '')
                except:
                    print('Server taking too long. Try again later')
                else:
                    instructions[
                        'instruction'] = 'What suggestions or comments you can provide to solve or relieve ' + topic + '?'
                    instructions['context'] = instructions_temp
                    instructions['response'] = suggestions_temp
                # with open(save_sample, "a") as outfile:
                #     json.dump(instructions, outfile)
                #     json.dump('\n', outfile)
                Instructions_chatgpt_train.append(instructions.copy())
                print('********************From ChatGPT*****************')
                print(instructions)
                instructions.clear()
                time.sleep(1)

            content_flag = False
            topic_fuse_flag = False
            i += 1
            instructions_temp = ''
            suggestions_temp = ''
            continue


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

    print('Hello')

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

        ha = 1