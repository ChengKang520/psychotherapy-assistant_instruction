
import time
import json
import os

# # Open the file in read mode
# with open("sample.txt", "r") as file:
# # Read the entire content of the file
# content = file.read()
#
# # Split the content into words using whitespace as the delimiter
# words = content.split()
#
# # Count the number of words
# word_count = len(words)
#
# # Print the total word count
# print(f"Total number of words: {word_count}")



if __name__ == "__main__":

    # *********************************************** data
    json_path = 'data/CTV_data/'
    json_base_path = 'data/CTV_json_base/'
    json_chatgpt_path = 'data/CTV_json_chatgpt/'

    ###########################
    # Original Data
    ###########################
    text_path = json_path
    files = os.listdir(text_path)
    count_num = 0
    for file in files:
        with open(os.path.join(text_path, file), "r", encoding="UTF-8") as f:
            data = f.read()
            lines = data.split()
            # Iterating over every word in lines
            for word in lines:
                # checking if the word is numeric or not
                if not word.isnumeric():
                    count_num += 1

    print("The number of words in Original Data is: " + str(count_num))

    ###########################
    # Natural Instruction Augmented Data
    ###########################
    text_path = json_base_path
    files = os.listdir(text_path)
    count_num = 0
    for file in files:
        with open(os.path.join(text_path, file), "r", encoding="UTF-8") as f:
            data = f.read()
            lines = data.split()
            # Iterating over every word in lines
            for word in lines:
                # checking if the word is numeric or not
                if not word.isnumeric():
                    count_num += 1

    print("The number of words in Natural Instruction Augmented Data is: " + str(count_num))

    ###########################
    # Assistant Instruction Augmented Data
    ###########################
    text_path = json_chatgpt_path
    files = os.listdir(text_path)
    count_num = 0
    for file in files:
        with open(os.path.join(text_path, file), "r", encoding="UTF-8") as f:
            data = f.read()
            lines = data.split()
            # Iterating over every word in lines
            for word in lines:
                # checking if the word is numeric or not
                if not word.isnumeric():
                    count_num += 1

    print("The number of words in Assistant Instruction Augmented Data is: " + str(count_num))