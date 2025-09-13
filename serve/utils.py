def generate_prompt(instruction, input=None):
    if input:
        return ("The following is an instruction that describes the task and an input related to the task information. Please write a response that appropriately completes this instruction.\n\n"
                f"### Command：\n{instruction}\n\n### Input：\n{input}\n\n### Response：")
    else:
        return ("The following is an instruction that describs the task. Please write a response that appropriately completes this instruction.\n\n"
                f"### Command：\n{instruction}\n\n### Response：\n")
