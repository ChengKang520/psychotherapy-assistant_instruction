

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from datasets import load_dataset
from peft import LoraConfig, PeftModel

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import AsyncChromiumLoader



from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
import nest_asyncio
nest_asyncio.apply()






#################################################################
# Tokenizer
#################################################################

model_name = 'mistralai/Mistral-7B-Instruct-v0.1'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#################################################################
# bitsandbytes parameters
#################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

#################################################################
# Set up quantization config
#################################################################
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

#################################################################
# Load pre-trained config
#################################################################
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)



def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(model))


text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=1000,
)

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)




############   TEXT files loader   ###########
loader = DirectoryLoader('/home/kangchen/Chatbot/Psych_BioGPT/CTV_data/', glob="*.txt", loader_cls=TextLoader)

# ############   PDF files loader   ###########
# loader = DirectoryLoader('output/', glob="*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()

#splitting the text into
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunked_documents = text_splitter.split_documents(documents)

# # Articles to index
# articles = ["https://www.fantasypros.com/2023/11/rival-fantasy-nfl-week-10/"]
#
# # Scrapes the blogs above
# loader = AsyncChromiumLoader(articles)
# docs = loader.load()
# # Converts HTML to plain text
# html2text = Html2TextTransformer()
# docs_transformed = html2text.transform_documents(docs)
#
# # Chunk text
# text_splitter = CharacterTextSplitter(chunk_size=100,
#                                       chunk_overlap=0)
#
# chunked_documents = text_splitter.split_documents(docs_transformed)

# Load chunked documents into the FAISS index
db = FAISS.from_documents(chunked_documents,
                          HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

retriever = db.as_retriever()



prompt_template = """
### [INST] Instruction: Answer the question based on your fantasy football knowledge. Here is context to help:

{context}

### QUESTION:
{question} [/INST]
 """

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Create llm chain
llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)



questions = "What is Depression Behavioral Activation and Cognitive Change？"
############   Original   ###########
original_result = llm_chain.invoke({"context": "", "question": questions})
# print("RAG text: " + original_result['text'])
print("*******************************************************")
# print(original_result)
print("Original output: " + original_result['text'])



############   RAG   ###########
rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | llm_chain)
rag_result = rag_chain.invoke(questions)
print("*******************************************************")
# print(rag_result)
print("RAG output: " + rag_result['text'])


