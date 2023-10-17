import os
import sys

import torch
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    set_seed,
)

# Set seed for reproducibility
set_seed(42)

# Define generation methods and parameters
GENERATION_METHODS = {
    "sampling": {
        "do_sample": True,
        "temperature": 0.125,
        "top_p": 0.9,
    },
    "beam_search": {
        "do_sample": False,
        "num_beams": 10,
        "early_stopping": True,
    }
}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Prompt modification
template = """
You are an assistant answering questions.
{chat_history}
________________
Question: {question_input}
Answer:
"""

prompt = PromptTemplate(input_variables=["chat_history", "question_input"], template=template)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chains = {}
model_list = [
    # "facebook/opt-350m",
    # "gpt2",
    "facebook/opt-1.3b",
    # "facebook/blenderbot-3B",
    # "meta-llama/Llama-2-7b-chat-hf"
]

for model_name in model_list:
    llm = HuggingFacePipeline(pipeline=None)  # Will set the pipeline later
    llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory)
    llm_chains[model_name] = llm_chain

def load_models(model_list):
    loaded_models = {}
    loaded_tokenizers = {}
    for model_name in model_list:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            offload_folder="offload", 
            offload_state_dict=True, 
            quantization_config=bnb_config
        )
        loaded_models[model_name] = model
        loaded_tokenizers[model_name] = tokenizer
    return loaded_models, loaded_tokenizers

loaded_models, loaded_tokenizers = load_models(model_list)

def generate(prompt_data, model_name, tokenizer, method="beam_search"):
    model = loaded_models[model_name]
    # pipe = pipeline('text2text-generation', model='facebook/blenderbot-3B', tokenizer='facebook/blenderbot-3B', d_type=torch.float16)
    pipe = pipeline(
        "text-generation",
        model=model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=50,
        no_repeat_ngram_size=3,
        top_k=50,
        **GENERATION_METHODS[method]
    )
    llm_chain = llm_chains[model_name]
    llm_chain.llm.pipeline = pipe
    result = llm_chain.apply([prompt_data])
    return result

def chatbot_response(chat_history, user_input, model_name):
    tokenizer = loaded_tokenizers[model_name]
    chat_prompt_data = {"chat_history": chat_history, "question_input": user_input}
    response = generate(chat_prompt_data, model_name, tokenizer)
    full_response = response[0]['text']
    prefix = f"________________\nQuestion: {user_input}\nAnswer: "
    bot_response = full_response.replace(prefix, '').strip()
    chat_history += f"________________\nQuestion: {user_input}\nAnswer: {bot_response}\n"
    return bot_response, chat_history

questions = [
    "What is ray tracing?",
    "What is the difference between ray tracing and path tracing?",
    "What is my last question?"
]

chat_history = "________________\nQuestion: Who are you?\nAnswer: Hello! I am your assistant. You can ask any question using 'Question: ' and I will answer using 'Answer: '."
print(chat_history)
for user_input in questions:
    print('________________\n')
    print(f'Question: {user_input}')
    for model_name in model_list:
        bot_response, chat_history = chatbot_response(chat_history, user_input, model_name)
        print(f'Answer from {model_name}: {bot_response}')



# https://twitter.com/joao_gante/status/1679775399172251648
# https://huggingface.co/inference-endpoints
# https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da


# https://github.com/facebookresearch/llama-recipes/tree/main

# https://github.com/huggingface/text-generation-inference
