import os
import sys

import torch
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
template = """Question: {question}\nAnswer: """
prompt = PromptTemplate(template=template, input_variables=["question"])

# Load models outside the function so it's done only once
llm_chains = {}
for model_name in ["facebook/opt-350m"]:  # "gpt2", "facebook/opt-1.3b", "facebook/blenderbot-3B", "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=os.getenv("HUGGINGFACE_API_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=os.getenv("HUGGINGFACE_API_TOKEN"), offload_folder="offload", offload_state_dict=True, load_in_4bit=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        device_map="auto",
        torch_dtype=torch.float16,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.5,
        max_new_tokens=50,
        do_sample=True,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
        top_p=0.9,
        top_k=50,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    llm_chains[model_name] = LLMChain(prompt=prompt, llm=llm)

def generate(question):
    results = {}
    for model_name, llm_chain in llm_chains.items():
        result = llm_chain.run(question)
        results[model_name] = result
    return results

def main():
    while True:
        question = input("Question: ").strip()
        if not question:  # Exit loop if input is empty
            break
        answers = generate(question)
        for model_name, answer in answers.items():
            print(f"Model: {model_name}\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()
    
