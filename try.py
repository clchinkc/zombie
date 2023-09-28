import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()

question = "What is the capital of France?"
template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

for model in ["meta-llama/Llama-2-7b-hf", "gpt2"]:
    tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(model, use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=10,
        device=0,  # -1 for CPU
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    result = llm_chain.run(question)
    print(
        """Model: {}
Result: {}""".format(
            model, result
        )
    )
    
