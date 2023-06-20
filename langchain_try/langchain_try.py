
from dotenv import load_dotenv
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.llms.fake import FakeListLLM
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, PromptTemplate

# load_dotenv()

# Initialize the language model
# llm = OpenAI(temperature=0.1)

# Initialize the fake language model
responses=[
    "Use print function to print the sum of 2 and 2.",
    "Action: Python REPL\nAction Input: print(2 + 2)",
    "Final Answer: 4",
]
llm = FakeListLLM(responses=responses)

# Initialize the memory
memory = ConversationBufferMemory()

# Define the templates
design_template = """
You are tasked to design a Python software component to implement a new feature.

Feature: {feature}
Design:"""

structure_template = """
You need to come up with Python classes and functions to implement the following software design: 

Design: {design}
Structure:"""

code_template = """
Now, you are to write Python code based on the following structure: 

Structure: {structure}
Code:"""

# Initialize the PromptTemplates
design_prompt = PromptTemplate(
    input_variables=["feature"],
    template=design_template,
    validate_template=True,
)
structure_prompt = PromptTemplate(
    input_variables=["design"],
    template=structure_template,
    validate_template=True,
)
code_prompt = PromptTemplate(
    input_variables=["structure"],
    template=code_template,
    validate_template=True,
)

# Create the chains
design_chain = LLMChain(llm=llm, prompt=design_prompt, memory=memory)
structure_chain = LLMChain(llm=llm, prompt=structure_prompt, memory=memory)
code_chain = LLMChain(llm=llm, prompt=code_prompt, memory=memory)

# Combine the chains
overall_chain = SimpleSequentialChain(chains=[design_chain, structure_chain, code_chain])

# Run the chain
feature = "a user authentication system"
design_and_code = overall_chain.run(feature)

print(design_and_code)
