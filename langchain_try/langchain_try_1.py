
from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.llms.fake import FakeListLLM
from langchain.memory import SimpleMemory
from langchain.prompts import ChatPromptTemplate, PromptTemplate

# load_dotenv()

# llm = OpenAI(temperature=0.1)

responses=[
    "Use print function to print the sum of 2 and 2.",
    "Action: Python REPL\nAction Input: print(2 + 2)",
    "Final Answer: 4",
]
llm = FakeListLLM(responses=responses)

# This is an LLMChain to write a synopsis given a title of a movie and a one-liner summary.
template = """You are a screenwriter. Given the title of the movie and a one-liner summary, it is your job to expand it into a full synopsis for that movie.

Title: {title}
One-liner: {one_liner}
Screenwriter: This is a synopsis for the above movie:
"""
prompt_template = PromptTemplate(input_variables=["title", 'one_liner'], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="synopsis")

# This is an LLMChain to write a review of a movie given a synopsis.
template = """You are a film critic from the New York Times. Given the synopsis of a movie, it is your job to write a review for that movie.

Movie Synopsis:
{synopsis}
Review from a New York Times film critic of the above movie:
"""
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")

# This is an LLMChain to write a social media post given a title of a movie, a one-liner summary, the date, time and location, the synopsis of the movie, and the review of the movie.
template = """You are a social media manager for a film production company. Given the title of the movie, a one-liner summary, the date, time and location, the synopsis of the movie, and the review of the movie, it is your job to write a social media post for that movie.

Here is some context about the time and location of the movie premiere:
Date and Time: {time}
Location: {location}

Movie Synopsis:
{synopsis}
Review from a New York Times film critic of the above movie:
{review}

Social Media Post:
"""
prompt_template = PromptTemplate(input_variables=["synopsis", "review", "time", "location"], template=template)
social_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="social_post_text")

memory = SimpleMemory(memories={"time": "December 25th, 8pm PST", "location": "Los Angeles, CA"})

overall_chain = SequentialChain(
    memory=memory,
    chains=[synopsis_chain, review_chain, social_chain],
    input_variables=["title", "one_liner"],
    output_variables=["synopsis", "review", "social_post_text"],
    verbose=True)

results = overall_chain({"title": "Love Amidst the Pandemic", "one_liner": "A heart-wrenching tale of love and loss in the midst of a pandemic."})

print(results)

