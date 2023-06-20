
from dotenv import load_dotenv
from langchain.chains import (
    ConversationChain,
    LLMChain,
    SequentialChain,
    SimpleSequentialChain,
)
from langchain.chains.llm import LLMChain
from langchain.chains.router import MultiPromptChain, MultiRouteChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.llms.fake import FakeListLLM
from langchain.memory import SimpleMemory
from langchain.prompts import ChatPromptTemplate, PromptTemplate

load_dotenv()

# llm = ChatOpenAI(temperature=0., model="gpt-3.5-turbo", max_tokens=1500)

llm = OpenAI(temperature=0., model='text-davinci-003', max_tokens=1500)


# This is an LLMChain to write a synopsis given a title of a movie and a one-liner summary.
template = """You are a screenwriter. Given the title of the movie and a one-liner summary, it is your job to expand it into a full synopsis for that movie. You have to describe the character development and the plot of the movie.

Title: {title}
One-liner: {one_liner}
Screenwriter: This is a synopsis for the above movie:
"""
prompt_template = PromptTemplate(input_variables=["title", 'one_liner'], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="synopsis")

# This is an LLMChain to write a review of a movie given a synopsis.
template = """You are a film critic from the New York Times. Given the synopsis of a movie, it is your job to write a critical review for that movie. You have to point out the flaws of the character development and the plot and give constructive comments.

Movie Synopsis:
{synopsis}
Review from a New York Times film critic of the above movie:
"""
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")

# This is an mliti prompt chain to refine the synopsis based on the review. It either correct the character development or the plot.
character_development_template = """You are a screenwriter. Given the title of the movie, a one-liner summary, and the review of the movie, it is your job to refine the character development of the synopsis for that movie.

{input}
Screenwriter: This is a revised synopsis for the above movie:
"""

plot_template = """You are a screenwriter. Given the title of the movie, a one-liner summary, and the review of the movie, it is your job to refine the plot of the synopsis for that movie.

{input}
Screenwriter: This is a revised synopsis for the above movie:
"""

prompt_infos = [
    {
        "name": "character_development",
        "description": "Refine the character development of the synopsis based on the review.",
        "prompt_template": character_development_template,
    },
    {
        "name": "plot",
        "description": "Refine the plot of the synopsis based on the review.",
        "prompt_template": plot_template,
    }
]

# Create chains for character development and plot refinement
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt, output_key="text")
    destination_chains[name] = chain

# Create a default chain for fallback
default_chain = ConversationChain(llm=llm, output_key="text")

multi_prompt_router_template = """\
Given a raw text input to a language model select the model prompt best suited for \
the input. You will be given the names of the available prompts and a description of \
what the prompt is best suited for. You may also revise the original input if you \
think that revising it will ultimately lead to a better response from the language \
model.

REMEMBER: The JSON object MUST be wrapped in 4 curly braces ({{{{}}}}) to escape the \
markdown formatting. End the response immediately after the closing curly braces.
REMEMBER: "destination" MUST be one of the candidate prompt names specified below.
REMEMBER: "next_inputs" can just be the original input if you don't think any \
modifications are needed.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \\ name of the prompt to use
    "next_inputs": string \\ a potentially modified version of the original input
}}}}
```

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{title}}
{{synopsis}}
{{review}}

<< OUTPUT >>
"""

# Create a router chain for selecting between character development and plot refinement
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = multi_prompt_router_template.format(destinations=destinations_str)
router_prompt = PromptTemplate(template=router_template, input_variables=["title", "synopsis", "review"], output_parser=RouterOutputParser())
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

# Create a multi-prompt chain
refine_synopsis_chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True
)



memory = SimpleMemory(memories={"time": "December 25th, 8pm PST", "location": "Los Angeles, CA"})

overall_chain = SequentialChain(
    chains=[synopsis_chain, review_chain, refine_synopsis_chain],
    input_variables=["title", "one_liner"],
    output_variables=["synopsis", "review", "text"],
    verbose=True)

results = overall_chain({"title": "Love Amidst the Pandemic", "one_liner": "A heart-wrenching tale of love and loss in the midst of a pandemic."})

print(results)





