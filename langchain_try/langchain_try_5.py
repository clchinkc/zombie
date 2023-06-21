

from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

with open("../../state_of_the_union.txt") as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)



embeddings = OpenAIEmbeddings()


docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()


query = "What did the president say about Justice Breyer"
docs = docsearch.get_relevant_documents(query)


from langchain.chains.summarize import load_summarize_chain

prompt_template = """Write a concise summary of the following:


{text}


CONCISE SUMMARY IN TRADITIONAL CHINESE:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])


map_prompt_template = """Extract the most important information from the following text:


{text}


EXTRACTED INFORMATION IN TRADITIONAL CHINESE:"""
MAP_PROMPT = PromptTemplate(template=map_prompt_template, input_variables=["text"])


combine_prompt_template = """Write a concise summary of the following:


{text}


CONCISE SUMMARY IN TRADITIONAL CHINESE:"""
COMBINE_PROMPT = PromptTemplate(template=combine_prompt_template, input_variables=["text"])


refine_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, refine the original summary in TRADITIONAL CHINESE"
    "If the context isn't useful, return the original summary."
)
refine_prompt = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template,
)

chain = load_summarize_chain(OpenAI(temperature=0), chain_type="stuff", prompt=PROMPT)
chain = load_summarize_chain(OpenAI(temperature=0), chain_type="map_reduce", return_intermediate_steps=True, map_prompt=MAP_PROMPT, combine_prompt=COMBINE_PROMPT)
chain = load_summarize_chain(OpenAI(temperature=0), chain_type="refine", return_intermediate_steps=True, question_prompt=PROMPT, refine_prompt=refine_prompt) # or "stuff" or "map_reduce"


chain({"input_documents": docs}, return_only_outputs=True)
