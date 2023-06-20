

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


prompt_template = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in Italian:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", prompt=PROMPT)
query = "What did the president say about Justice Breyer"
print(chain({"input_documents": docs, "question": query}, return_only_outputs=True))


question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
Return any relevant text translated into italian.
{context}
Question: {question}
Relevant text, if any, in Italian:"""
QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template, input_variables=["context", "question"]
)
combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer italian. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

QUESTION: {question}
=========
{summaries}
=========
Answer in Italian:"""
COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template, input_variables=["summaries", "question"]
)
chain = load_qa_chain(OpenAI(temperature=0), chain_type="map_reduce", return_map_steps=True, question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT)
query = "What did the president say about Justice Breyer"
print(chain({"input_documents": docs, "question": query}, return_only_outputs=True))


refine_prompt_template = (
    "The original question is as follows: {question}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_str}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question. "
    "If the context isn't useful, return the original answer. Reply in Italian."
)
refine_prompt = PromptTemplate(
    input_variables=["question", "existing_answer", "context_str"],
    template=refine_prompt_template,
)
initial_qa_template = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {question}\nYour answer should be in Italian.\n"
)
initial_qa_prompt = PromptTemplate(
    input_variables=["context_str", "question"], template=initial_qa_template
)
chain = load_qa_chain(OpenAI(temperature=0), chain_type="refine", return_refine_steps=True,
                     question_prompt=initial_qa_prompt, refine_prompt=refine_prompt)
print(chain({"input_documents": docs, "question": query}, return_only_outputs=True))



from langchain.chains.summarize import load_summarize_chain

prompt_template = """Write a concise summary of the following:


{text}


CONCISE SUMMARY IN ITALIAN:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
refine_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, refine the original summary in Italian"
    "If the context isn't useful, return the original summary."
)
refine_prompt = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template,
)
chain = load_summarize_chain(OpenAI(temperature=0), chain_type="refine", return_intermediate_steps=True, question_prompt=PROMPT, refine_prompt=refine_prompt) # or "stuff" or "map_reduce"
chain({"input_documents": docs}, return_only_outputs=True)
