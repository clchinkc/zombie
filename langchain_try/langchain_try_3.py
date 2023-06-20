

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.embeddings import (
    HuggingFaceEmbeddings,
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
    TensorflowHubEmbeddings,
)
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
    LLMChainExtractor,
)
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import FAISS, Chroma

load_dotenv()


documents = TextLoader(r'C:\\Users\\kevin\\Desktop\\try.txt', encoding="utf-8").load()
texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, add_start_index = True).split_documents(documents)
embeddings = TensorflowHubEmbeddings()
# embeddings = OpenAIEmbeddings()
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings =  SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(texts, embeddings)
# db = Chroma.from_documents(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))])


retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 1}) # Maximum Marginal Relevance Retrieval, specifying top k documents to return
# retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5}) # Similarity Score Threshold Retrieval, only returns documents with a score above that threshold


splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator="。")
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
# llm_compressor = LLMChainExtractor.from_llm(ChatOpenAI(model="gpt-3.5-turbo"))
# relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter]
)


compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)


prompt_template = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in Italian:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
Return any relevant text.
{context}
Question: {question}
Relevant text, if any, in Traditional Chinese:"""
QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template, input_variables=["context", "question"]
)

combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer in Traditional Chinese.

QUESTION: {question}
=========
{summaries}
=========
Answer in Traditional Chinese:"""
COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template, input_variables=["summaries", "question"]
)


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
    "If the context isn't useful, return the original answer. Reply in Traditional Chinese."
)
REFINE_PROMPT = PromptTemplate(
    input_variables=["question", "existing_answer", "context_str"],
    template=refine_prompt_template,
)
initial_qa_template = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {question}\nYour answer should be in Traditional Chinese.\n"
)
INITIAL_QA_PROMPT = PromptTemplate(
    input_variables=["context_str", "question"], template=initial_qa_template
)


# If you want to use input_variables of the PromptTemplate, you can do so with the following:
# chat_prompt.format_prompt(input_language="English", output_language="French", text="I love programming.").to_messages()


qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0., n=2, best_of=2), chain_type="stuff", retriever=compression_retriever, chain_type_kwargs={"prompt": QUESTION_PROMPT}, return_source_documents=True)
# qa = RetrievalQA.from_chain_type(llm=OpenAI(model="text-davinci-003", temperature=0., n=2, best_of=2), chain_type="map_reduce", retriever=compression_retriever, chain_type_kwargs={"question_prompt": QUESTION_PROMPT, "combine_prompt": COMBINE_PROMPT, "return_map_steps": False}, return_source_documents=True)
# qa = RetrievalQA.from_chain_type(llm=OpenAI(model="text-davinci-003", temperature=0., n=2, best_of=2), chain_type="refine", retriever=compression_retriever, chain_type_kwargs={"question_prompt": INITIAL_QA_PROMPT, "refine_prompt": REFINE_PROMPT, "return_refine_steps": False}, return_source_documents=True)
# qa = load_qa_chain(OpenAI(model="text-davinci-003", temperature=0, n=2, best_of=2), chain_type="stuff", prompt=PROMPT)
# qa = load_qa_chain(OpenAI(model="text-davinci-003", temperature=0, n=2, best_of=2), chain_type="map_reduce", return_map_steps=True, question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT)
# qa = load_qa_chain(OpenAI(model="text-davinci-003", temperature=0, n=2, best_of=2), chain_type="refine", return_refine_steps=True, question_prompt=INITIAL_QA_PROMPT, refine_prompt=REFINE_PROMPT)


query = "什麼是愛？"


# docs = compression_retriever.get_relevant_documents(query)
# returned_result = qa({"input_documents": docs, "question": query})


returned_result = qa({"query": query})


# print(returned_result["input_documents"])
# print(returned_result["intermediate_steps"])
# print(returned_result["output_text"])

print(returned_result['source_documents'])
print(returned_result['result'])


