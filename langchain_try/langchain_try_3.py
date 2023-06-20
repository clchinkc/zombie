

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
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
texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index = True).split_documents(documents)
embeddings = TensorflowHubEmbeddings()
# embeddings = OpenAIEmbeddings()
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings =  SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(texts, embeddings)
# db = Chroma.from_documents(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))])


retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 1}) # Maximum Marginal Relevance Retrieval, specifying top k documents to return
# retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5}) # Similarity Score Threshold Retrieval, only returns documents with a score above that threshold


splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10, separator="。")
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
llm_compressor = LLMChainExtractor.from_llm(ChatOpenAI(model="gpt-3.5-turbo"))
# relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter]
)


compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)


system_template = "You are a helpful assistant that answer the questions based on the context provided."
system_prompt = PromptTemplate(template=system_template, input_variables=[])
system_message_prompt = SystemMessagePromptTemplate(prompt=system_prompt)


human_template = """Use the following pieces of context to answer the question at the end. Please answer solely based on the context provided and do not use any outside knowledge. But please answer the question as best as you can.

{context}

Question: {question}
Answer in Traditional Chinese:"""
human_prompt = PromptTemplate(
    template=human_template, input_variables=["context", "question"]
)
human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)


PROMPT = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])


# If you want to use input_variables of the PromptTemplate, you can do so with the following:
# PROMPT.format_prompt(input_language="English", output_language="French", text="I love programming.").to_messages()


qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1), chain_type="stuff", retriever=compression_retriever, chain_type_kwargs={"prompt": PROMPT}, return_source_documents=True)


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


# qa = RetrievalQA.from_chain_type(llm=OpenAI(model="text-davinci-003", temperature=0.), chain_type="map_reduce", retriever=compression_retriever, chain_type_kwargs={"question_prompt": QUESTION_PROMPT, "combine_prompt": COMBINE_PROMPT, "return_map_steps": False}, return_source_documents=True)


# If you want to take a look at the documents in the database, you can do so with the following:
# docs = retriever.get_relevant_documents("什麼是愛？")


query = "什麼是愛？"


returned_result = qa(query)


print(returned_result['result'])
print(returned_result['source_documents'])

