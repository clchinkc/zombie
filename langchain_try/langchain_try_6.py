
import pathlib
import subprocess
import tempfile

import requests
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def get_github_docs(repo_owner, repo_name):
    with tempfile.TemporaryDirectory() as d:
        subprocess.check_call(
            f"git clone --depth 1 https://github.com/{repo_owner}/{repo_name}.git .",
            cwd=d,
            shell=True,
        )
        git_sha = (
            subprocess.check_output("git rev-parse HEAD", shell=True, cwd=d)
            .decode("utf-8")
            .strip()
        )
        repo_path = pathlib.Path(d)
        markdown_files = list(repo_path.glob("*/*.md")) + list(
            repo_path.glob("*/*.mdx")
        )
        for markdown_file in markdown_files:
            with open(markdown_file, "r") as f:
                relative_path = markdown_file.relative_to(repo_path)
                github_url = f"https://github.com/{repo_owner}/{repo_name}/blob/{git_sha}/{relative_path}"
                yield Document(page_content=f.read(), metadata={"source": github_url})


sources = get_github_docs("yirenlu92", "deno-manual-forked")

source_chunks = []
splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
for source in sources:
    for chunk in splitter.split_text(source.page_content):
        source_chunks.append(Document(page_content=chunk, metadata=source.metadata))


search_index = Chroma.from_documents(source_chunks, OpenAIEmbeddings())


from langchain.chains import LLMChain

prompt_template = """Use the context below to write a 400 word blog post about the topic below:
    Context: {context}
    Topic: {topic}
    Blog post:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "topic"])

llm = OpenAI(temperature=0)

chain = LLMChain(llm=llm, prompt=PROMPT)


def generate_blog_post(topic):
    docs = search_index.similarity_search(topic, k=4)
    inputs = [{"context": doc.page_content, "topic": topic} for doc in docs]
    print(chain.apply(inputs))


print(generate_blog_post("environment variables"))




from langchain.text_splitter import (
    CharacterTextSplitter,
    Language,
    RecursiveCharacterTextSplitter,
)

PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
python_docs = python_splitter.create_documents([PYTHON_CODE])
print(python_docs[0])
