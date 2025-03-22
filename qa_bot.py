# LangChain components to use
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
# from langchain.llms import OpenAI
# from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub  # for text generation.
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader

# Support for dataset retrieval with Hugging Face
from datasets import load_dataset

# With CassIO, the engine powering the Astra DB integration in LangChain,
# you will also initialize the DB connection:
import cassio

ASTRA_DB_APPLICATION_TOKEN = "" # enter the "AstraCS:..." string found in in your Token JSON file
ASTRA_DB_ID = "" # enter your Database ID


# provide the path of  pdf file/files.
pdfreader = PdfReader('lalkitab.pdf')

from typing_extensions import Concatenate

# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
if content:
    raw_text += content

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
llm = HuggingFaceHub(repo_id="google/flan-t5-small",
                 huggingfacehub_api_token="",
                 model_kwargs={"temperature": 0.5, "max_length": 512})

astra_vector_store = Cassandra(
embedding=embedding,
table_name="qa_mini_demo",
session=None,
keyspace=None,
)

# We need to split the text using Character Text Split such that it sshould not increse token size
text_splitter = CharacterTextSplitter(
separator="\n",
chunk_size=800,
chunk_overlap=200,
length_function=len,
)
texts = text_splitter.split_text(raw_text)

astra_vector_store.add_texts(texts[:50])

print("Inserted %i headlines." % len(texts[:50]))

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)


def query_and_generate_answer(query_text):
    """Queries the vector store and generates an answer."""

    answer = astra_vector_index.query(query_text, llm=llm).strip()
    print(f"ANSWER: \"{answer}\"\n")

    print("FIRST DOCUMENTS BY RELEVANCE:")
    for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
        print(f"    [{score:.4f}] \"{doc.page_content[:84]} ...\"")

    return answer


# if __name__ == "__main__":
#     while True:
#         query = input("Enter your question (or 'quit' to exit): ").strip()
#         if query.lower() == "quit":
#             break
#         query_and_generate_answer(query)