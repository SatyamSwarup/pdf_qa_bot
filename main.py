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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # provide the path of  pdf file/files.
    pdfreader = PdfReader('Satyam_Swarup_Resume.pdf')

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

    first_question = True
    while True:
        if first_question:
            query_text = input("\nEnter your question (or type 'quit' to exit): ").strip()
        else:
            query_text = input("\nWhat's your next question (or type 'quit' to exit): ").strip()

        if query_text.lower() == "quit":
            break

        if query_text == "":
            continue

        first_question = False

        print("\nQUESTION: \"%s\"" % query_text)
        answer = astra_vector_index.query(query_text, llm=llm).strip()
        print("ANSWER: \"%s\"\n" % answer)

        print("FIRST DOCUMENTS BY RELEVANCE:")
        for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
            print("    [%0.4f] \"%s ...\"" % (score, doc.page_content[:84]))


def get_answer(query_text):
    answer = ""
    if query_text == "":
        answer = "Please Enter valid prompt"
    else:
        answer = astra_vector_index.query(query_text, llm=llm).strip()
    return answer
