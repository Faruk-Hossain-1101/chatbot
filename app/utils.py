import io
import os
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint


def generate_file_path(ip_address: str) -> str:
    """To genarate PKL path path to store traind data"""
    directory = os.path.join(os.getcwd(),'pkls')
    if not os.path.exists(directory):
        os.makedirs(directory)

    return os.path.join(directory, f'vectorstore_{ip_address}.pkl')

def save_vectorstore(vectorstore, file_path):
    """To save train data into a PKL file"""
    with open(file_path, 'wb') as f:
        pickle.dump(vectorstore, f)

def load_vectorstore(file_path):
    """To load the train PKL data"""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

def process_pdf(file_content: bytes) -> str:
    """To process the pdf files get from user and return the text value"""
    try:
        pdf_file = io.BytesIO(file_content)
        reader = PdfReader(pdf_file)

        # Extract text from all pages
        text = ""
        for page in reader.pages:
            text +=  page.extract_text()

        return text
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")
    
def get_text_chunks(text):
    """To make chhunks form the processed text"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # Remove newline characters from each chunk
    cleaned_chunks = [chunk.replace('\n', ' ').strip() for chunk in chunks]

    return cleaned_chunks

def get_vectorstore(text_chunks):
    """To train the chunks data using HuggingFace api and store the data into FAISS database"""
    try:
        hf_embeddings = HuggingFaceInferenceAPIEmbeddings(
                    api_key=os.environ.get('HUGGINGFACEHUB_API_TOKEN'),
                    model_name="sentence-transformers/all-MiniLM-l6-v2"
                )
        
        # Create a vector store with FAISS using the embeddings
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=hf_embeddings)
        return vectorstore
    except Exception as e:
        raise Exception(f"Error while embeding: {str(e)}")
    
def get_conversation_chain(vectorstore, question):
    """To create a RAG conversation using the train data from users pdf file and user query"""
    try:
        retriever = vectorstore.as_retriever()
        repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            max_new_tokens=512,
            temperature=0.5,
            huggingfacehub_api_token=os.environ.get('HUGGINGFACEHUB_API_TOKEN'),
        )

        # 2. Incorporate the retriever into a question-answering chain.
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input": question})
        return response
    except Exception as e:
        raise Exception(f"Error while embeding: {str(e)}")
    
