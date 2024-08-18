import os
import io
import pickle
import pytest
import hashlib
from fastapi import Request
from fastapi.testclient import TestClient
from main import app
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
from unittest.mock import patch
from utils import  ( 
    generate_file_path,
    save_vectorstore,
    load_vectorstore,
    calculate_data_hash,
    process_pdf,
    get_text_chunks,
    get_vectorstore,
    get_conversation_chain,
)

client = TestClient(app)
client2 = TestClient(app)

@pytest.fixture
def test_request():
    return Request(scope={"type": "http"})

# Create a temporary file for testing
def create_single_page_pdf(file_path):
    c = canvas.Canvas(file_path, pagesize=letter)
    c.drawString(100, 750, "This is a test PDF with a single page.")
    c.save()

## Fixtures
@pytest.fixture
def vectorstore_from_pdf():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        create_single_page_pdf(tmp.name) 

    # Step 2: Process the PDF to extract text
    try:
        with open(tmp.name, 'rb') as file:
            file_content = file.read()
            text = process_pdf(file_content)
        
        # Step 3: Generate text chunks
        text_chunks = get_text_chunks(text)

        # Step 4: Create the vector store
        vectorstore = get_vectorstore(text_chunks)

        yield vectorstore
    finally:
        # Clean up: Remove the temporary PDF file
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

@pytest.fixture
def single_page_pdf():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        create_single_page_pdf(tmp.name) 
        yield tmp.name
        os.remove(tmp.name)


## Function Test
def test_generate_file_path():
    ip_address = '127.0.0.1'
    expected_path = os.path.join(os.getcwd(), 'pkls', f'vectorstore_{ip_address}.pkl')
    assert generate_file_path(ip_address) == expected_path

def test_save_load_vectorstore():
    # Mock vectorstore
    vectorstore = {'key': 'value'}
    file_path = 'test_vectorstore.pkl'
    save_vectorstore(vectorstore, file_path)
    loaded_vectorstore = load_vectorstore(file_path)
    assert loaded_vectorstore == vectorstore
    os.remove(file_path)

def test_calculate_data_hash():
    data = "test data"
    expected_hash = hashlib.sha256(data.encode()).hexdigest()
    assert calculate_data_hash(data) == expected_hash

def test_process_pdf(single_page_pdf):
    try:
        with open(single_page_pdf, 'rb') as f:
            file_content = f.read()
            result = process_pdf(file_content)
            assert "This is a test PDF with a single page." in result
    except Exception as e:
        pytest.fail(f"Exception raised: {str(e)}")


def test_get_text_chunks():
    text = "This is a test text that will be split into chunks.\nNew line here."
    chunks = get_text_chunks(text)
    assert len(chunks) > 0
    assert "This is a test text that will be split into chunks." in chunks[0]
    
@patch('langchain_community.vectorstores.FAISS.from_texts')
def test_get_vectorstore(mock_faiss):
    mock_faiss.return_value = 'mock_vectorstore'
    text_chunks = ["chunk1", "chunk2"]
    vectorstore = get_vectorstore(text_chunks)
    assert vectorstore == 'mock_vectorstore'


@patch('langchain_community.vectorstores.FAISS.from_texts')
def test_get_vectorstore_raise_exception(mock_faiss):
    text_chunks = ["chunk1", "chunk2"]
    # Simulate the behavior of the exception
    mock_faiss.side_effect = Exception("FAISS error")
    
    # Call get_vectorstore and check if it raises an exception
    with pytest.raises(Exception, match="Error while embeding: FAISS error"):
        get_vectorstore(text_chunks)

def test_get_conversation_chain(vectorstore_from_pdf):
    mock_vectorstore = vectorstore_from_pdf
    question = "What is the summary?"
    response = get_conversation_chain(mock_vectorstore, question)
    assert response is not None
    assert 'answer' in response


## Api Test
def test_upload_pdf(test_request, single_page_pdf):
    with open(single_page_pdf, "rb") as f:
        response = client.post(
            "/upload-pdf/",
            files={"files": ("example.pdf", f, "application/pdf")},
        )
    assert response.status_code == 200
    assert response.json()["message"] == "PDF processed and vector store created."

def test_upload_pdf_failed(test_request, single_page_pdf):
    response = client.post(
        "/upload-pdf/",
        files={"files": ("example.pdf", 'text data', "application/pdf")},
    )
    assert response.status_code == 500

def test_ask_question(test_request, vectorstore_from_pdf):
    # Simulate asking a question
    question_data = {"question": "What is the content of the PDF?"}
    response = client.post("/ask-question/", json=question_data)

    # Ensure that the response is correct
    assert response.status_code == 200
    json_response = response.json()
    assert "answer" in json_response
    assert "conversation" in json_response
    assert json_response["conversation"][-1]["content"] == json_response["answer"]
    assert json_response["conversation"][-1]["role"] == "assistant"

    
def test_ask_question_without_vectorstore(test_request):
    # Simulate asking a question
    question_data = {"question": "What is the content of the PDF?"}
    response = client2.post("/ask-question/", json=question_data)

    # Ensure that the response is correct
    assert response.status_code == 200
    json_response = response.json()
    assert "answer" in json_response
    assert "conversation" in json_response
    assert json_response["conversation"][-1]["content"] == json_response["answer"]
    assert json_response["conversation"][-1]["role"] == "assistant"

if __name__ == "__main__":
    pytest.main()
