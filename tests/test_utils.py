import os
import pytest
import hashlib
from unittest.mock import patch, Mock
from app.utils import  ( 
    generate_file_path,
    save_vectorstore,
    load_vectorstore,
    calculate_data_hash,
    process_pdf,
    get_text_chunks,
    get_vectorstore,
    get_conversation_chain,
)

def test_generate_file_path():
    ip = "127.198.0.1"
    file_path = generate_file_path(ip)
    directory = os.path.join(os.getcwd(),'pkls') 
    # Check if the file exists
    assert os.path.exists(directory), "File was not created"

def test_save_and_load_vectorstore():
    # Mock vectorstore
    vectorstore = {'key': 'value'}
    file_path = 'test_vectorstore.pkl'
    save_vectorstore(vectorstore, file_path)
    loaded_vectorstore = load_vectorstore(file_path)
    assert loaded_vectorstore == vectorstore
    os.remove(file_path)

def test_load_vectorstore_with_blank_file():
    response = load_vectorstore("dummy.txt")
    assert response == None

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

def test_process_pdf_with_invalid_file():
    invalid_pdf_content = b"This is just a text file, not a PDF."
    with pytest.raises(Exception, match="Error processing PDF:"):
        process_pdf(invalid_pdf_content)


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

def test_get_conversation_chain_with_invalid_vectorstore():
    # Create an invalid vectorstore mock
    invalid_vectorstore = Mock()
    invalid_vectorstore.as_retriever.side_effect = Exception("Invalid vectorstore")

    # Define a sample question
    question = "What is the capital of France?"

    # Test the function and verify it raises an exception
    with pytest.raises(Exception, match="Error while embeding:"):
        get_conversation_chain(invalid_vectorstore, question)

