import os
import tempfile
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from app.utils import process_pdf, get_text_chunks, get_vectorstore

# Create a temporary file for testing
def create_single_page_pdf(file_path):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(file_path, pagesize=letter)
    c.drawString(100, 750, "This is a test PDF with a single page.")
    c.save()

@pytest.fixture
def single_page_pdf():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        create_single_page_pdf(tmp.name)
        yield tmp.name
        os.remove(tmp.name)

@pytest.fixture
def vectorstore_from_pdf():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        create_single_page_pdf(tmp.name)
    
    try:
        with open(tmp.name, 'rb') as file:
            file_content = file.read()
            text = process_pdf(file_content)
            text_chunks = get_text_chunks(text)
            vectorstore = get_vectorstore(text_chunks)
            yield vectorstore
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

@pytest.fixture
def test_request():
    return Request(scope={"type": "http"})

@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)
