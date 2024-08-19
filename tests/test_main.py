
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


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
    custom_ip = "198.162.0.0"  # Example IP address to simulate
    question_data = {"question": "What is the content of the PDF?"}

    # Use a custom header to simulate the IP address
    response = client.post("/ask-question/", json=question_data, headers={"X-Forwarded-For": custom_ip})

    # Ensure that the response is correct
    assert response.status_code == 200
    json_response = response.json()
    assert "answer" in json_response
    assert "conversation" in json_response
    assert json_response["conversation"][-1]["content"] == json_response["answer"]
    assert json_response["conversation"][-1]["role"] == "assistant"

