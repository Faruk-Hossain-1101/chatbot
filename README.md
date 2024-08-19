# PDF Conversational API

This API allows users to upload PDF files and ask questions based on the content of the PDFs. The system processes the PDFs, creates a vector store, and allows for conversational interactions based on the document contents.

## Table of Contents
- [Installation](#installation)
- [Run Test](#run-test)
- [API Endpoints Documentation](#api-endpoints-documentation)
  - [Upload PDF](#upload-pdf)
  - [Ask Question](#ask-question)
  
## Installation

### Prerequisites
- Python 3.10+
- FastAPI
- Uvicorn
- Other dependencies are listed in `requirements.txt`

### Setup

1. **Clone the repository:**
   ```
   git clone https://github.com/Faruk-Hossain-1101/chatbot.git
   cd chatbot
   ```

2. **Create and activate a virtual environment:**
    ```
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**
    ```
    pip install -r requirements.txt
    ```

4. **Run the application:**
    ```
    uvicorn main:app --reload
    ```
## Run Test
```
coverage run -m pytest && coverage html
```

## API Endpoints Documentation

## Upload PDF

**Endpoint:** `/upload-pdf/`  
**Method:** `POST`  
**Description:** Upload one or more PDF files to the server. The content of the PDFs will be processed, and a vector store will be created for future queries.

### Request

- **Headers:** 
  - `Content-Type: multipart/form-data`

- **Body:** 
  - `files` (list of `UploadFile`): One or more PDF files to be uploaded.

- **Example Request:**

    ```bash
    curl -X POST "http://localhost:8000/upload-pdf/" \
         -F "files=@example.pdf" \
         -F "files=@example2.pdf"
    ```

### Response

- **Success:**

    ```json
    {
      "message": "PDF processed and vector store created."
    }
    ```

- **Error:** 

    ```json
    {
      "detail": "Error processing PDF: {error_message}"
    }
    ```

## Ask Question

**Endpoint:** `/ask-question/`  
**Method:** `POST`  
**Description:** Ask a question based on the content of the previously uploaded PDFs.

### Request

- **Headers:** 
  - `Content-Type: application/json`

- **Body:** 
  - `question` (string): The question you want to ask.

- **Example Request:**

    ```bash
    curl -X POST "http://localhost:8000/ask-question/" \
         -H "Content-Type: application/json" \
         -d '{"question": "What is the summary of the document?"}'
    ```

### Response

- **Success:**

    ```json
    {
      "answer": "This is the summary of the document...",
      "conversation": [
        {"role": "user", "content": "What is the summary of the document?"},
        {"role": "assistant", "content": "This is the summary of the document..."}
      ]
    }
    ```

- **Error:**

---

Feel free to use this markdown file to document the API endpoints for your application. Adjust the details as needed based on your implementation.
