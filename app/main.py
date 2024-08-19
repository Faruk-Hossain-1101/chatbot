import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
import multiprocessing
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from .utils import (
    process_pdf, 
    get_text_chunks, 
    get_conversation_chain, 
    get_vectorstore, 
    generate_file_path,
    save_vectorstore,
    load_vectorstore,
)

app = FastAPI()
load_dotenv()
app.add_middleware(SessionMiddleware, secret_key=os.environ.get('SECRET_KEY'))


@app.post("/upload-pdf/")
async def read_pdfs(request: Request, files: list[UploadFile] = File(...)):
    ip_address = request.client.host
    try:
        file_contents = [await file.read() for file in files]

        # Use multiprocessing to handle the PDF processing in parallel
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(process_pdf, file_contents)

        # Combine all extracted text into a single variable
        combined_text = " ".join(results)

        # get the text chunks
        text_chunks = get_text_chunks(combined_text)

        # create vector store
        vectorstore = get_vectorstore(text_chunks)

        # Genarate file path
        file_path = generate_file_path(ip_address=ip_address)

        # Save vectorstore in file
        save_vectorstore(vectorstore=vectorstore, file_path=file_path)

        return {"message": "PDF processed and vector store created."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask-question/")
async def ask_question(request: Request, body: QuestionRequest ):
    ip_address = request.headers.get("X-Forwarded-For", request.client.host)
    vectorstore_path = generate_file_path(ip_address)
    # Load vectorstore
    vectorstore = load_vectorstore(vectorstore_path)

    if not vectorstore:
        vectorstore_path = generate_file_path('default')
        vectorstore = load_vectorstore(vectorstore_path)

    # Retrieve conversation history from session
    session = request.session
    conversation_history = session.get("conversation_history", [])

    # Append the new question to conversation history
    conversation_history.append({"role": "user", "content": body.question})

    response = get_conversation_chain(vectorstore, body.question )
    answer = response['answer']

    # Update conversation history with the response
    conversation_history.append({"role": "assistant", "content": answer})

    # Save updated conversation history to session
    session["conversation_history"] = conversation_history

    return {
        "answer": answer,
        "conversation": conversation_history,
    }