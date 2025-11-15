import os
import requests
from flask import Flask, request, jsonify
from openai import OpenAI

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

import json

# ----------------------------
# CONFIG
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MEMBER_API_URL = "https://november7-730026606190.europe-west1.run.app/messages/?skip=0&limit=100"  # The external JSON API
VECTORSTORE_DIR = "vectorstore"

app = Flask(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)


# ----------------------------
# LOAD OR CREATE VECTORSTORE
# ----------------------------
def load_or_create_vectorstore():
    # If vectorstore already exists → load it
    if os.path.exists(VECTORSTORE_DIR):
        print("Loading existing vectorstore...")
        return FAISS.load_local(
            VECTORSTORE_DIR,
            OpenAIEmbeddings(api_key=OPENAI_API_KEY),
            allow_dangerous_deserialization=True
        )

    print("Fetching messages from API:", MEMBER_API_URL)
    response = requests.get(MEMBER_API_URL)
    response.raise_for_status()

    data = response.json()

    messages = []
    for msg in data.get("items", []):
        text = msg.get("message", "")
        user = msg.get("user_name", "Unknown")
        messages.append(f"{user}: {text}")

    # Convert to LangChain document format
    from langchain.schema import Document
    docs = [Document(page_content=m) for m in messages]

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local(VECTORSTORE_DIR)
    return vectorstore


vectorstore = load_or_create_vectorstore()


# ----------------------------
# RAG ENDPOINT
# ----------------------------
@app.post("/ask")
def ask_question():
    data = request.get_json()
    question = data.get("question", "")

    # similarity search
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You must answer based ONLY on the following member messages:

{context}

Question: {question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content
    print(answer)
    return jsonify({"answer": answer})


@app.get("/")
def home():
    return {"status": "RAG service running"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))