import os
import threading
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import traceback

# LangChain, Pinecone ve Gemini Modül Bağlantıları
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone, ServerlessSpec

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "psychotherapy-rag")

qa_chain = None
retriever = None


def initialize_rag_chain():
    global qa_chain, retriever
    print("🔄 RAG zinciri başlatılıyor...")

    try:
        # 1️⃣ Basit bir test için kısaltılmış başlangıç (gerekirse tam versiyonunu koy)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        vector_store = LangchainPinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)
        SYSTEM_PROMPT_TEMPLATE = "Sen empatik bir BDT danışmanısın. {context}"
        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT_TEMPLATE), ("human", "{input}")])
        document_chain = create_stuff_documents_chain(llm, prompt)
        qa_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

        print("✅ RAG zinciri başarıyla yüklendi.")
    except Exception as e:
        print(f"❌ HATA: RAG zinciri başlatılamadı: {e}")
        traceback.print_exc()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask_question():
    global qa_chain
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"answer": "Please provide a question."}), 400
    if qa_chain is None:
        return jsonify({"answer": "RAG system is still initializing. Please try again in a moment."}), 503

    try:
        result = qa_chain.invoke({"input": question})
        return jsonify({"answer": result.get("answer", "No response")})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"answer": f"Error: {e}"}), 500


if __name__ == "__main__":
    # Arka planda RAG zincirini başlat
    threading.Thread(target=initialize_rag_chain).start()

    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Flask app starting on port {port}")
    app.run(host="0.0.0.0", port=port)
