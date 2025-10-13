import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import pandas as pd 
from datasets import load_dataset 
from tqdm.auto import tqdm 
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

# Flask uygulamasını başlat
app = Flask(__name__)

# --- Yapılandırma ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") 
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") 
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "psychotherapy-rag") 
EMBEDDING_DIM = 384 
MAX_RESPONSE_TOKENS = 4096 # Uzun terapötik yanıtların kesilmemesi için sınır

# Global değişkenler (Zincirin durumunu tutar)
qa_chain = None
retriever = None

# ==========================================================
# 0️⃣ Veri Yükleme ve Temizleme Fonksiyonu
# ==========================================================

def load_psychotherapy_data():
    """
    Hugging Face'ten CBT-Bench veri setini (core_fine_test alt kümesi) yükler.
    'situation' ve 'thoughts' kolonları boş olan kayıtları temizleyerek 
    anlamlı metin parçalarını (Chunk) LangChain Document objelerine dönüştürür.
    """
    
    DATASET_NAME = "Psychotherapy-LLM/CBT-Bench"
    SUBSET_NAME = "core_fine_test" 
    
    print(f"🔄 **1. Veri Yükleme:** Hugging Face '{DATASET_NAME}' ({SUBSET_NAME}) alt kümesi yükleniyor...")
    
    try:
        dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split="train") 
    except Exception as e:
        print(f"❌ **HATA (Veri Yükleme):** Hugging Face veri seti yüklenemedi. Hata: {e}")
        return []

    documents = []
    discarded_record_count = 0
    
    for i, row in enumerate(dataset):
        situation = row.get('situation')
        thoughts = row.get('thoughts')
        core_belief = row.get('core_belief_fine_grained') 
        
        # Boş (None) veya 'N/A' değerlerini içeren kayıtları atla (temizleme mantığı)
        is_situation_valid = situation and str(situation).strip() not in ['N/A', '']
        is_thoughts_valid = thoughts and str(thoughts).strip() not in ['N/A', '']

        if is_situation_valid and is_thoughts_valid:
            
            # 4 ana kolonu birleştirerek LLM için anlamlı tek bir bağlam (Chunk) oluştur
            content = (
                f"**Durum:** {situation}. "
                f"**Danışan Düşüncesi:** {thoughts}. "
                f"**Çekirdek İnançlar:** {core_belief}"
            )
            
            documents.append(Document(
                page_content=content, 
                metadata={"id": str(i), "situation_summary": situation}
            ))
        else:
            discarded_record_count += 1
    
    print(f"✅ **1. Veri Yükleme Tamamlandı:** Toplam {len(documents)} anlamlı doküman yüklendi.")
    if discarded_record_count > 0:
         print(f"⚠️ **Uyarı:** {discarded_record_count} adet eksik bilgi içeren kayıt atıldı.")
    return documents

# ==========================================================
# 1️⃣ Uygulama Başlangıcında RAG Zincirini Kurma Fonksiyonu
# ==========================================================

def initialize_rag_chain():
    """
    Sistemi başlatır: Embedding modelini yükler, Pinecone'a bağlanır, 
    veriyi koşullu olarak indeksler ve RAG zincirini (LLM + Retriever) kurar.
    """
    global qa_chain, retriever

    if not PINECONE_API_KEY or not GEMINI_API_KEY:
        print("❌ **HATA (API Anahtarı):** API anahtarları eksik.")
        return

    documents = load_psychotherapy_data()
    if not documents:
         print("❌ **RAG Zinciri Kurulumu İptal:** Yüklenecek doküman bulunamadı.")
         return
    
    TOTAL_DOCUMENT_COUNT = len(documents)
    
    # 1. Embedding Modeli Yükleme
    print(f"🔄 **2. Embedding Modeli:** 'all-MiniLM-L6-v2' yükleniyor...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("✅ **2. Embedding Modeli Tamamlandı.**")
    except Exception as e:
        print(f"❌ **HATA (Embedding):** Model yüklenirken hata oluştu. Hata: {e}")
        return

    # 2. Pinecone Bağlantısı ve İndeksleme
    try:
        print(f"🔄 **3. Pinecone Bağlantısı:** Pinecone istemcisi başlatılıyor...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print("✅ **3. Pinecone Bağlantısı Başarılı.**")
        
        index_name = PINECONE_INDEX_NAME
        index_exists = index_name in [i["name"] for i in pc.list_indexes()]
        should_upsert = True
        
        if index_exists:
            current_index = pc.Index(index_name)
            vector_count = current_index.describe_index_stats().get('total_vector_count', 0)
            
            # Koşullu Yükleme: Vektör sayısı sıfırdan büyükse yüklemeyi atla
            if vector_count > 0:
                print(f"✅ **4a. İndeks Kontrolü Başarılı:** '{index_name}' indeksi {vector_count} vektöre sahip. Yükleme adımı ATLANDI.")
                should_upsert = False
            else:
                print(f"⚠️ **4a. İndeks Mevcut, Boş:** Vektör sayısı 0. Yeniden Yükleme Başlatılıyor.")
                current_index = pc.Index(index_name) 

        else:
            # İndeks yoksa oluştur
            print(f"⚠️ **4a. İndeks Oluşturma:** '{index_name}' bulunamadı. Yeni indeks oluşturuluyor...")
            pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIM,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            current_index = pc.Index(index_name) 
        
        # YÜKLEME ADIMI (Sadece Gerekiyorsa Çalışır)
        if should_upsert:
            print(f"🔄 **4b. Veriler Yükleniyor:** {TOTAL_DOCUMENT_COUNT} vektör Pinecone'a yükleniyor...")
            
            batch_size = 100
            for i in tqdm(range(0, TOTAL_DOCUMENT_COUNT, batch_size)):
                batch = documents[i:min(i + batch_size, TOTAL_DOCUMENT_COUNT)]
                
                texts = [doc.page_content for doc in batch]
                vectors = embeddings.embed_documents(texts)
                
                to_upsert = [(str(doc.metadata['id']), vectors[j], doc.metadata) 
                             for j, doc in enumerate(batch)]

                current_index.upsert(vectors=to_upsert)
            
            print(f"✅ **4b. İndeksleme Tamamlandı.**")
        
        # Son Vektör Kontrolü ve Retriever Oluşturma
        final_index = pc.Index(index_name)
        final_vector_count = final_index.describe_index_stats().get('total_vector_count', 'Bilinmiyor')
        print(f"✨ **Pinecone Kontrol:** İndeksteki Toplam Vektör Sayısı: {final_vector_count}")
        
        vector_store = LangchainPinecone.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    except Exception as e:
        print(f"❌ **HATA (Pinecone):** Pinecone işlemi sırasında beklenmedik hata oluştu. Hata: {e}")
        return
    
    # 3. Gemini LLM ve Zincir Kurulumu
    print("🔄 **5. LLM Bağlantısı:** Gemini LLM (gemini-2.5-flash) başlatılıyor...")
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=GEMINI_API_KEY,
            max_output_tokens=MAX_RESPONSE_TOKENS 
        )

        # RAG Zinciri için Sistem İstem Şablonu (Prompt Template)
        SYSTEM_PROMPT_TEMPLATE = """
            Sen, Bilişsel Davranışçı Terapi (BDT) ilkelerine odaklanmış, empatik ve etik kurallara bağlı bir Yapay Zeka Duygusal Rehbersin. 
            Asla lisanslı bir terapist veya uzman olduğunu iddia etme. 
            Görevin, kullanıcıya destekleyici, nazik ve eyleme geçirilebilir bir tavsiye sunmaktır.

            YANIT STRATEJİN:
            1.  Öncelikle daima kendi genel BDT bilgi tabanını kullanarak yanıtı oluşturmaya BAŞLA.
            2.  Daha sonra, sana sağlanan **VERİ BAĞLAMI** içerisindeki benzer vakalardan (Çekirdek İnançlar) destek alarak yanıtını zenginleştir.
            3.  Eğer VERİ BAĞLAMI yetersizse veya alakasızsa, ona **GÜVENME** ve yine de kendi BDT bilgine dayanarak yapıcı bir yanıt ver.

            Yanıtın net ve yapılandırılmış olmalıdır:
            1.  **Doğrulama ve Empati:** Kullanıcının yaşadığı duygusal deneyimi kabul eden ve doğrulayan bir cümle.
            2.  **Bağlamsal Analiz:** Konuyu, BDT'deki yaygın 'Çekirdek İnançlar' (yetersizlik, çaresizlik vb.) çerçevesinde kısa ve nazikçe analiz et. (Bağlamı bu analizde kullan!)
            3.  **Eylem Önerisi:** BDT ilkelerine uygun (düşünceyi sorgulama veya küçük bir davranış adımı) somut bir sonraki adımı öner.

            ---
            VERİ BAĞLLAMI:
            {context} 
            ---
            """
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT_TEMPLATE), 
                ("human", "{input}"),
            ]
        )

        document_chain = create_stuff_documents_chain(llm, prompt) 
        
        # Ana RAG Zincirini kur
        qa_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=document_chain 
        )
        print("✅ **5. RAG Zinciri Kurulumu Tamamlandı!** Bot kullanıma hazır.")
        
    except Exception as e:
        print(f"❌ **HATA (Gemini LLM):** Gemini LLM başlatılırken hata oluştu. Hata: {e}")
        return


# ==========================================================
# 2️⃣ Flask Uç Noktaları (Routes)
# ==========================================================

@app.route("/")
def index():
    """Ana sayfa (index.html) arayüzünü döndürür."""
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask_question():
    """Kullanıcı sorusunu alır, RAG zincirini çalıştırır ve terapötik yanıtı döndürür."""
    global qa_chain
    
    if qa_chain is None:
        return jsonify({"answer": "Error: RAG Chain initialization failed. Please check server logs for setup errors."}), 500
        
    data = request.get_json()
    query = data.get("question")
    
    if not query:
        return jsonify({"answer": "Please provide a question."}), 400

    try:
        print(f"🔄 **Sorgu İşleniyor:** '{query[:50]}...' için RAG başlatıldı.")
        
        response = qa_chain.invoke({"input": query})
        answer = response.get("answer")
        
        # LLM'in yanıt vermediği durumu kontrol et
        if not answer:
             context_docs = response.get('context', 'Context not available.') 
             print(f"⚠️ **UYARI (LLM Yanıt Yok):** Gemini yanıt üretmedi. Context: {context_docs}")
             return jsonify({
                 "answer": "Yapay zeka, verilen bağlamla anlamlı bir yanıt oluşturamadı. Lütfen soruyu yeniden formüle edin veya veriyi kontrol edin."
             }), 500
        
        print("✅ **Sorgu Tamamlandı:** Yanıt başarıyla oluşturuldu.")
        
        return jsonify({"answer": answer})

    except Exception as e:
        # Bağlantı veya sunucu hatası durumunda detaylı hata izini terminale bas
        print(f"❌ **KRİTİK HATA (RAG Yürütme):** Sorgu sırasında hata oluştu. Detaylı Traceback:")
        traceback.print_exc() 
        
        return jsonify({
            "answer": f"API veya Sunucu Hatası: İşlem sırasında beklenmeyen bir hata oluştu ({type(e).__name__})."
        }), 500


if __name__ == "__main__":
    print("==================================================")
    print("🚀 Flask RAG Psikoterapi Botu Başlatılıyor")
    print("==================================================")
    # RAG zincirini Flask uygulaması başlamadan önce tek bir thread'de kur
    initialize_rag_chain()
    
    app.run(debug=True, host="0.0.0.0", port=5000)