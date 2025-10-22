# ---------------------------------------------------------------------------
# Proje: Kariyer Rehberi Sohbet Botu (Career Guide Chatbot)
#
# Amaç:
# - Bu uygulama, RAG (Retrieval-Augmented Generation) mimarisiyle çalışan
#   bir sohbet botu örneğidir. Kullanıcılardan gelen kariyerle ilgili
#   soruları yanıtlamak için halka açık bir Soru-Cevap veri kümesini ve
#   Gemini dil modelini kullanır.
#
# Genel Yapı:
# - Uygulama, kullanıcının sorusuna en uygun yanıtı üretmek için önce
#   vektör benzerliğiyle ilgili verileri Pinecone üzerinden geri getirir
#   (retrieval), ardından bu verileri Gemini LLM ile birleştirerek
#   doğal dilde yanıt oluşturur (generation).
#
# Temel Bileşenler:
# - EmbeddingService: Yerel olarak çalışan "sentence-transformers/all-MiniLM-L6-v2"
#   modelini kullanarak metinleri sayısal vektörlere dönüştürür.
# - GeminiClient: Google Gemini API’sine istek gönderen yardımcı sınıf.
# - Pinecone Entegrasyonu: Vektör benzerliği tabanlı sorgulama için kullanılır.
# - CareerAssistant: Geri getirme (retrieval) ve yanıt üretme (generation)
#   işlemlerini yöneten ana bileşendir.
#
# Notlar:
# - Uygulama CPU üzerinde çalışacak şekilde optimize edilmiştir; GPU zorunlu değildir.
# - Ortam değişkenleri (.env dosyasında) üzerinden API anahtarları ayarlanmalıdır.
# - Gerçek bir LLM erişimi olmadan da demo amaçlı çalışacak şekilde tasarlanmıştır.
#
# Dağıtım (Deploy) Bilgisi:
# - Kod Flask tabanlı bir web uygulaması olarak yapılandırılmıştır.
# - Render, Railway veya benzeri platformlarda doğrudan çalıştırılabilir.
# - Gerekli bağımlılıklar requirements.txt dosyasında listelenmiştir.
#
# Detaylı kullanım, mimari açıklama ve veri kümesi bilgileri için README.md dosyasına bakınız.
# ---------------------------------------------------------------------------


import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import pandas as pd 
from datasets import load_dataset 
from tqdm.auto import tqdm 
import traceback 

# ==========================================================
# 🚨 BELLEK VE DONANIM OPTİMİZASYONU
# ==========================================================
# GPU varsa bile zorla CPU kullanımına yönlendiriyoruz.
# Render gibi ortamlarda GPU bulunmadığı için bu güvenli bir çözümdür.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
os.environ['NO_GPUTILS'] = '1'

# ==========================================================
# 🔗 LangChain, Pinecone ve Gemini Modül Bağlantıları
# ==========================================================
# LangChain: Zincir tabanlı LLM entegrasyon framework’ü
# Pinecone: Vektör veritabanı
# Gemini: Google Generative AI LLM modeli
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone, ServerlessSpec 

# ==========================================================
# ⚙️ Flask Uygulaması Başlatma
# ==========================================================
app = Flask(__name__)

# Ortam değişkenlerini (.env) yükle
load_dotenv()

# ==========================================================
# 🔐 Ortam Değişkenleri ve Genel Sabitler
# ==========================================================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") 
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") 
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "psychotherapy-rag") 
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 modeline uygun embedding boyutu
MAX_RESPONSE_TOKENS = 4096  # Gemini’nin maksimum çıktı uzunluğu

# Global değişkenler (zincir durumu burada tutulur)
qa_chain = None
retriever = None

# ==========================================================
# 🧩 0️⃣ Veri Yükleme Fonksiyonu (CBT-Bench Dataset)
# ==========================================================
def load_psychotherapy_data():
    """
    HuggingFace üzerinden Bilişsel Davranışçı Terapi (CBT) veri kümesini indirir.
    Eksik ya da boş verileri atar, her satırı Document objesine çevirir.
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
        
        # Geçerli kayıtları filtrele
        is_situation_valid = situation and str(situation).strip() not in ['N/A', '']
        is_thoughts_valid = thoughts and str(thoughts).strip() not in ['N/A', '']

        # Geçerli verilerden Document objesi oluştur
        if is_situation_valid and is_thoughts_valid:
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
    
    print(f"✅ **1. Veri Yükleme Tamamlandı:** {len(documents)} anlamlı doküman yüklendi.")
    if discarded_record_count > 0:
        print(f"⚠️ {discarded_record_count} adet eksik bilgi içeren kayıt atıldı.")
    return documents


# ==========================================================
# 🧠 1️⃣ RAG Zinciri Başlatma Fonksiyonu
# ==========================================================
def initialize_rag_chain():
    """
    RAG pipeline’ını kurar:
    - Veri kümesini yükler
    - all-MiniLM-L6-v2 modelinden embedding üretir
    - Pinecone’a yükler veya var olan indeksi bağlar
    - Gemini LLM’i retriever zinciriyle bağlar
    """
    global qa_chain, retriever

    # Ortam değişkenlerinin doğruluğunu kontrol et
    if not PINECONE_API_KEY or not GEMINI_API_KEY:
        raise ValueError("API Anahtarları eksik. Lütfen Render Ortam Değişkenlerini kontrol edin.")

    # 1️⃣ Veri kümesini yükle
    documents = load_psychotherapy_data()
    if not documents:
        raise ValueError("Veri kümesinden yüklenecek geçerli doküman bulunamadı.")
    
    TOTAL_DOCUMENT_COUNT = len(documents)
    
    # 2️⃣ Embedding Modeli (all-MiniLM-L6-v2)
    MODEL_NAME_OPTIMIZED = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"🔄 **2. Embedding Modeli:** '{MODEL_NAME_OPTIMIZED}' yükleniyor (CPU modunda)...")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME_OPTIMIZED,
            model_kwargs={'device': 'cpu'}  # GPU yerine CPU kullan
        )
        print("✅ **2. Embedding Modeli Başarıyla Yüklendi.**")
    except Exception as e:
        raise Exception(f"HATA (Embedding): Model yüklenirken hata oluştu. {e}")

    # 3️⃣ Pinecone Bağlantısı
    try:
        print(f"🔄 **3. Pinecone Bağlantısı:** Pinecone istemcisi başlatılıyor...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print("✅ **3. Pinecone Bağlantısı Başarılı.**")
        
        index_name = PINECONE_INDEX_NAME
        index_exists = index_name in [i["name"] for i in pc.list_indexes()]
        should_upsert = True
        
        # Eğer indeks zaten varsa, tekrar yüklememek için kontrol et
        if index_exists:
            current_index = pc.Index(index_name)
            vector_count = current_index.describe_index_stats().get('total_vector_count', 0)
            
            if vector_count > 0:
                print(f"✅ '{index_name}' indeksi {vector_count} vektöre sahip. Yeniden yükleme atlandı.")
                should_upsert = False
            else:
                print(f"⚠️ İndeks mevcut ama boş. Yeniden yükleme başlatılıyor.")
        else:
            # İndeks yoksa yeni oluştur
            print(f"⚠️ '{index_name}' bulunamadı. Yeni indeks oluşturuluyor...")
            pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIM,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            current_index = pc.Index(index_name) 
        
        # 4️⃣ Verileri Pinecone’a yükle
        if should_upsert:
            print(f"🔄 **4. Vektör Yükleme:** {TOTAL_DOCUMENT_COUNT} belge Pinecone’a aktarılıyor...")
            batch_size = 100
            for i in tqdm(range(0, TOTAL_DOCUMENT_COUNT, batch_size)):
                batch = documents[i:min(i + batch_size, TOTAL_DOCUMENT_COUNT)]
                texts = [doc.page_content for doc in batch]
                vectors = embeddings.embed_documents(texts)
                to_upsert = [(str(doc.metadata['id']), vectors[j], doc.metadata) 
                             for j, doc in enumerate(batch)]
                current_index.upsert(vectors=to_upsert)
            print(f"✅ **4. İndeksleme Tamamlandı.**")
        
        # Final kontrol
        final_index = pc.Index(index_name)
        final_vector_count = final_index.describe_index_stats().get('total_vector_count', 'Bilinmiyor')
        print(f"✨ Pinecone İndeks Durumu: {final_vector_count} toplam vektör mevcut.")
        
        # LangChain için retriever oluştur
        vector_store = LangchainPinecone.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    except Exception as e:
        raise Exception(f"HATA (Pinecone Zinciri): Pinecone başlatılamadı. {e}")
    
    # 5️⃣ Gemini LLM ile Zincir Kurulumu
    print("🔄 **5. LLM Bağlantısı:** Gemini (gemini-2.0-flash) modeli başlatılıyor...")
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=GEMINI_API_KEY,
            max_output_tokens=MAX_RESPONSE_TOKENS 
        )

        # Sistem promptu: Terapötik ve empatik yanıt üretimi
        SYSTEM_PROMPT_TEMPLATE = """
            Sen, Bilişsel Davranışçı Terapi (BDT) ilkelerine odaklanmış, empatik ve etik kurallara bağlı bir Yapay Zeka Duygusal Rehbersin. 
            Aşağıdaki VERİ BAĞLLAMI'nı kullanarak kullanıcı sorusuna destekleyici ve rehberlik edici bir yanıt ver.
            VERİ BAĞLLAMI:
            {context} 
            ---
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_TEMPLATE),
            ("human", "{input}")
        ])

        # LLM + Retriever zinciri (RAG) oluşturma
        document_chain = create_stuff_documents_chain(llm, prompt) 
        qa_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

        print("✅ **5. RAG Zinciri Başarıyla Kuruldu.** Bot kullanıma hazır.")

    except Exception as e:
        raise Exception(f"HATA (Gemini LLM): Gemini başlatılırken hata oluştu. {e}")


# ==========================================================
# 🌐 2️⃣ Flask Uç Noktaları (Routes)
# ==========================================================

# Uygulama başlarken zinciri başlat
try:
    print("==================================================")
    print("🚀 Flask RAG Psikoterapi Botu Başlatılıyor...")
    initialize_rag_chain()
    print("==================================================")

except Exception as startup_error:
    print(f"\n\n\n!!!! KRİTİK BAŞLANGIÇ HATASI !!!!\n\n")
    print(f"Hata Tipi: {type(startup_error).__name__}")
    print(f"Mesaj: {startup_error}")
    print("Render Ortam Değişkenlerinizi kontrol edin (PINECONE_API_KEY, GOOGLE_API_KEY).")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
    
# Ana sayfa (index.html)
@app.route("/")
def index():
    """Ana sayfayı (HTML arayüzü) döndürür."""
    return render_template("index.html")

# Soru-cevap endpoint’i
@app.route("/ask", methods=["POST"])
def ask_question():
    """Kullanıcı sorusunu alır, RAG zincirini çalıştırır, yanıt döndürür."""
    global qa_chain
    
    if qa_chain is None:
        return jsonify({"answer": "RAG Chain başlatılamadı. Sunucu loglarını kontrol edin."}), 500
        
    data = request.get_json()
    query = data.get("question")
    
    if not query:
        return jsonify({"answer": "Lütfen bir soru gönderin."}), 400

    try:
        print(f"🔄 **Sorgu İşleniyor:** '{query[:50]}...'")
        response = qa_chain.invoke({"input": query})
        answer = response.get("answer")
        
        if not answer:
            print("⚠️ Gemini yanıt üretmedi. Context kontrol ediliyor...")
            return jsonify({
                "answer": "Yapay zeka anlamlı bir yanıt oluşturamadı. Lütfen soruyu yeniden formüle edin."
            }), 500
        
        print("✅ **Yanıt Üretildi:** Başarılı.")
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"❌ **HATA:** Sorgu sırasında hata oluştu.")
        traceback.print_exc() 
        return jsonify({
            "answer": f"Sunucu hatası: {type(e).__name__}"
        }), 500


# ==========================================================
# 🏁 Uygulamayı Başlat (Yerel test için)
# ==========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host="0.0.0.0", port=port)
