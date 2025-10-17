import os
import json
import logging
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from datasets import load_dataset 
from tqdm.auto import tqdm 
import traceback 
import requests # Yeni eklenen: Doğrudan API çağrıları için
import numpy as np # Yeni eklenen: Embedding sonuçları için
from pinecone import Pinecone, ServerlessSpec 
from sentence_transformers import SentenceTransformer # Yeni eklenen: Bellek dostu embedding için

# --- 1. LOGGING VE CONFIG ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

class Config:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "psychotherapy-rag")
    PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1") # Pinecone region'ınızı kontrol edin
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # Bellek aşımı durumunda bunu değiştirebilirsiniz
    EMBEDDING_DIM = 384
    MAX_RESPONSE_TOKENS = 4096 
    BATCH_SIZE = 100
    K_RETRIEVAL = 3

# --- 2. EMBEDDING SERVİSİ (Bellek Yüklemesini Yönetmek İçin) ---
class EmbeddingService:
    def __init__(self, model_name: str):
        self.model = None
        logging.info(f"🔄 **2. Embedding Modeli:** '{model_name}' yükleniyor...")
        try:
            # Model, STransformer ile yüklenerek bellek yükü optimize edilir.
            self.model = SentenceTransformer(model_name)
            logging.info("✅ **2. Embedding Modeli Tamamlandı.**")
        except Exception as e:
            logging.error(f"❌ HATA (Embedding): Model yüklenirken hata oluştu. {e}")
            raise RuntimeError("Embedding modeli yüklenemedi.")

    def embed(self, text: str):
        if not self.model:
            raise RuntimeError("Embedding modeli yüklenmedi.")
        # NumPy dizisini doğrudan Pinecone için List'e çeviriyoruz
        return self.model.encode(text).tolist()

# --- 3. GEMINI MÜŞTERİSİ (Doğrudan API Erişimi İçin) ---
class GeminiClient:
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model_name
        self.headers = {"Content-Type": "application/json"}
        if not api_key:
            logging.warning("⚠️ Gemini API Key eksik. Mock yanıtlar kullanılacak.")

    def generate(self, question: str, context: str):
        if not self.api_key:
            return "Mock Response: Gemini API Key not configured. Context: " + context

        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"
        
        # BDT odaklı sistem talimatı (Sizin prompt'unuza dayalı)
        system_instruction = f"""
            Sen, Bilişsel Davranışçı Terapi (BDT) ilkelerine odaklanmış, empatik ve etik kurallara bağlı bir Yapay Zeka Duygusal Rehbersin. 
            Kullanıcının sorusuna yalnızca aşağıdaki VERİ BAĞLAMI'nı kullanarak BDT prensiplerine uygun, destekleyici ve rehberlik edici bir yanıt ver.
            Eğer bağlam yetersizse, etik kurallara bağlı kalarak genel bir BDT rehberliği yap.
            
            VERİ BAĞLLAMI:
            {context} 
            ---
            Kullanıcının Sorusu:
        """
        
        # Payload oluştur
        payload = {
            "contents": [{"role": "user", "parts": [{"text": system_instruction + question}]}],
            "config": {
                "systemInstruction": system_instruction,
                "maxOutputTokens": Config.MAX_RESPONSE_TOKENS
            }
        }

        try:
            r = requests.post(url, headers=self.headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            
            # API yanıtını güvenli şekilde çöz
            if "candidates" in data and data["candidates"]:
                content = data["candidates"][0].get("content", {})
                if "parts" in content and content["parts"]:
                    return content["parts"][0].get("text", "Gemini'den yanıt alınamadı.")
            return "Gemini'den geçerli bir yanıt gelmedi."

        except requests.exceptions.HTTPError as http_err:
            logging.error(f"❌ Gemini HTTP hatası: {http_err} - {r.text}")
            return f"API Hatası: HTTP {r.status_code}."
        except Exception as e:
            logging.error(f"❌ Gemini API genel hatası: {e}")
            return "Gemini API bağlantı hatası."

# --- 4. PSİKOTERAPİ ASİSTANI (Ana Orkestratör) ---
class PsychotherapyAssistant:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Bellek aşımı riskini azaltmak için Embedding Modelini burada yüklüyoruz.
        self.embedder = EmbeddingService(cfg.EMBEDDING_MODEL) 
        self.gemini = GeminiClient(cfg.GEMINI_API_KEY)
        self.pinecone_index = self._setup_pinecone()
        self.documents = self._load_psychotherapy_data()
        self._load_dataset_to_pinecone() # İndeks boşsa yükleme yapar

    # Sizin load_psychotherapy_data fonksiyonunuz (Değişmedi)
    def _load_psychotherapy_data(self):
        DATASET_NAME = "Psychotherapy-LLM/CBT-Bench"
        SUBSET_NAME = "core_fine_test" 
        logging.info(f"🔄 **1. Veri Yükleme:** Hugging Face '{DATASET_NAME}' yükleniyor...")
        # Hafıza problemini azaltmak için "streaming=True" denenebilir, 
        # ancak burada tüm datayı RAM'e alıyoruz.
        try:
            dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split="train") 
        except Exception as e:
            logging.error(f"❌ HATA (Veri Yükleme): Hugging Face yüklenemedi. Hata: {e}")
            return []

        documents = []
        for i, row in enumerate(dataset):
             # ... (Veri temizleme mantığı aynı) ...
            situation = row.get('situation')
            thoughts = row.get('thoughts')
            core_belief = row.get('core_belief_fine_grained') 
            is_valid = situation and str(situation).strip() not in ['N/A', ''] and \
                       thoughts and str(thoughts).strip() not in ['N/A', '']
            
            if is_valid:
                content = (
                    f"Durum: {situation}. "
                    f"Danışan Düşüncesi: {thoughts}. "
                    f"Çekirdek İnançlar: {core_belief}"
                )
                documents.append({"id": str(i), "content": content})
        
        logging.info(f"✅ **1. Veri Yükleme Tamamlandı:** Toplam {len(documents)} doküman yüklendi.")
        return documents

    # Referans projedeki setup mantığı (Pinecone Client)
    def _setup_pinecone(self):
        if not self.cfg.PINECONE_API_KEY: 
            logging.warning("⚠️ PINECONE_API_KEY eksik. Retrieval devre dışı.")
            return None
        try:
            logging.info(f"🔄 **3. Pinecone Bağlantısı:** Pinecone istemcisi başlatılıyor...")
            pc = Pinecone(api_key=self.cfg.PINECONE_API_KEY)
            index_name = self.cfg.PINECONE_INDEX_NAME
            
            if index_name not in pc.list_indexes().names:
                logging.warning(f"⚠️ **4a. İndeks Oluşturma:** '{index_name}' bulunamadı. Yeni indeks oluşturuluyor...")
                pc.create_index(
                    name=index_name,
                    dimension=self.cfg.EMBEDDING_DIM,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region=self.cfg.PINECONE_ENV)
                )
            
            logging.info("✅ **3. Pinecone Bağlantısı Başarılı.**")
            return pc.Index(index_name)
        except Exception as e:
            logging.error(f"❌ HATA (Pinecone): Başlatılamadı. {e}")
            return None

    # Referans projedeki yükleme mantığı
    def _load_dataset_to_pinecone(self):
        if not self.pinecone_index: return
        
        vector_count = self.pinecone_index.describe_index_stats().get("total_vector_count", 0)
        if vector_count > 0:
            logging.info(f"✅ **4a. İndeks Kontrolü Başarılı:** {vector_count} vektöre sahip. Yükleme ATLANDI.")
            return

        logging.info(f"🔄 **4b. Veriler Yükleniyor:** {len(self.documents)} vektör Pinecone'a yükleniyor...")
        
        vectors = []
        try:
            for i, doc in enumerate(tqdm(self.documents)):
                # Embedding'i oluştur ve listeye ekle
                vectors.append((
                    doc["id"], 
                    self.embedder.embed(doc["content"]), 
                    {"text": doc["content"]} # Metadata sadece text'i tutsun
                ))
                
                if len(vectors) >= self.cfg.BATCH_SIZE:
                    self.pinecone_index.upsert(vectors=vectors)
                    vectors = [] # Batch temizle
            
            if vectors: # Kalanları yükle
                self.pinecone_index.upsert(vectors=vectors)

            logging.info(f"✅ **4b. İndeksleme Tamamlandı.**")
        except Exception as e:
            logging.error(f"❌ HATA (Veri Yükleme/Upsert): {e}")

    # RAG sorgu mantığı (Referans projedeki gibi, manuel olarak)
    def get_answer(self, question: str):
        if not question:
            return "Lütfen bir soru girin."

        context = "Relevant context not found."
        
        try:
            if self.pinecone_index:
                # 1. Sorgu Embedding'i
                query_emb = self.embedder.embed(question)
                
                # 2. Pinecone Sorgusu
                res = self.pinecone_index.query(
                    vector=query_emb, 
                    top_k=self.cfg.K_RETRIEVAL, 
                    include_metadata=True
                )
                
                # 3. Bağlam Oluşturma (Sadece skor > 0.6 olanları alalım)
                relevant_texts = [
                    m['metadata'].get('text', '') 
                    for m in res['matches'] 
                    if m['score'] > 0.6
                ]
                
                if relevant_texts:
                    context = "\n---\n".join(relevant_texts)
                    logging.info("🔍 Pinecone'dan bağlam bulundu.")
                else:
                    logging.info("🔍 Pinecone'da yüksek skorlu bağlam bulunamadı.")
            
            # 4. LLM Yanıtı
            return self.gemini.generate(question, context)

        except RuntimeError as e: # Embedding hatası
            logging.error(f"❌ RAG Runtime Hatası: {e}")
            return "Hata: Embedding modelini kullanamıyorum. Sunucu loglarını kontrol edin."
        except Exception as e:
            logging.error(f"❌ RAG Genel Hatası: {e}")
            traceback.print_exc() 
            return "Sunucu Hatası: Sorgu yürütülürken beklenmeyen bir hata oluştu."

# --- 5. FLASK APP ---
app = Flask(__name__)

# Global asistan objesi (Gunicorn'da tekil kalacaktır)
try:
    cfg = Config()
    assistant = PsychotherapyAssistant(cfg)
    logging.info("✅ **5. RAG Zinciri Kurulumu Tamamlandı!** Bot kullanıma hazır.")
except Exception as startup_error:
    logging.error(f"\n!!!! KRİTİK BAŞLANGIÇ HATASI !!!!\nMesaj: {startup_error}")
    # Başarısız olursa 'assistant' değişkenini None bırakabiliriz veya bir Mock obje atayabiliriz.
    assistant = None 

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask_question():
    if assistant is None:
        return jsonify({"answer": "Error: RAG Chain initialization failed. Please check server logs for setup errors."}), 500
        
    data = request.get_json()
    query = data.get("question")
    
    if not query:
        return jsonify({"answer": "Please provide a question."}), 400

    try:
        logging.info(f"🔄 **Sorgu İşleniyor:** '{query[:50]}...' için RAG başlatıldı.")
        answer = assistant.get_answer(query)
        logging.info("✅ **Sorgu Tamamlandı:** Yanıt başarıyla oluşturuldu.")
        
        return jsonify({"answer": answer})

    except Exception as e:
        logging.error(f"❌ KRİTİK HATA (RAG Yürütme): Sorgu sırasında hata oluştu.")
        traceback.print_exc() 
        
        return jsonify({
            "answer": f"API veya Sunucu Hatası: İşlem sırasında beklenmeyen bir hata oluştu ({type(e).__name__})."
        }), 500


if __name__ == "__main__":
    # Render'ın verdiği PORT değişkenini al
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host="0.0.0.0", port=port)