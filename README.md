# 💬 AI Duygusal Rehber: BDT Odaklı RAG Chatbot

Bu proje, Hugging Face'teki profesyonel bir psikoterapi veri seti üzerine kurulu, **Retrieval-Augmented Generation (RAG)** mimarisini kullanan, Bilişsel Davranışçı Terapi (BDT) ilkelerine dayalı empatik bir sohbet robotudur.

Amaç, Büyük Dil Modelinin (LLM) genel bilgiler yerine, yüksek kaliteli terapötik kayıtlardan edindiği bağlamı kullanarak daha güvenilir, tutarlı ve yapılandırılmış (Empati, Analiz, Öneri) yanıtlar üretmektir.

---

## ⚠️ Etik Sorumluluk Reddi (Çok Önemli)

Bu Yapay Zeka Duygusal Rehber, **lisanslı bir terapist veya psikolog DEĞİLDİR**. Yalnızca BDT prensiplerine dayalı destekleyici rehberlik sunmak üzere tasarlanmıştır. Ciddi mental sağlık sorunları veya kriz durumlarında, profesyonel bir ruh sağlığı uzmanına başvurulmalıdır.

---

## 🛠️ Teknoloji Yığını

| Bileşen | Teknoloji | Rolü |
| :--- | :--- | :--- |
| **Orkestrasyon**| `Flask`, `LangChain` | Uygulama çatısı ve RAG zincir yönetimi. |
| **Bilgi Kaynağı (Dataset)**| `Psychotherapy-LLM/CBT-Bench` (`core_fine_test`) | RAG için kullanılan, danışan durumları ve çekirdek inançları içeren profesyonel veri seti. |
| **Vektör Veritabanı**| `Pinecone` | Yüksek boyutlu vektörlerin depolanması ve hızlı anlamsal arama (Retrieval) yapılması. |
| **Embedding Modeli**| `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) | Terapötik metinleri sayısal vektörlere dönüştürme. |
| **Büyük Dil Modeli (LLM)**| `Gemini 2.5 Flash` | Çekilen bağlamı okuyarak nihai, yapılandırılmış (3 bölümlü) yanıtı oluşturma. |

---

## ⚙️ RAG Mimarisi ve Akış Şeması

Bu chatbot, geleneksel LLM'lerin aksine, yanıtını her zaman dış bir bilgi tabanına dayandırır:

1.  **Veri Hazırlığı (Indexing):** Proje başlangıcında, CBT-Bench veri setinden `situation`, `thoughts` ve `core_belief_fine_grained` kolonları birleştirilir (Chunking) ve vektörlere dönüştürülerek Pinecone'a yüklenir. **(Koşullu Yükleme: Vektörler zaten varsa yükleme atlanır.)**
2.  **Sorgu (User Input):** Kullanıcı duygusal durumunu yazar.
3.  **Alma (Retrieval):** Kullanıcının sorgusu vektörleştirilir ve Pinecone'da anlamsal olarak en alakalı 3 terapötik kayıt (`k=3`) çekilir.
4.  **Oluşturma (Generation):** Çekilen kayıtlar, katı bir **Sistem İstem Şablonu** ile birlikte Gemini LLM'e gönderilir.
5.  **Yanıt:** Gemini, kendi genel BDT bilgisini çekilen **çekirdek inançlarla** birleştirerek, Empati, Analiz ve Eylem Önerisi içeren nihai, yapılandırılmış yanıtı üretir.

---

## 🚀 Kurulum ve Çalıştırma

### Ön Koşullar

1.  **Python 3.8+**
2.  **API Anahtarları:**
    * **Pinecone API Key:** Pinecone hesabınızdan alın.
    * **Google Gemini API Key:** Google AI Studio'dan alın.
3.  **Pinecone Index:** Kod, `psychotherapy-rag` adında 384 boyutlu (all-MiniLM-L6-v2'ye uygun) bir Serverless Index oluşturur.

### Adım 1: Klonlama ve Ortam Kurulumu

```bash
# Projeyi klonlayın
git clone [REPOSITORY_URL]
cd [REPOSITORY_NAME]

# Sanal ortamı oluşturun ve aktive edin
python -m venv venv
source venv/bin/activate  # Linux/macOS
# .\venv\Scripts\activate # Windows

---

# Gerekli kütüphaneleri kurun
```bash
pip install -r requirements.txt

---

### Adım 2: Ortam Değişkenlerini Ayarlama
Projeyi çalıştırmadan önce, klonladığınız dizinde bir .env dosyası oluşturun ve tüm API anahtarlarınızı buraya ekleyin:

```bash
# .env Dosyası İçeriği
PINECONE_API_KEY="[BURAYA_PINECONE_ANAHTARINIZI_EKLEYIN]"
GOOGLE_API_KEY="[BURAYA_GEMINI_API_ANAHTARINIZI_EKLEYIN]"

# Index adı varsayılan olarak 'psychotherapy-rag'dır.
PINECONE_INDEX="psychotherapy-rag"

---

### Adım 3: Uygulamayı Başlatma ve İndeksleme
Uygulama ilk kez başlatıldığında, veri setini otomatik olarak Hugging Face'ten çekecek ve Pinecone'a yükleyecektir. İndeks zaten doluysa bu adım otomatik olarak atlanır.

```bash
python app.py

---

### Adım 4: Chatbot'u Kullanma
Uygulama çalıştıktan sonra:

Tarayıcınızda local adresinize gidin.

Chat arayüzünde duygu ve sorunlarınızı yazın.

Bot, Pinecone'daki terapötik kayıtlara dayanarak yapılandırılmış (Empati, Analiz, Öneri) yanıtını sunacaktır.

🧩 Dosya Yapısı
Projenin temel klasör yapısı şöyledir:

rag-chatbot-psychotherapy/
├── app.py                     # Ana Flask uygulaması ve RAG zincirinin kurulduğu yer
├── .env                       # API Anahtarları (Yerel, Git'e yüklenmez)
├── requirements.txt           # Gerekli Python kütüphaneleri
├── README.md                  # Bu belge
├── static/
│   ├── css/
│   │   └── style.css          # Modern chat arayüzü stilleri
│   └── js/
│       └── app.js             # Yazım efekti ve AJAX (fetch) mantığı
└── templates/
    └── index.html             # Chat arayüzünün ana HTML şablonu
🤝 Katkıda Bulunma
Pull request'ler (Çekme İstekleri) ve issue'lar (Sorunlar) memnuniyetle karşılanır. Bu proje, yeni terapi protokolleri (örneğin ACT, DBT) için genişletilebilir yapıdadır.

Lisans: Bu proje MIT Lisansı ile yayınlanmıştır.
