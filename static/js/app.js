const chatMessages = document.getElementById('chat-messages');
const inputElement = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');

// =========================================================
// TEXTAREA OTO-BÜYÜTME (Auto-Resize)
// =========================================================

// Textarea'nın içeriğe göre otomatik olarak satır yüksekliğini ayarlayan fonksiyon
function autoResizeTextarea() {
    // Scroll Height (Kaydırma Yüksekliği), içeriğin kapladığı tüm yüksekliği verir.
    inputElement.style.height = 'auto'; 
    
    let newHeight = inputElement.scrollHeight;
    
    // Maksimum yüksekliği 150px ile sınırla
    if (newHeight > 150) {
        newHeight = 150;
    }
    
    inputElement.style.height = newHeight + 'px';
}

// Enter tuşuna basıldığında soru sormayı sağlar (SHIFT+ENTER satır atlar)
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault(); 
        askQuestion();
    }
}
// Event Listener'ı Textarea'ya bağlama
inputElement.addEventListener('input', autoResizeTextarea);


// =========================================================
// CHATBOT ANİMASYON VE MESAJ MANTIĞI
// =========================================================

// Yükleme sırasında gösterilecek "Yazıyor..." balonu
function createTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', 'bot', 'typing-indicator');

    const avatar = document.createElement('div');
    avatar.classList.add('avatar');
    avatar.innerHTML = '<i class="fas fa-brain"></i>';
    messageDiv.appendChild(avatar);

    const bubble = document.createElement('div');
    bubble.classList.add('message-bubble', 'typing-dots');
    bubble.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
    
    messageDiv.appendChild(bubble);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return messageDiv; // Kaldırmak için bu elementi döndürürüz
}


// Mesajı akışa ekleyen yardımcı fonksiyon
function appendMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);

    // Bot mesajlarına avatar ekle
    if (sender === 'bot') {
        const avatar = document.createElement('div');
        avatar.classList.add('avatar');
        avatar.innerHTML = '<i class="fas fa-brain"></i>'; // Bot simgesi
        messageDiv.appendChild(avatar);
    }

    const bubble = document.createElement('div');
    bubble.classList.add('message-bubble');

    if (sender === 'bot') {
        // Bot mesajları Markdown olarak işlenir
        if (typeof marked !== 'undefined') {
            bubble.innerHTML = marked.parse(text);
        } else {
            bubble.textContent = text;
        }
    } else {
        bubble.textContent = text;
    }

    messageDiv.appendChild(bubble);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return bubble;
}


// API'ye soruyu gönderme işlevi
async function askQuestion() {
    const query = inputElement.value.trim();

    if (!query) return;

    // 1. Kullanıcı mesajını ekle ve textarea'yı resetle
    appendMessage(query, 'user');
    inputElement.value = '';
    inputElement.style.height = '45px'; // Textarea'yı varsayılan boyuta küçült

    // 2. Yükleme ve durum göstergelerini ayarla
    sendButton.disabled = true;
    sendButton.innerHTML = '<i class="fas fa-spinner"></i>'; // Dönen simge
    
    // 3. Yazıyor animasyonunu ekle
    const typingIndicator = createTypingIndicator();


    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: query })
        });

        const data = await response.json();

        // 4. Yazıyor animasyonunu kaldır
        chatMessages.removeChild(typingIndicator);


        if (response.ok) {
            appendMessage(data.answer, 'bot');
        } else {
            appendMessage(`[HATA] Sunucu yanıtı işleyemedi: ${data.answer}`, 'bot');
        }

    } catch (error) {
        // Hata durumunda yükleme göstergelerini temizle
        if (chatMessages.lastChild && chatMessages.lastChild.classList.contains('typing-indicator')) {
             chatMessages.removeChild(chatMessages.lastChild);
        }
        appendMessage("API bağlantı hatası oluştu. Lütfen sunucu loglarını kontrol edin.", 'bot');
        console.error('Fetch Hatası:', error);
    } finally {
        // Butonu tekrar etkinleştir
        sendButton.disabled = false;
        sendButton.innerHTML = '➔'; 
        inputElement.focus();
    }
}
// HTML'den erişilebilmesi için fonksiyonları window objesine bağlıyoruz
window.askQuestion = askQuestion;
window.handleKeyPress = handleKeyPress;