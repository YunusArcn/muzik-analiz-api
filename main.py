from flask import Flask, request, jsonify
from flask_cors import CORS
import yt_dlp
import librosa
import numpy as np
import os
import ssl
import random
import time

# SSL Hatasını Önleme (Sunucular için kritik)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

app = Flask(__name__)
CORS(app)

# --- MÜZİK TEORİSİ FONKSİYONLARI ---
def get_chord_from_chroma(chroma, time_idx):
    # Basit Akor Şablonları
    templates = {
        'C': [1,0,0,0,1,0,0,1,0,0,0,0], 'Cm': [1,0,0,1,0,0,0,1,0,0,0,0],
        'C#': [0,1,0,0,0,1,0,0,1,0,0,0], 'C#m': [0,1,0,0,1,0,0,0,1,0,0,0],
        'D': [0,0,1,0,0,0,1,0,0,1,0,0], 'Dm': [0,0,1,0,0,1,0,0,0,1,0,0],
        'D#': [0,0,0,1,0,0,0,1,0,0,1,0], 'D#m': [0,0,0,1,0,0,1,0,0,0,1,0],
        'E': [0,0,0,0,1,0,0,0,1,0,0,1], 'Em': [0,0,0,0,1,0,0,1,0,0,0,1],
        'F': [1,0,0,0,0,1,0,0,0,1,0,0], 'Fm': [1,0,0,0,0,1,0,0,1,0,0,0],
        'F#': [0,1,0,0,0,0,1,0,0,0,1,0], 'F#m': [0,1,0,0,0,0,1,0,0,1,0,0],
        'G': [0,0,1,0,0,0,0,1,0,0,0,1], 'Gm': [0,0,1,0,0,0,0,1,0,0,1,0],
        'G#': [1,0,0,1,0,0,0,0,1,0,0,0], 'G#m': [0,0,0,1,0,0,0,0,1,0,0,1],
        'A': [0,1,0,0,1,0,0,0,0,1,0,0], 'Am': [1,0,0,0,1,0,0,0,0,1,0,0],
        'A#': [0,0,1,0,0,1,0,0,0,0,1,0], 'A#m': [0,0,1,0,0,1,0,0,0,0,1,0],
        'B': [0,0,0,1,0,0,1,0,0,0,0,1], 'Bm': [0,0,0,1,0,0,1,0,0,0,0,1],
    }
    
    col = chroma[:, time_idx]
    max_score = -1
    best_chord = "-"
    
    for chord, template in templates.items():
        score = np.dot(col, template)
        if score > max_score:
            max_score = score
            best_chord = chord
    return best_chord

@app.route('/analiz-et', methods=['POST'])
def analiz_et():
    data = request.json
    youtube_url = data.get('url')
    
    if not youtube_url:
        return jsonify({"success": False, "error": "Lütfen bir YouTube linki gönderin."}), 400

    # Benzersiz dosya adı oluştur
    filename = f"temp_{int(time.time())}_{random.randint(1000,9999)}"
    
    try:
        # --- KRİTİK AYARLAR: YouTube Bot Korumasını Aşma ---
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': filename,
            'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3','preferredquality': '128'}],
            'noplaylist': True,
            
            # 1. Önbelleği KAPAT (Eski hataları hatırlamasın)
            'cachedir': False, 
            
            # 2. Android TV Taklidi Yap (En az güvenlik duvarı bundadır)
            'extractor_args': {'youtube': {'player_client': ['android_tv']}},
            
            # 3. Sertifika Hatalarını Yoksay
            'nocheckcertificate': True,
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
            
        file_path = f"{filename}.mp3"

        # --- SES ANALİZİ (Librosa) ---
        # Hız için sadece ilk 60 saniyeyi analiz ediyoruz
        y, sr = librosa.load(file_path, duration=60)
        
        # 1. BPM (Tempo)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # 2. TON (Key)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_idx = np.argmax(np.sum(chroma, axis=1))
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        detected_key = notes[key_idx]

        # 3. AKORLAR (Saniye Saniye)
        akorlar = []
        hop_length = 512
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Her 2 saniyede bir örnek al
        for t in range(0, int(duration), 2): 
            frame_idx = librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
            if frame_idx < chroma_cens.shape[1]:
                chord = get_chord_from_chroma(chroma_cens, frame_idx)
                akorlar.append({"zaman": f"{t//60}:{t%60:02d}", "akor": chord})

        # Temizlik (Dosyayı sil)
        if os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({
            "success": True,
            "bpm": round(tempo),
            "ton": detected_key,
            "akorlar": akorlar
        })

    except Exception as e:
        # Hata olsa bile dosyayı silmeye çalış
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
