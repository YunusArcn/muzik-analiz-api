from flask import Flask, request, jsonify
from flask_cors import CORS
import yt_dlp
import librosa
import numpy as np
import os
import ssl
import random
import time
import sys

# SSL Hatasını Önleme
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Sürüm kontrolü (Loglarda görmek için)
print(f"yt-dlp sürümü: {yt_dlp.version.__version__}")

app = Flask(__name__)
CORS(app)

# --- MÜZİK TEORİSİ FONKSİYONLARI ---
def get_chord_from_chroma(chroma, time_idx):
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

    filename = f"temp_{int(time.time())}_{random.randint(1000,9999)}"
    
    try:
        # --- iOS MASKESİ (iPhone Taklidi) ---
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': filename,
            'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3','preferredquality': '128'}],
            'noplaylist': True,
            
            # 1. Cache Yok (Hafızasız Mod)
            'cachedir': False, 
            
            # 2. iOS İstemcisi (iPhone gibi görünür)
            'extractor_args': {'youtube': {'player_client': ['ios']}},
            
            # 3. Ekstra Ayarlar
            'nocheckcertificate': True,
            'quiet': True,
            'no_warnings': True,
            # Mobil User Agent
            'user_agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
            
        file_path = f"{filename}.mp3"

        # --- SES ANALİZİ ---
        y, sr = librosa.load(file_path, duration=60)
        
        # BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # TON
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_idx = np.argmax(np.sum(chroma, axis=1))
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        detected_key = notes[key_idx]

        # AKORLAR
        akorlar = []
        hop_length = 512
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
        duration = librosa.get_duration(y=y, sr=sr)
        
        for t in range(0, int(duration), 2): 
            frame_idx = librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
            if frame_idx < chroma_cens.shape[1]:
                chord = get_chord_from_chroma(chroma_cens, frame_idx)
                akorlar.append({"zaman": f"{t//60}:{t%60:02d}", "akor": chord})

        # Temizlik
        if os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({
            "success": True,
            "bpm": round(tempo),
            "ton": detected_key,
            "akorlar": akorlar
        })

    except Exception as e:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
