from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import soundfile as sf
import os
import random
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)
# Maksimum dosya boyutu: 10 MB (Limiti de düşürdük)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return "Müzik Analiz API (Lite Mode) Çalışıyor! 🚀", 200

# --- MÜZİK TEORİSİ (Akor Bulucu) ---
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
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "Dosya yüklenmedi!"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "Dosya seçilmedi!"}), 400

    filename = secure_filename(file.filename)
    unique_name = f"upload_{int(time.time())}_{random.randint(1000,9999)}_{filename}"
    file.save(unique_name)
    
    try:
        # --- LITE MODE OPTİMİZASYONU ---
        # 1. Soundfile ile açıyoruz (RAM dostu)
        y, sr = sf.read(unique_name)
        
        # 2. Mono yapıyoruz
        if y.ndim > 1:
            y = np.mean(y, axis=1)
            
        # 3. SÜREYİ KISALTTIK: Sadece ilk 10 saniye! (İşlemciyi kurtarmak için)
        max_frames = 10 * sr
        if len(y) > max_frames:
            y = y[:max_frames]

        # 4. AĞIR İŞLEMİ DEĞİŞTİRDİK:
        # chroma_cqt yerine chroma_stft kullanıyoruz. CQT çok ağırdır, STFT çok hafiftir.
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        # BPM (Hala gerekli)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # TON (Key)
        key_idx = np.argmax(np.sum(chroma, axis=1))
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        detected_key = notes[key_idx]

        # AKORLAR
        akorlar = []
        # Hop length'i artırdık (Daha az veri işlesin diye)
        hop_length = 1024 
        
        # Cens yerine yine hafif olan stft kullanıyoruz
        chroma_for_chords = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        duration = len(y) / sr
        
        for t in range(0, int(duration), 2): 
            frame_idx = librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
            if frame_idx < chroma_for_chords.shape[1]:
                chord = get_chord_from_chroma(chroma_for_chords, frame_idx)
                akorlar.append({"zaman": f"{t//60}:{t%60:02d}", "akor": chord})

        if os.path.exists(unique_name):
            os.remove(unique_name)

        return jsonify({
            "success": True,
            "bpm": round(tempo),
            "ton": detected_key,
            "akorlar": akorlar
        })

    except Exception as e:
        if os.path.exists(unique_name):
            os.remove(unique_name)
        print(f"HATA: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
