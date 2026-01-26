from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import os
import random
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)
# Maksimum dosya boyutu (Örn: 16 MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
CORS(app)

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
    # 1. Dosya Kontrolü
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "Dosya yüklenmedi!"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"success": False, "error": "Dosya seçilmedi!"}), 400

    # Dosyayı Geçici Kaydet
    filename = secure_filename(file.filename)
    unique_name = f"upload_{int(time.time())}_{random.randint(1000,9999)}_{filename}"
    file.save(unique_name)
    
    try:
        # 2. Analiz (Librosa) - İlk 60 saniye
        y, sr = librosa.load(unique_name, duration=60)
        
        # BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # TON (Key)
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
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
