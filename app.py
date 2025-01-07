from flask import Flask, render_template, request
import os
import numpy as np
import librosa
from keras.models import load_model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Load the trained model
model = load_model('Dataset/model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Memeriksa apakah file suara telah diunggah
    if 'file' not in request.files:
        return 'File not uploaded'
    
    file = request.files['file']
    
    # Simpan file suara yang diunggah
    audio_path = 'uploaded_audio.wav'
    file.save(audio_path)
    
    # Melakukan ekstraksi fitur MFCC pada file suara yang diunggah
    def extract_mfcc(filename):
        y, sr = librosa.load(filename, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfcc

    mfcc = extract_mfcc(audio_path)
    X = np.expand_dims(mfcc, axis=0)
    X = np.expand_dims(X, axis=-1)

    # Melakukan prediksi menggunakan model Speech Emotion Recognition
    predictions = model.predict(X)
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
    predicted_emotion = emotion_labels[np.argmax(predictions)]

    # Menghapus file suara yang diunggah
    os.remove(audio_path)

    return render_template('index.html', emotion=predicted_emotion)

if __name__ == '__main__':
    app.run(debug=True)