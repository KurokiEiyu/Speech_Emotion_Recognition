from flask import Flask, render_template, request
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
from keras.models import load_model

app = Flask(__name__)

# Load model
model = load_model('Dataset/model.h5')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Global variables for audio recording
RECORD_SECONDS = 3
fs = 22050

@app.route('/')
def index():
    return render_template('indexxDomain.html')

@app.route('/record', methods=['POST'])
def record():
    # Record audio
    audio = sd.rec(int(RECORD_SECONDS * fs), samplerate=fs, channels=1)
    sd.wait()

    # Save audio file
    filepath = 'static/audio.wav'
    sf.write(filepath, audio, fs)

    # Extract features from audio
    features = extract_mfcc(filepath)

    # Reshape features
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=-1)

    # Predict emotion
    prediction = model.predict(features)
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]
    
    # Calculate accuracy
    accuracy = np.max(prediction) * 100  # Convert to percentage

    return {'emotion': emotion, 'accuracy': accuracy}

def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

if __name__ == '__main__':
    app.run(debug=True)