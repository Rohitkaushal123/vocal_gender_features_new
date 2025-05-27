from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from utils.feature_extract import extract_features
import pickle
import numpy as np
import os


with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('selectk.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods = ['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        audio_file = request.files.get('audio')
        if audio_file:
            filepath = os.path.join(UPLOAD_FOLDER, secure_filename(audio_file.filename))
            audio_file.save(filepath)

            features = extract_features(filepath).reshape(1, -1)
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)

            result = 'Male' if prediction[0] == 1 else 'Female'
            return render_template('index.html', result=result)

    # Make sure GET or missing file returns something
    return render_template('index.html', result=None)

        


if __name__ == '__main__':
    app.run(debug=True)

