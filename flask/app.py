from flask import Flask, request, render_template, jsonify, send_file
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Carregar o classificador de faces pré-treinado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')  # Página de upload

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Converter imagem para o formato que o OpenCV pode processar
    img = Image.open(file.stream)
    img = np.array(img)

    # Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectar faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    print(faces);

    # Desenhar retângulos ao redor das faces
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # # Converter a imagem com as faces marcadas de volta para um formato que pode ser enviado no Flask
    # _, img_encoded = cv2.imencode('.png', img)
    # img_bytes = img_encoded.tobytes()

    # return send_file(BytesIO(img_bytes), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)