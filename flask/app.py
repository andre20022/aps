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
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(30, 30))
    if len(faces) == 0:
        return jsonify({'error': 'No faces'}), 400
    
    # Desenhar retângulos ao redor das faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Converter a imagem com as faces marcadas de volta para um formato que pode ser enviado no Flask
    _, img_encoded = cv2.imencode('.png', img)
    img_bytes = img_encoded.tobytes()

    return send_file(BytesIO(img_bytes), mimetype='image/png')

@app.route('/comparar-rostos', methods=['POST'])
def comparar_rostos():
    # Verificar se as imagens foram enviadas
    if 'filea' not in request.files or 'fileb' not in request.files:
        return jsonify({"error": "Por favor, envie duas imagens para comparação."}), 400

    # Obter as imagens enviadas
    image1_file = request.files['filea']
    image2_file = request.files['fileb']

    # Converter as imagens para arrays NumPy
    image1 = np.asarray(bytearray(image1_file.read()), dtype=np.uint8)
    image2 = np.asarray(bytearray(image2_file.read()), dtype=np.uint8)

    # Decodificar as imagens
    image1 = cv2.imdecode(image1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imdecode(image2, cv2.IMREAD_GRAYSCALE)

    # Calcular os histogramas
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])

    # Normalizar os histogramas
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    # Comparar os histogramas
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    # Definir limite de similaridade
    if similarity > 0.8:
        result = "As imagens são semelhantes."
    else:
        result = "As imagens são diferentes."

    return jsonify({"similaridade": similarity, "resultado": result})


if __name__ == '__main__':
    app.run(debug=True)
