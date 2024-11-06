from flask import Flask, request, render_template, jsonify, url_for, redirect
from io import BytesIO
import dlib
import cv2
import numpy as np
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from datetime import datetime

app = Flask(__name__)

# Carregar o classificador de faces pré-treinado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
app.secret_key = 'sua_chave_secreta'  # Defina uma chave secreta para segurança
# Inicialize o serializer
serializer = URLSafeTimedSerializer(app.secret_key)

@app.route('/')
def index():
    return render_template('index.html')  # Página de upload

def gerar_link_temporario(validade_segundos=5):
    # Gera o token com e a validade
    data_hora_atual = datetime.today()
    token = serializer.dumps(f"{data_hora_atual}")
    # Cria o link usando o token
    return url_for('verificar_link', token=token, _external=True)

@app.route('/verificar/<token>')
def verificar_link(token):
    try:
        # Tenta carregar o token, verificando se ainda é válido
        data_hora_atual = serializer.loads(token, max_age=5)  # Validade em segundos
        return render_template('index.html')  # Página de upload
    except SignatureExpired:
        return "Link expirado!"
    except BadSignature:
        return "Link inválido!"

@app.route('/gerar-link/')
def gerar():
    # Chama a função para gerar o link com validade de 1 hora
    link = gerar_link_temporario(validade_segundos=5)
    return f"Seu link temporário é: {link}"


@app.route('/comparar-rostos', methods=['POST'])
def comparar_rostos():
    # Verificar se as imagens foram enviadas
    if 'filea' not in request.files or 'fileb' not in request.files:
        return jsonify({"error": "Por favor, envie duas imagens para comparação."}), 400

    # Obter as imagens enviadas
    image1_file = request.files['filea']
    image2_file = request.files['fileb']

    # Obtenha os descritores faciais para duas imagens
    face_descriptor1 = get_face_descriptor(image2_file.filename)
    face_descriptor2 = get_face_descriptor(image1_file.filename)

    # Verifique se ambos os descritores foram extraídos
    if face_descriptor1 is not None and face_descriptor2 is not None:
        # Calcule a distância euclidiana entre os descritores
        distance = np.linalg.norm(face_descriptor1 - face_descriptor2)

        # Definir um limiar para decidir se são a mesma pessoa (exemplo: 0.6)
        if distance < 0.6:
            return jsonify({"result": True, "message": "As faces são da mesma pessoa.", "diferenca_face": distance}), 200
        else:
            return jsonify({"result": False, "message": "As faces são de pessoas diferentes.","diferenca_face": distance}), 400
    else:
        return jsonify({"result": False, "message": "Não foi possível obter descritores faciais para ambas as imagens."}), 400

def get_face_descriptor(image_path):
  
    # Carregar o detector de faces e o preditor de pontos de referência
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Baixe e especifique o caminho
    face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")  # Modelo de reconhecimento de rosto

    # Carregar a imagem
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectar a face
    faces = detector(gray)
    if len(faces) == 0:
        print("Nenhuma face detectada.")
        return None

    # Assumir que a primeira face detectada é a que queremos
    face = faces[0]

    # Extrair os pontos de referência (landmarks)
    shape = predictor(gray, face)

    # Obter o descritor facial
    face_descriptor = np.array(face_rec_model.compute_face_descriptor(img, shape))

    return face_descriptor

if __name__ == '__main__':
    app.run(debug=True)
