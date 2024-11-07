from flask import Flask, request, render_template, jsonify, url_for
from io import BytesIO
import dlib, json, cv2, numpy as np, sqlite3, os
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Diretório onde as imagens serão salvas
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite de 16 MB para uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Carregar o classificador de faces pré-treinado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
app.secret_key = 'sua_chave_secreta'  # Defina uma chave secreta para segurança
# Inicialize o serializer
serializer = URLSafeTimedSerializer(app.secret_key)
# Conecta ao banco de dados (ou cria um novo arquivo .db)
conexao = sqlite3.connect('banco.db')
# Cria um cursor para executar comandos SQL
cursor = conexao.cursor()
# Cria uma tabela chamada "usuarios"
cursor.execute('''
CREATE TABLE IF NOT EXISTS usuarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nome TEXT NOT NULL,
    login TEXT NOT NULL,
    email TEXT NOT NULL,
    date_cadastro TIMESTAMP NOT NULL,
    fotos TEXT NOT NULL,
    nivel INTEGER
)
''')
# Confirma a criação da tabela
conexao.commit()
conexao.close()

@app.route('/')
def index():
    return render_template('index.html')  # Página de upload

@app.route('/cadastro', methods=['POST'])
def cadastro():
    uploaded_files = request.files.getlist("images")  # Pega todos os arquivos com o nome "images"
    user_login = request.form.get("login")
    user_email = request.form.get("email")
    user_nome = request.form.get("nome")
    file_paths = []
    erro = ""
    if user_login == "":
        erro = "Login invalido."
    if user_email == "":
        erro = "E-mail invalido."
    if user_nome == "":
        erro = "Nome invalido."
     # Cria uma nova conexão para cada chamada
    conexao = sqlite3.connect('banco.db')
    cursor = conexao.cursor()
    cursor.execute("SELECT * FROM usuarios WHERE login = ?", (user_login,))
    # Busca o resultado
    userFind = cursor.fetchone()
    if userFind:
        erro = f"Usuário já existe: {user_login}"
    for file in uploaded_files:
        if file and file.filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Adiciona o login do usuário ao nome do arquivo
            filename = secure_filename(f"{user_login}_{file.filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)
        else:
            erro = "Imagens contem o formato invalido."
    data_hora_atual = datetime.today()
    # Converte file_paths para JSON
    file_paths_json = json.dumps(file_paths)
    # Inserir dados
    cursor.execute('''
    INSERT INTO usuarios (nome, email, login, date_cadastro, fotos)
    VALUES (?, ?, ?, ?, ?)
    ''', (user_nome, user_email, user_login, data_hora_atual, file_paths_json))
    conexao.commit()
    # Fecha a conexão
    conexao.close()
    if erro != "":
        return render_template('erro.html', mensagem=erro)
    else:
        return render_template('success.html', mensagem='Usuário Cadastrado')
    
def gerar_link_temporario(validade_segundos=3600):
    # Gera o token com e a validade
    data_hora_atual = datetime.today()
    token = serializer.dumps(f"{data_hora_atual}")
    # Cria o link usando o token
    return url_for('verificar_link', token=token, _external=True)

@app.route('/verificar/<token>')
def verificar_link(token):
    try:
        # Tenta carregar o token, verificando se ainda é válido
        data_hora_atual = serializer.loads(token, max_age=3600)  # Validade em segundos
        return render_template('form.html')  # Página de upload
    except SignatureExpired:
        return "Link expirado!"
    except BadSignature:
        return "Link inválido!"

@app.route('/gerar-link/')
def gerar():
    # Chama a função para gerar o link com validade de 1 hora
    link = gerar_link_temporario(validade_segundos=3600)
    return link

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
