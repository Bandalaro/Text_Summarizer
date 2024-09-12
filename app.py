import warnings
warnings.filterwarnings("ignore")
from flask import Flask, render_template, Response, request, redirect, url_for, session, flash
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from summarizer import Summarizer
import moviepy.editor as mp
from time import sleep, time
import json
import requests
from googletrans import Translator
from nltk.tokenize import sent_tokenize, word_tokenize
import speech_recognition as sr
from newspaper import Article
import os
from flask_mongoengine import MongoEngine
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)
app.config['MONGODB_SETTINGS'] = {
    'db': 'auth_db',
    'host': 'localhost',
    'port': 27017
}
app.config['SECRET_KEY'] = 'your_secret_key'
db = MongoEngine()
db.init_app(app)

def load_models():
    global model, tokenizer, bert_model
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')
        tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
        bert_model = Summarizer()
        print('BART and BERT models downloaded successfully')
    except Exception as e:
        print(f"Error loading models: {e}")
        exit(1)  # Exit if models fail to load

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('You need to log in first.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

class User(db.Document):
    username = db.StringField(unique=True, required=True)
    password = db.StringField(required=True)

API_key = "057a0adabc0e4d25ba5e9db2d155c8b0"
headers = {'authorization': API_key, 'content-type': 'application/json'}
upload_endpoint = 'https://api.assemblyai.com/v2/upload'
transcription_endpoint = "https://api.assemblyai.com/v2/transcript"    

def upload(file_path):
    def read_audio(file_path):
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(5_242_880)
                if not data:
                    break
                yield data
    upload_response = requests.post(upload_endpoint, headers=headers, data=read_audio(file_path))
    return upload_response.json().get('upload_url')

def transcribe(upload_url): 
    json_data = {"audio_url": upload_url, "auto_chapters": True, "iab_categories": True}
    response = requests.post(transcription_endpoint, json=json_data, headers=headers)
    transcription_id = response.json()['id']
    return transcription_id

def get_result(transcription_id, timeout=300): 
    current_status = "queued"
    endpoint = f"https://api.assemblyai.com/v2/transcript/{transcription_id}"
    start_time = time()
    while current_status not in ("completed", "error"):
        if time() - start_time > timeout:
            return None  # Timeout
        response = requests.get(endpoint, headers=headers)
        current_status = response.json().get('status', 'error')
        if current_status in ("completed", "error"):
            return response.json()
        sleep(10)

def post_process_summary(summary):
    sentences = sent_tokenize(summary)
    seen_sentences = set()
    unique_sentences = []
    for sent in sentences:
        if sent not in seen_sentences:
            unique_sentences.append(sent)
            seen_sentences.add(sent)
    return ' '.join(unique_sentences)

def handle_voice_input(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        command = recognizer.recognize_google(audio)
        return command
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition; {e}"
    except Exception as e:
        return str(e)

def extract_text_from_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
@login_required
def index():
    return render_template("index.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.objects(username=username).first()
        if user and check_password_hash(user.password, password):
            session['username'] = username
            return redirect(url_for('index'))
        flash('Invalid username or password')
    return render_template("login.html")

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='sha256')
        existing_user = User.objects(username=username).first()
        if existing_user:
            flash('Username already exists')
        else:
            User(username=username, password=hashed_password).save()
            flash('Signup successful! Please log in.')
            return redirect(url_for('login'))
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route("/audio", methods=['POST'])
@login_required
def audio():
    try:
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if not allowed_file(file.filename):
            return "Unsupported file type", 400
        upload_dir = "uploads/"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)
        upload_url = upload(file_path)
    except Exception as e:
        print("Error occurred:", e)
        return str(e), 400

    transcription_id = transcribe(upload_url)
    response = get_result(transcription_id)
    if not response:
        return "Transcription failed", 500

    summary = response.get("text", "")[0:500]
    headline = response.get("chapters", [{}])[0].get('headline', '')

    try:
        language = str(request.form["lang1"])
        translator = Translator()
        summarynew = translator.translate(summary, dest=language)
        head_new = translator.translate(headline, dest=language)
        headline = head_new.text
        summary = summarynew.text
    except:
        pass    

    themes = []
    categories = response.get('iab_categories_result', {}).get('summary', [])
    for cat in categories:
        themes.append(cat)

    return render_template("result2.html", summary=summary, headline=headline, themes=themes)

@app.route("/model", methods=['POST'])
@login_required
def model():
    try:
        text = str(request.form['text'])
        style = request.form.get('style', 'formal')
        if not text.strip():
            return "No text provided", 400
        prompt = {
            'formal': "summarize: ",
            'informal': "simplify: ",
            'technical': "detailed summarize: "
        }.get(style, "summarize: ")
        max_input_length = tokenizer.model_max_length
        tokens_input = tokenizer.encode(prompt + text, return_tensors='pt', max_length=max_input_length, truncation=True)
        print(f"Tokenized input length: {tokens_input.size(1)} / {max_input_length}")
        if tokens_input.size(1) > max_input_length:
            return f"Input is too long. Please reduce the length of your text. Max tokens allowed: {max_input_length}", 400
        chunk_size = 512
        tokens = word_tokenize(text)
        if len(tokens) > chunk_size:
            text_chunks = [' '.join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
            summaries = []
            for chunk in text_chunks:
                tokens_input = tokenizer.encode(prompt + chunk, return_tensors='pt', max_length=max_input_length, truncation=True)
                summary_ids = model.generate(tokens_input, min_length=100, max_length=200, num_beams=4, early_stopping=True)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summaries.append(summary)
            summary = ' '.join(summaries)
        else:
            summary_ids = model.generate(tokens_input, min_length=100, max_length=200, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        ext_summary = bert_model(text, ratio=0.5)
        summary = post_process_summary(summary)
        voice_feedback = request.files.get('voice_feedback')
        feedback_command = handle_voice_input(voice_feedback) if voice_feedback else None
        if feedback_command:
            print(f"Voice feedback command: {feedback_command}")
        try:
            language = str(request.form.get("lang", ""))
            if language:
                translator = Translator()
                summarynew = translator.translate(summary, dest=language)
                summary = summarynew.text
        except Exception as e:
            print(f"Translation failed: {e}")
        return render_template("result.html", summary=summary, ext_summary=ext_summary, feedback_command=feedback_command)
    except Exception as e:
        print(f"Error occurred: {e}")
        return str(e), 500

@app.route("/url", methods=['POST'])
@login_required
def url():
    url = request.form['url']
    extracted_text = extract_text_from_url(url)
    summary = bert_model(extracted_text, ratio=0.5)
    return render_template("result3.html", summary=summary)

if __name__ == '__main__':
    load_models()
    app.run(debug=True, port=5000)
