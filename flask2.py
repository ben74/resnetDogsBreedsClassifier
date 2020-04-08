#
import os,sys,flask
from flask import Flask, render_template, url_for, request
app = Flask(__name__)
@app.route('/')
def home():
    return """<form method='post' enctype='multipart/form-data' action='/'>Local filepath : <input name='filepath' value='pilou.jpg'><br>Or upload file here<input type='file' name='dogPicture'><br><input type=submit value=submit></form>""";
    
@app.route('/', methods=['POST'])
def post():
    if (request.method == 'POST'):
        if 'filepath' in request.form.keys():
            if(len(request.form['filepath'])>1):
                return '1>'+request.form['filepath']
            
        if 'dogPicture' in request.files.keys():
            file = request.files['dogPicture']
            return '2>'+file.filename
    return 'ko'
app.run(host='0.0.0.0', port=8080, debug=True)

#curl -k 127.0.0.1:8080 -F "input=unexpected"
#curl -k 127.0.0.1:8080 -F "filepath=pilou.jpg"
#curl -k 127.0.0.1:8080 -F "dogPicture=@pilou.jpg"

