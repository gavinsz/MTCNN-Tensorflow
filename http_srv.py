from flask import Flask
from flask import request
from flask import send_file
from werkzeug import secure_filename
import json
import detect
import os

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = 'tmp'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        #f = request.files['file']
        #print('upload file %s'%(f.filename))
        #f.save('tmp/tmp.jpg')
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print('save_path=%s'%(save_path))
            file.save(save_path)
        
            bb, _=detect.face_detector.detect(save_path)
        return json.dumps(bb.tolist())
        return send_file('data/tmp_cropped.png')
    return 'Hello, World!'

if __name__ == '__main__':
    app.debug = True
    app.run()