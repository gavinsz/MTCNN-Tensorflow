from flask import Flask
from flask import request
from flask import send_file
import json
import detect

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save('tmp/tmp.jpg')
        print('recv upload req')
        bb, _=detect.face_detector.detect('tmp/tmp.jpg')
        print('boxes=', bb)
        return json.dumps(bb.tolist())
        return send_file('data/tmp_cropped.png')
    return 'Hello, World!'

if __name__ == '__main__':
    app.debug = True
    app.run()