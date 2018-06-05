# coding: utf-8
import detect
import json
import logging
from flask import Flask
from flask import request
from flask import send_file
import mxnet as mx
from mtcnn_detector import MtcnnDetector


# 通过下面的方式进行简单配置输出方式与日志级别
#logging.basicConfig(filename='logger.log', level=logging.INFO)
logging.basicConfig(filename='logger.log', 
                    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s', 
                    level = logging.INFO,
                    filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save('tmp/tmp.jpg')
        print('recv upload req')
        bb=detect.face.detect('tmp/tmp.jpg')
        #return json.dumps(bb.tolist())
        return send_file('data/tmp_cropped.png')
    return 'Hello, World!'

@app.route('/uploadStream', methods=['GET', 'POST'])
def upload_stream_file():
    if request.method == 'POST':
        f = request.files['file']
        bb=detect.face.detectStream(f.stream)
        return json.dumps(bb.tolist())
    return 'Hello, World!'

if __name__ == '__main__':
    app.debug = True
    app.run()
