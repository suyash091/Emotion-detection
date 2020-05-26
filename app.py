from flask import Flask, render_template, session, request, jsonify
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout, BatchNormalization
import numpy as np
import keras
from keras.optimizers import SGD, Adam, Adadelta
import tensorflow as tf
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from PIL import Image
from keras.preprocessing import image as imag
import base64
import efficientnet.keras as efn
import tensorflow as tf
global model
import time
from tensorflow.python.keras.backend import set_session
sess = tf.Session()
set_session(sess)
model=keras.models.load_model('efficientnetb7valEmotion.h5')
global graph
graph = tf.get_default_graph()
import tensorflow as tf
emotions=['neutral',	'happiness',	'surprise',	'sadness',	'anger',	'disgust',	'fear', 'contempt',	'unknown',	'NF']

def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
   return img

def extract_face(img_data, required_size=(224, 224)):
    # load image from file
    pixels =readb64(img_data)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    im_pil = Image.fromarray(face_array)
    img = im_pil.convert('LA').convert('RGB')
    img = img.resize((48, 48))
    img = imag.img_to_array(img)
    img = img / 255.0
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        Ypred = model.predict(np.array([img]))
    return Ypred





app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def home():
    return render_template('home0.htm')

@app.route('/real-time', methods=['GET','POST'])
def realtime():
    return render_template('home.htm')

@app.route('/process', methods=['GET','POST'])
def process():
    start = time.time()
    try:
        value=emotions[np.argmax(extract_face(request.json['value']))]
    except:
        value="No Face, bad lighting or low Quality error"
    end = time.time()
    print("Model: seconds {}".format( end - start))
    return jsonify({'key':value})



if __name__=='__main__':
    app.run(host='0.0.0.0' , port=8000)