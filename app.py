from flask import Flask, render_template, request, redirect, url_for,Response,jsonify
from werkzeug.utils import secure_filename
# import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions 
import os
from PIL import Image
import numpy as np
import torch
import cv2
# print("hello")
# model = load_model('resnet.h5')
model1=load_model('my_model_incep.h5')
from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__ ,static_url_path='/static')
modela = YOLO('best1.pt')


@app.route('/webcamv', methods=['POST'])
def detectw():
    result = modela.predict(source=0, imgsz=640, conf=0.6, show=True)
    return result.imgs[0]
app.config['UPLOAD_FOLDER'] = 'static/uploads'
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/classification.html')
def classification():
    return render_template('classification.html')

@app.route('/classify', methods=['POST'])
def classify():
    
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Call function to classify the image and get predicted weed name
        predicted_weed = classify_image(file_path)
        return redirect(url_for('result', filename=filename, weed=predicted_weed))

@app.route('/result')
def result():
    filename = request.args.get('filename')
    predicted_weed = request.args.get('weed')
    return render_template('result.html', filename=filename, predicted_weed=predicted_weed)

def classify_image(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    print(model1.predict(img_array))
    prediction = np.argmax(model1.predict(img_array))
    # Load the classification model (classification.h5)
    # Process the uploaded image using the classification model
    # Get the predicted weed name
    # predicted_weed = "carpetweed"  # Replace with actual prediction
    res = {0:"CarpetWeeds",1:"Crabgrass",2:"Eclipta",3:"Goosegrass",4:"Morningglory",5:"Nutsedge",6:"PalmerAmaranth",7:"Prickly Sida",8:"Purslane",9:"Ragweed",10:"Sicklepod",11:"SpottedSpurge",12:"SpurredAnoda",13:"Swinecress",14:"Waterhemp"}
    print(prediction)
    return res[prediction]

@app.route('/webcam.html')
def detection():
    return render_template('webcam.html')
classNames = ['CarpetWeeds', 'Crabgrass', 'Eclipta', 'Goosegrass', 'Morningglory', 'Nutsedge', 'PalmerAmaranth', 'Prickly Sida', 'Purslane', 'Ragweed', 'Sicklepod', 'SpottedSpurge']
@app.route('/index1.html')
def index1():
    return render_template('index1.html')

def predict_image1(image):
    try:
        # Resize the image to fit the YOLO model input size (optional based on model requirements)
        input_size = (299, 299)
        img = image.resize(input_size)
        img_array = np.array(image)

        # Perform YOLO prediction
        results = modela(img_array)

        # Process results and extract class name and confidence score
        predictions = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf)
                cls = int(box.cls)
                class_name = classNames[cls]
                predictions.append({'class': class_name, 'confidence': conf})

        return predictions
    except Exception as e:
        print("Error during prediction:", str(e))  # Print the exception message for debugging
        return []

@app.route('/detect1', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        image_file = request.files['file']
        image = Image.open(image_file.stream)

        # Perform prediction
        predictions = predict_image1(image)

        # Return predictions as JSON
        return jsonify(predictions)
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': 'An error occurred during prediction.'})

@app.route('/analytics.html')
def analytics():
    return render_template('analytics.html')
def detect_objects(image_path):
    # Your code for object detection with YOLO model
    # Replace this with actual detection code
    # Example: read image and return the same image
    image = cv2.imread(image_path)
    return image
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        result_image_path = detect_objects(file_path)
        return redirect(url_for('detection_result', filename=filename))
@app.route('/detectionresult')
def detection_result():
    filename = request.args.get('filename')
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return render_template('detectionresult.html', result_image=result_image_path)

if __name__ == '__main__':
    app.run(debug=True)
