from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("best1.pt")
classNames = ['CarpetWeeds', 'Crabgrass', 'Eclipta', 'Goosegrass', 'Morningglory', 'Nutsedge', 'PalmerAmaranth', 'Prickly Sida', 'Purslane', 'Ragweed', 'Sicklepod', 'SpottedSpurge']


def predict_image1(image):
    try:
        # Resize the image to fit the YOLO model input size (optional based on model requirements)
        input_size = (299, 299)
        img = image.resize(input_size)
        img_array = np.array(image)

        # Perform YOLO prediction
        results = model(img_array)

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

@app.route('/')
def index():
    return render_template('index1.html')

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

if __name__ == "__main__":
    app.run(debug=True)
