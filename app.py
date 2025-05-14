from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('runs/detect/train4/weights/best.pt')  # Path to your trained YOLOv8 model

@app.route('/detect', methods=['POST'])
def detect():
    # Expect a JPEG image in the request body
    nparr = np.frombuffer(request.data, np.uint8)  # Convert request data into numpy array
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode the image
    res = model(img, conf=0.3)[0]  # Run inference with confidence threshold of 0.3

    # Extract class names from the model output
    classes = [model.names[int(c)] for c in res.boxes.cls]
    
    # Example AI action: if 'Lamp' detected, suggest dimming lights
    actions = []
    if 'Lamp' in classes:
        actions.append('It seems you have a lamp; would you like to adjust brightness?')
    
    # Return the detected classes and suggested actions
    return jsonify({'classes': classes, 'actions': actions})

if __name__ == '__main__':
    app.run(port=5000)  # Run the Flask app on port 5000
