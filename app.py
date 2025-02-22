from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import requests  # Untuk mengirim data ke Laravel
import os

app = Flask(__name__)
CORS(app)

# Load model saat aplikasi dimulai
MODEL_PATH = "d:/Other/flask_api/pothole_swin.pth"
swin_model = torch.jit.load(MODEL_PATH)
swin_model.eval()

# Laravel API Endpoint untuk menerima hasil prediksi
LARAVEL_API_URL = "http://127.0.0.1:8000/api/save-prediction"  # Sesuaikan dengan Laravel

def transform_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    image = Image.open(image).convert("RGB")

    input_tensor = transform_image(image)

    with torch.no_grad():
        output = swin_model(input_tensor)

    predicted_class = torch.argmax(output, dim=1).item()

    label_dict = {0: "Pothole", 1: "Repair Area", 2: "Rut", 3: "Longitudinal Crack", 4: "Lateral Crack", 5: "Alligator Crack"}
    label = label_dict.get(predicted_class, "Unknown")

    # Kirim hasil prediksi ke Laravel
    data = {
        "label": label
    }
    
    try:
        response = requests.post(LARAVEL_API_URL, json=data)
        if response.status_code == 200:
            return jsonify({"label": label, "message": "Prediction sent to Laravel"})
        else:
            return jsonify({"label": label, "error": "Failed to send data to Laravel"}), 500
    except Exception as e:
        return jsonify({"label": label, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

