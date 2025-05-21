from flask import Flask, request, jsonify
import torch
import timm
import cv2
import numpy as np
from PIL import Image
import io
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pickle
import joblib
import os
import requests

app = Flask(__name__)

# ==============================
# Model Definition
# ==============================
class MultiTaskSwin(torch.nn.Module):
    def __init__(self, num_damage_classes, num_material_classes):
        super(MultiTaskSwin, self).__init__()
        self.backbone = timm.create_model("swin_small_patch4_window7_224", pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features

        self.damage_head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, num_damage_classes)
        )

        self.material_head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, num_material_classes)
        )

        self.size_head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 2)
        )

        self.time_head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 1)
        )

        self.quantity_head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 1),
            torch.nn.ReLU()
        )

    def forward(self, x):
        features = self.backbone(x)
        return {
            "damage": self.damage_head(features),
            "material": self.material_head(features),
            "size": self.size_head(features),
            "time": self.time_head(features),
            "quantity": self.quantity_head(features)
        }

# ==============================
# Load Label Encoders and Scalers
# ==============================
with open('./model/Si Pendekar/damage_le.pkl', 'rb') as f:
    damage_le = pickle.load(f)
with open('./model/Si Pendekar/material_le.pkl', 'rb') as f:
    material_le = pickle.load(f)

time_scaler = joblib.load('./model/Si Pendekar/time_scaler.pkl')
quantity_scaler = joblib.load('./model/Si Pendekar/quantity_scaler.pkl')

# Mapping label untuk damage (index ke string)
damage_mapping = {i: label for i, label in enumerate(damage_le.classes_)}

# ==============================
# Global Variables: Device & Model
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_damage_classes = len(damage_le.classes_)
num_material_classes = len(material_le.classes_)
model = MultiTaskSwin(num_damage_classes, num_material_classes).to(device)
model_path = "./model/Si Pendekar/model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found.")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ==============================
# TTA Predict Function
# ==============================
def tta_predict(image, tta=5):
    """
    Lakukan Test Time Augmentation (TTA) untuk mendapatkan prediksi yang stabil.
    """
    outputs_sum = None
    for _ in range(tta):
        # TTA transform: Resize dan random horizontal flip
        tta_transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        transformed = tta_transform(image=image)['image']
        input_tensor = transformed.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
        if outputs_sum is None:
            outputs_sum = {key: outputs[key] for key in outputs}
        else:
            for key in outputs:
                outputs_sum[key] += outputs[key]
    averaged_outputs = {key: outputs_sum[key] / tta for key in outputs_sum}
    return averaged_outputs

# ==============================
# Map Predictions Function
# ==============================
def map_predictions(damage_label, material_label, size_pred, time_pred_norm, quantity_pred_norm):
    """
    Map prediksi dari model ke format output yang diinginkan.
    Lakukan inverse transform untuk waktu perbaikan dan quantity.
    """
    repair_time = time_scaler.inverse_transform(np.array([[time_pred_norm]]))[0][0]
    quantity = quantity_scaler.inverse_transform(np.array([[quantity_pred_norm]]))[0][0]
    result = {
        "damage": damage_label,
        "material": material_label,
        "size": size_pred,  # list dengan 2 nilai
        "repair_time": repair_time,
        "quantity": quantity,
        "quantity_unit": "Kg"  # default quantity unit, sesuaikan bila perlu
    }
    return result

##########################################
# Flask Route: /predict
##########################################
@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    try:
        image = Image.open(image_file).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Invalid image format", "details": str(e)}), 400

    # Konversi PIL image ke numpy array
    image_np = np.array(image)

    # Gunakan TTA untuk memperoleh prediksi yang stabil
    outputs = tta_predict(image_np, tta=5)

    # Prediksi tiap head
    _, damage_pred = torch.max(outputs["damage"], dim=1)
    _, material_pred = torch.max(outputs["material"], dim=1)
    damage_label = damage_mapping.get(damage_pred.item(), "Unknown")
    material_label = material_le.inverse_transform(material_pred.cpu().numpy())[0]
    size_pred = outputs["size"].cpu().numpy()  # bentuk: [batch, 2]
    time_pred_norm = outputs["time"].squeeze().cpu().item()
    quantity_pred_norm = outputs["quantity"].squeeze().cpu().item()

    result = map_predictions(damage_label, material_label, size_pred[0].tolist(), time_pred_norm, quantity_pred_norm)

    # Opsional: kirim hasil ke service eksternal (misal, Laravel)
    try:
        laravel_response = requests.post("http://localhost:8000/api/save-prediction", json=result)
        if laravel_response.status_code == 200:
            result["message"] = "Prediction sent to Laravel"
        else:
            result["error"] = "Failed to send data to Laravel"
    except Exception as e:
        result["error"] = str(e)

    return jsonify(result)

# ==============================
# Jalankan Aplikasi Flask
# ==============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
