import pickle
import os
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from PIL import Image
import requests
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Tentukan device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################
# Definisikan Arsitektur Model MultiTaskSwin
##########################################
class MultiTaskSwin(nn.Module):
    def __init__(self, num_damage_classes, num_material_classes):
        super(MultiTaskSwin, self).__init__()
        self.backbone = timm.create_model("swin_small_patch4_window7_224", pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features

        self.damage_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_damage_classes)
        )
        self.material_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_material_classes)
        )
        self.size_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2)
        )
        self.time_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        self.quantity_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.ReLU()
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

##########################################
# Load scaler dari file (hasil training)
##########################################
SCALER_PATH = "/mnt/d/Projek/Road Damage Classification (SiPendekar)/model/"
with open(os.path.join(SCALER_PATH, "time_scaler.pkl"), "rb") as f:
    time_scaler = pickle.load(f)
with open(os.path.join(SCALER_PATH, "quantity_scaler.pkl"), "rb") as f:
    quantity_scaler = pickle.load(f)

##########################################
# Load Model
##########################################
MODEL_PATH = os.path.join(SCALER_PATH, "road_classification_model.pth")
num_damage_classes = 6
num_material_classes = 3
model = MultiTaskSwin(num_damage_classes, num_material_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

##########################################
# Mapping Label
##########################################
damage_mapping = {
    0: "lubang jalan", 
    1: "area perbaikan", 
    2: "alur jalan",
    3: "retakan memanjang", 
    4: "retakan lateral", 
    5: "retakan buaya"
}

##########################################
# Transformasi Gambar untuk Inferensi
##########################################
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
def transform_image(image):
    return val_transform(image).unsqueeze(0)

##########################################
# Fungsi Mapping Output Prediksi
##########################################
def map_predictions(damage_label, size, time_norm, quantity_norm):
    # Inverse transform repair time dan quantity
    repair_time = time_scaler.inverse_transform(np.array([[time_norm]]))[0][0]
    quantity_value = quantity_scaler.inverse_transform(np.array([[quantity_norm]]))[0][0]
    
    damage_label_lower = damage_label.lower()
    if damage_label_lower in ["lubang jalan", "area perbaikan"]:
        material_final = "Aspal Panas Campuran"
        quantity_unit = "Kg"
        final_size = float(size[0])  # Diameter
    elif damage_label_lower == "alur jalan":
        material_final = "Penggilingan dan Pelapisan Aspal"
        quantity_unit = "Kg"
        final_size = size.tolist()  # [panjang, lebar]
    else:
        material_final = "Penambal Retakan"
        quantity_unit = "L"
        final_size = size.tolist()  # [panjang, lebar]
        
    return {
        "damage": damage_label,
        "size": final_size,
        "repair_time": repair_time,
        "material": material_final,
        "quantity": quantity_value,
        "quantity_unit": quantity_unit
    }

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

    input_tensor = transform_image(image).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)

    # Prediksi tiap head
    _, damage_pred = torch.max(outputs["damage"], dim=1)
    size_pred = outputs["size"].cpu().numpy()  # [batch, 2]
    time_pred_norm = outputs["time"].squeeze().cpu().item()
    quantity_pred_norm = outputs["quantity"].squeeze().cpu().item()

    damage_label = damage_mapping.get(damage_pred.item(), "Unknown")
    result = map_predictions(damage_label, size_pred[0], time_pred_norm, quantity_pred_norm)

    # Kirim hasil ke Laravel
    try:
        laravel_response = requests.post("http://localhost:8000/api/save-prediction", json=result)
        if laravel_response.status_code == 200:
            result["message"] = "Prediction sent to Laravel"
        else:
            result["error"] = "Failed to send data to Laravel"
    except Exception as e:
        result["error"] = str(e)

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
