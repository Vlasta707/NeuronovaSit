# model_utils.py
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image

def get_resnet18_model(num_classes=2):
    """Inicializuje architekturu ResNet18 pro daný počet tříd."""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def classify_image(model, image_path, device, transform, class_names):
    """Načte obrázek, provede transformaci a vrátí predikovanou třídu a spolehlivost."""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        return class_names[predicted.item()], confidence.item() * 100
    except Exception as e:
        print(f"[CHYBA] Nepodařilo se klasifikovat obrázek {image_path}: {e}")
        return None