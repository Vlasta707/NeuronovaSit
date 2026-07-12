import torch
import torch.nn as nn
from PIL import Image

# --- Přesná architektura z train.py ---
class PrumyslovaSit(nn.Module):
    def __init__(self):
        super(PrumyslovaSit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)
        
        self.relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_resnet18_model(num_classes=2):
    """Vrací tvůj skutečný model (název zůstává kvůli kompatibilitě s inference.py)."""
    return PrumyslovaSit()

def classify_image(model, image_path, device, transform, class_names):
    """Načte obrázek v šedotónovém režimu a spustí inferenci."""
    try:
        # Tvá síť vyžaduje 1 kanál -> převod na 'L' (Grayscale)
        image = Image.open(image_path).convert('L')
        
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