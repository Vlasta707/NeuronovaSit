import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os # Pro kontrolu existence souborů

# --- 1. Definice architektury CNN (musí být shodná s tréninkovým modelem) ---
# Tuto definici jsem zkopíroval z tvého train.py
class PrumyslovaSit(nn.Module):
    def __init__(self):
        super(PrumyslovaSit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128) # Po třech poolech
        self.fc2 = nn.Linear(128, 2) # Výstup: 2 třídy (0 = BAD, 1 = OK)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # Důležité: musí odpovídat tréninkovému modelu

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- 2. Nastavení transformací pro vstupní obrázek ---
# Tyto transformace musí být PŘESNĚ STEJNÉ jako ty použité při tréninku
transform = transforms.Compose([
    transforms.Resize((256, 256)),        # Změní velikost obrázku na 256x256
    transforms.ToTensor(),                # Škáluje [0, 255] na [0.0, 1.0]
    transforms.Normalize((0.5,), (0.5,))  # Standardní normalizace pro šedotónové obrázky
])

# --- 3. Funkce pro klasifikaci jednoho obrázku ---
def classify_image(model, image_path, device, transform, class_names):
    if not os.path.exists(image_path):
        print(f"Chyba: Soubor obrázku '{image_path}' nebyl nalezen.")
        return None

    try:
        image = Image.open(image_path).convert('L') # Načte obrázek a převede na šedotónový
    except Exception as e:
        print(f"Chyba při načítání nebo zpracování obrázku '{image_path}': {e}")
        return None

    image_tensor = transform(image).unsqueeze(0) # Přidá dimenzi pro batch (1 obrázek)
    image_tensor = image_tensor.to(device)

    model.eval() # Přepne model do evaluačního módu (vypne dropout)
    with torch.no_grad(): # Vypne výpočet gradientů pro úsporu paměti a rychlost
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1) # Převede logits na pravděpodobnosti
        predicted_prob, predicted_idx = torch.max(probabilities, 1) # Získá nejpravděpodobnější třídu

    prediction = class_names[predicted_idx.item()]
    confidence = predicted_prob.item() * 100

    return prediction, confidence

# --- 4. Hlavní část skriptu ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Používám zařízení: {device}")

    # Definice jmen tříd (musí odpovídat indexům ve výstupu modelu)
    class_names = {0: "BAD", 1: "OK"}

    # --- 4.1. Načtení modelu ---
    model_name_input = input("Zadejte jméno souboru modelu k načtení (např. 'První pokus.pth'): ").strip()
    
    # Přidáme .pth, pokud není přítomno
    if not (model_name_input.endswith('.pth') or model_name_input.endswith('.pt')):
        model_name_input += '.pth'
    
    model_path = f'./{model_name_input}' # Předpokládáme, že je model ve stejném adresáři

    if not os.path.exists(model_path):
        print(f"Chyba: Soubor modelu '{model_path}' nebyl nalezen.")
    else:
        model = PrumyslovaSit().to(device)
        try:
            # Načte state_dict na příslušné zařízení (CPU nebo GPU)
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model '{model_name_input}' byl úspěšně načten.")

            # --- 4.2. Klasifikace obrázku ---
            image_file_path = input("Zadejte plnou cestu k obrázku .png, který chcete klasifikovat: ").strip()

            result = classify_image(model, image_file_path, device, transform, class_names)

            if result:
                prediction, confidence = result
                print(f"\nPredikce pro obrázek '{image_file_path}':")
                print(f"Třída: {prediction}")
                print(f"Spolehlivost (Confidence): {confidence:.2f}%")
            else:
                print("Klasifikace obrázku selhala.")

        except Exception as e:
            print(f"Chyba při načítání nebo použití modelu: {e}")

