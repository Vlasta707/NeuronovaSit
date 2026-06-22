import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

# --- 1. Definice Datasetu pro načítání .npy souborů ---
class VLISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_data_path, label_data_path, transform=None):
        self.data = np.load(image_data_path) 
        self.targets = np.load(label_data_path) 
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        
        # Převede numpy array na PIL Image (L = šedotónový)
        image = Image.fromarray(image, mode='L') 

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        return image, label

# --- 2. Nastavení transformací a DataLoaderů ---
# POZOR: Musí odpovídat rozměru 256x256 z přípravného skriptu
transform = transforms.Compose([
    transforms.ToTensor(),                # Škáluje [0, 255] na [0.0, 1.0]
    transforms.Normalize((0.5,), (0.5,))  # Standardní normalizace pro šedotónové obrázky
])

print("Načítám připravená data z .npy souborů...")

train_dataset = VLISTDataset(
    image_data_path='./data/vlist_train_images.npy',
    label_data_path='./data/vlist_train_labels.npy',
    transform=transform
)

test_dataset = VLISTDataset(
    image_data_path='./data/vlist_test_images.npy',
    label_data_path='./data/vlist_test_labels.npy',
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# --- 3. Definice architektury CNN (pro rozměr 256x256) ---
class PrumyslovaSit(nn.Module):
    def __init__(self):
        super(PrumyslovaSit, self).__init__()
        # 1. konvoluční blok: vstup 1 kanál -> výstup 32 filtrů
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Zmenší rozměr na polovinu
        
        # 2. konvoluční blok: vstup 32 filtrů -> výstup 64 filtrů
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Výpočet rozměru po TŘECH MaxPool2d:
        # Vstup: 256x256
        # -> po 1. pool: 128x128
        # -> po 2. pool: 64x64
        # -> po 3. pool: 32x32
        # Výsledný rozměr ploché vrstvy: 64 kanálů * 32 * 32 lineárních vstupů
        self.fc1 = nn.Linear(64 * 32 * 32, 128) # Změněno z 64*64*64 na 64*32*32
        self.fc2 = nn.Linear(128, 2) # Výstup: 2 třídy (index 0 = BAD, index 1 = OK)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # Přidáváme Dropout vrstvu s pravděpodobností 0.5

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(x) # Nová pooling vrstva pro další snížení rozměru
        x = x.view(x.size(0), -1) # Zploštění (Flatten)
        x = self.relu(self.fc1(x))
        x = self.dropout(x) # Aplikujeme Dropout po ReLU v první plně propojené vrstvě
        x = self.fc2(x)
        return x


# --- 4. Inicializace hardwaru, modelu a algoritmů ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PrumyslovaSit().to(device)
print(f"Model byl odeslán na zařízení: {device}")

criterion = nn.CrossEntropyLoss()
# Používáme Adam optimalizátor – učí se stabilněji než základní SGD
optimizer = optim.Adam(model.parameters(), lr=0.001)


# --- 5. Trénovací cyklus (Training Loop) ---
def train(model, train_loader, optimizer, criterion, epochs=5):
    print(f"\nSpouštím trénink na {epochs} epoch...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()   # Vynulování přechozích gradientů
            output = model(data)    # Dopředný chod (predikce)
            loss = criterion(output, target) # Výpočet chyby
            loss.backward()         # Zpětný chod (výpočet parciálních derivací)
            optimizer.step()        # Aktualizace vah neuronů

            running_loss += loss.item()
            
            # Vypíše stav každých 5 dávek (vhodné pro menší datasety)
            if batch_idx % 5 == 4: 
                print(f"Epocha: {epoch+1}/{epochs} | Dávka: {batch_idx+1}/{len(train_loader)} | Ztráta (Loss): {running_loss / 5:.4f}")
                running_loss = 0.0


# --- 6. Vyhodnocení úspěšnosti (Testing) ---
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    print("\nVyhodnocuji model na testovací sadě...")
    
    with torch.no_grad(): # Vypne sledování gradientů (ušetří paměť při testu)
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # Vybere index s nejvyšší pravděpodobností (0 nebo 1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    print(f"Výsledná úspěšnost: {correct}/{total} ({100. * correct / total:.2f}%)")


if __name__ == "__main__":
    # Spustí trénink na 20 epoch a následně otestuje model
    train(model, train_loader, optimizer, criterion, epochs=20) # Změněno z 5 na 20
    test(model, test_loader)

    # --- 7. Uložení natrénovaného modelu s interaktivními dotazy ---
    save_decision = input("Uložit model? (Ano/Ne): ").lower()

    if save_decision == 'ano' or save_decision == 'y':
        model_name = input("Zadejte jméno souboru pro model (bez přípony, např. 'muj_prvni_model'): ").strip()

        if model_name:
            # Zajistíme, že soubor bude mít příponu .pth
            if not (model_name.endswith('.pth') or model_name.endswith('.pt')):
                model_name += '.pth'

            model_path = f'./{model_name}'
            torch.save(model.state_dict(), model_path)
            print(f"Model byl úspěšně uložen do: {model_path}")
        else:
            print("Jméno modelu nebylo zadáno, model nebude uložen.")
    else:
        print("Model nebyl uložen.")

