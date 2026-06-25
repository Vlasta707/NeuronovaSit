import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import json
import os

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

# Augmentace pro TRÉNOVÁNÍ (probíhá dynamicky v paměti)
train_transform = transforms.Compose([
    transforms.RandomRotation(10),               # Náhodná rotace o max trochu tam a zpět (-10° až +10°)
    transforms.RandomHorizontalFlip(p=0.5),      # Překlopení vodorovně s pravděpodobností 50 %
    transforms.RandomVerticalFlip(p=0.5),        # Překlopení svisle (ideální pro kruhová dna)
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Náhodná změna jasu a kontrastu
    transforms.ToTensor(),                       # Škáluje [0, 255] na [0.0, 1.0]
    transforms.Normalize((0.5,), (0.5,))         # Standardní normalizace
])

# ČISTÁ transformace pro TESTOVÁNÍ (žádná augmentace!)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Načítám připravená data z .npy souborů...")

# --- 3. Definice architektury CNN ---
class PrumyslovaSit(nn.Module):
    def __init__(self):
        super(PrumyslovaSit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# --- 4. Inicializace hardwaru, modelu a algoritmů ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PrumyslovaSit().to(device)
print(f"Model byl odeslán na zařízení: {device}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Nastavení parametrů pro trénování ---
if not os.path.exists('train_config.json'):
    with open('train_config.json', 'w') as f:
        json.dump({'epochs': 20, 'lr': 0.001, 'batch_size': 16}, f)
try:
    with open('train_config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    config = {'epochs': 20, 'lr': 0.001, 'batch_size': 16}

def get_param(prompt, default):
    while True:
        user_input = input(prompt + f" ({default})? ")
        if user_input == '':
            return default
        try:
            return float(user_input)
        except ValueError:
            print("Neplatná hodnota. Zkuste to znovu.")

epochs = int(get_param('Zadejte počet epoch: ', config['epochs']))
lr = get_param('Zadejte rychlost optimalizace (LR): ', config['lr'])
batch_size = int(get_param('Zadejte velikost batchu: ', config['batch_size']))

with open('train_config.json', 'w') as f:
    json.dump({'epochs': epochs, 'lr': lr, 'batch_size': batch_size}, f)

print(f"\nNastavení pro trénování:")
print(f"Epochs: {epochs}")
print(f"LR: {lr}")
print(f"Batch size: {batch_size}")

optimizer = optim.Adam(model.parameters(), lr=lr)

train_dataset = VLISTDataset(image_data_path='./data/vlist_train_images.npy', label_data_path='./data/vlist_train_labels.npy', transform=train_transform)
test_dataset = VLISTDataset(image_data_path='./data/vlist_test_images.npy', label_data_path='./data/vlist_test_labels.npy', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# --- 5. Trénovací cyklus (Training Loop) ---
def train(model, train_loader, optimizer, criterion, epochs):
    print(f"\nSpouštím trénink na {epochs} epoch...")
    loss_history = [] # Zde budeme ukládat záznamy o ztrátě
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if batch_idx % 5 == 4: 
                current_loss = running_loss / 5
                print(f"Epocha: {epoch+1}/{epochs} | Dávka: {batch_idx+1}/{len(train_loader)} | Ztráta (Loss): {current_loss:.4f}")
                
                # Uložíme si data pro pozdější export do Markdownu
                loss_history.append({
                    'epoch': epoch + 1,
                    'batch': batch_idx + 1,
                    'total_batches': len(train_loader),
                    'loss': current_loss
                })
                running_loss = 0.0
                
    return loss_history


# --- 6. Vyhodnocení úspěšnosti (Testing) ---
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    print("\nVyhodnocuji model na testovací sadě...")
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100. * correct / total
    print(f"Výsledná úspěšnost: {correct}/{total} ({accuracy:.2f}%)")
    
    return accuracy, correct, total


if __name__ == "__main__":
    # Spustíme trénink a zachytíme historii ztrát
    loss_history = train(model, train_loader, optimizer, criterion, epochs=epochs)
    accuracy, correct, total = test(model, test_loader)

    # --- 7. Uložení natrénovaného modelu a statistik do Markdownu ---
    save_decision = input("\nUložit model? (Ano/Ne): ").lower()

    if save_decision in ['ano', 'y']:
        model_name = input("Zadejte jméno souboru pro model (bez přípony, např. 'muj_prvni_model'): ").strip()

        if model_name:
            if model_name.endswith('.pth') or model_name.endswith('.pt'):
                base_name = os.path.splitext(model_name)[0]
            else:
                base_name = model_name

            model_filename = f"{base_name}.pth"
            md_filename = f"{base_name}.md"

            # Zajištění existence podadresáře "moje_modely" vzhledem k aktuálnímu adresáři
            output_dir = './moje_modely'
            os.makedirs(output_dir, exist_ok=True)

            # 1. Uložení .pth modelu do podadresáře
            model_path = os.path.join(output_dir, model_filename)
            torch.save(model.state_dict(), model_path)
            print(f"Model byl úspěšně uložen do: {model_path}")

            # 2. Příprava tabulky historie ztrát (Loss History) v Markdownu
            loss_table_rows = []
            for record in loss_history:
                loss_table_rows.append(
                    f"| {record['epoch']}/{epochs} | {record['batch']}/{record['total_batches']} | {record['loss']:.4f} |"
                )
            loss_table_content = "\n".join(loss_table_rows)

            # 3. Sestavení celkového Markdown dokumentu a uložení do podadresáře
            md_path = os.path.join(output_dir, md_filename)
            
            markdown_content = f"""# Vyhodnocení tréninku modelu: {base_name}

### Trénovací parametry
| Parametr | Hodnota |
| :--- | :--- |
| **Počet epoch** | {epochs} |
| **Learning Rate (LR)** | {lr} |
| **Batch Size** | {batch_size} |
| **Použité zařízení** | {device} |

### Výsledky testování
| Metrika | Hodnota |
| :--- | :--- |
| **Celková úspěšnost** | **{accuracy:.2f} %** |
| **Správně klasifikováno** | {correct} z {total} |

### Průběh trénování (Historie ztrát)
| Epocha | Dávka (Batch) | Ztráta (Loss) |
| :--- | :--- | :--- |
{loss_table_content}
"""
            
            with open(md_path, 'w', encoding='utf-8') as md_file:
                md_file.write(markdown_content)
            print(f"Statistiky a historie ztrát byly úspěšně uloženy do: {md_path}")

        else:
            print("Jméno modelu nebylo zadáno, model nebude uložen.")
    else:
        print("Model nebyl uložen.")