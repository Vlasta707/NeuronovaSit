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
# Původní předdefinované `train_transform` a `test_transform` byly odstraněny,
# protože jsou dynamicky vytvořeny později na základě uživatelského vstupu.

# --- 3. Definice architektury CNN ---
class PrumyslovaSit(nn.Module):
    def __init__(self):
        super(PrumyslovaSit, self).__init__()
        # 1. konvoluční blok (vstup: 1 kanál -> výstup: 32 kanálů)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Společný pooling (zmenšuje rozměr feature mapy na polovinu)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2. konvoluční blok (vstup: 32 kanálů -> výstup: 64 kanálů)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # === NOVÁ VRSTVA: 3. konvoluční blok (vstup: 64 kanálů -> výstup: 64 kanálů) ===
        # padding=1 a kernel_size=3 zajišťují, že konvoluce zachová prostorový rozměr (změnu provede až pool)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Plně propojené (fully connected) vrstvy
        # Výsledný rozměr po 3x poolingu: 256 / 2 / 2 / 2 = 32 -> 64 * 32 * 32 zůstává ZACHOVÁN
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)
        
        self.relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 1. blok: Conv -> BN -> LeakyReLU -> Pool (z 256x256 na 128x128)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # 2. blok: Conv -> BN -> LeakyReLU -> Pool (z 128x128 na 64x64)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # === OPRAVA: 3. blok nyní řádně extrahuje rysy před poolingem (z 64x64 na 32x32) ===
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Narovnání (flatten) pro lineární vrstvu
        x = x.view(x.size(0), -1)
        
        # Klasifikační hlava
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# --- 4. Inicializace hardwaru, modelu a algoritmů ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PrumyslovaSit().to(device)
print(f"Model byl odeslán na zařízení: {device}")

criterion = nn.CrossEntropyLoss()
# Původní inicializace optimizeru zde byla odstraněna, protože je ihned přepsána
# po načtení konfigurace.

# --- Nastavení parametrů pro trénování
if not os.path.exists('train_config.json'):
    with open('train_config.json', 'w') as f:
        json.dump({'epochs': 20, 'lr': 0.001, 'batch_size': 16, 'use_aug': 'a', 'rot': 8, 'h_flip': 0.5, 'v_flip': 0.5, 'cj': 0.1}, f)
try:
    with open('train_config.json', 'r') as f:
        config = json.load(f)
except Exception:
    config = {'epochs': 20, 'lr': 0.001, 'batch_size': 16, 'use_aug': 'a', 'rot': 8, 'h_flip': 0.5, 'v_flip': 0.5, 'cj': 0.1}

# Pomocná funkce pro bezpečné načítání z konzole
def get_param(prompt, default, is_int=False, is_text=False):
    while True:
        user_input = input(prompt + f" ({default})? ").strip()
        if user_input == '':
            return default
        if is_text:
            return user_input.lower()
        try:
            return int(user_input) if is_int else float(user_input)
        except ValueError:
            print("Neplatná hodnota. Zkuste to znovu.")

# --- INTAREKTIVNÍ PROMPTY (NAČÍTÁNÍ) ---
print("\n--- Nastavení parametrů trénování ---")
epochs = int(get_param('Zadejte počet epoch', config.get('epochs', 20), is_int=True))
lr = get_param('Zadejte rychlost optimalizace (LR)', config.get('lr', 0.001))
batch_size = int(get_param('Zadejte velikost batchu', config.get('batch_size', 16), is_int=True))

print("\n--- Nastavení Data Augmentace ---")
use_aug = get_param('Chcete použít Data Augmentaci? (A/N)', config.get('use_aug', 'a'), is_text=True)
use_augmentation = use_aug != 'n'

# Inicializace výchozích hodnot pro případ, že je augmentace vypnutá
rot, h_flip, v_flip, cj_bright, cj_contrast = 0, 0.0, 0.0, 0.0, 0.0

if use_augmentation:
    rot = int(get_param('  Zadejte max. úhel rotace ve stupních', config.get('rot', 8), is_int=True))
    h_flip = get_param('  Pravděpodobnost horizontálního překlopení (0.0 - 1.0)', config.get('h_flip', 0.5))
    v_flip = get_param('  Pravděpodobnost vertikálního překlopení (0.0 - 1.0)', config.get('v_flip', 0.5))
    cj_val = get_param('  Intenzita ColorJitter jasu/kontrastu (0.0 - 1.0)', config.get('cj', 0.1))
    cj_bright = cj_contrast = cj_val

# --- ULOŽENÍ AKTUÁLNÍ KONFIGURACE PRO PŘÍŠTĚ ---
with open('train_config.json', 'w') as f:
    json.dump({
        'epochs': epochs, 'lr': lr, 'batch_size': batch_size, 
        'use_aug': 'a' if use_augmentation else 'n', 
        'rot': rot, 'h_flip': h_flip, 'v_flip': v_flip, 'cj': cj_bright
    }, f)

# --- DEFINICE TRANSFORMAČNÍCH PIPELINES ---
train_transform_list = []
if use_augmentation:
    if rot > 0: train_transform_list.append(transforms.RandomRotation(rot))
    # Opraveno: odstraněno zbytečné přiřazení k train_transform_list_and
    if h_flip > 0: train_transform_list.append(transforms.RandomHorizontalFlip(p=h_flip))
    if v_flip > 0: train_transform_list.append(transforms.RandomVerticalFlip(p=v_flip))
    if cj_bright > 0: train_transform_list.append(transforms.ColorJitter(brightness=cj_bright, contrast=cj_contrast))

train_transform_list.append(transforms.ToTensor())
train_transform_list.append(transforms.Normalize((0.5,), (0.5,)))

train_transform = transforms.Compose(train_transform_list)
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# --- REKAPITULACE V KONZOLI ---
print("\n" + "="*40)
print("Konfigurace spuštění:")
print(f"  Epochy: {epochs}")
print(f"  Learning Rate: {lr}")
print(f"  Velikost batchu: {batch_size}")
print(f"  Augmentace dat: {'ZAPNUTA' if use_augmentation else 'VYPNUTA'}")
if use_augmentation:
    print(f"    - Max rotace: {rot}°")
    print(f"    - Horiz. flip (p): {h_flip}")
    print(f"    - Vert. flip (p): {v_flip}")
    print(f"    - ColorJitter (b/c): {cj_bright}")
print("="*40 + "\n")

# --- KONEČNÁ ČÁST SKRIPTU ---
# Optimalizátor je inicializován zde s načteným learning rate
optimizer = optim.Adam(model.parameters(), lr=lr)

print("Načítám připravená data z .npy souborů...") # Přesunuto sem pro lepší logický tok

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