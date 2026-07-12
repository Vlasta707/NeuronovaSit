import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
# Import vah pro moderní API torchvision
from torchvision.models import ResNet18_Weights
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
        
        # VARIANTA A: Načteme šedotónový obrázek z .npy a převedeme ho na RGB.
        # Tím zkopírujeme 1 kanál do 3 identických vrstev, které vyžaduje ResNet.
        image = Image.fromarray(image, mode='L').convert('RGB') 

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        return image, label

# --- 2. Nastavení transformací a DataLoaderů ---
def calculate_dataset_stats(image_data_path, label_data_path):
    print("Vypočítávám průměr, směrodatnou odchylku a rozložení tříd z trénovacích dat...")
    
    try:
        labels = np.load(label_data_path)
        unique, counts = np.unique(labels, return_counts=True)
        class_distribution = dict(zip(unique.tolist(), counts.tolist()))
        print(f"Rozložení tříd v trénovacích datech: {class_distribution}")
    except Exception as e:
        print(f"Nepodařilo se spočítat rozložení tříd: {e}")
        class_distribution = None

    temp_transform = transforms.Compose([transforms.ToTensor()])
    temp_dataset = VLISTDataset(image_data_path=image_data_path, label_data_path=label_data_path, transform=temp_transform)
    temp_loader = DataLoader(temp_dataset, batch_size=128, shuffle=False, num_workers=0)

    # UPRAVENO: Statistiky nyní počítáme pro 3 kanály nezávisle
    sum_pixels = torch.zeros(3)
    sum_sq_pixels = torch.zeros(3)
    num_pixels = 0

    for images, _ in temp_loader:
        images = images.cpu() 
        # Celkový počet pixelů na jeden kanál v batchi
        num_pixels += images.size(0) * images.size(2) * images.size(3)
        for i in range(3):
            sum_pixels[i] += torch.sum(images[:, i, :, :])
            sum_sq_pixels[i] += torch.sum(images[:, i, :, :] ** 2)

    if num_pixels == 0:
        print("Upozornění: Dataset neobsahuje žádné pixely. Používám výchozí hodnoty (0.5).")
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], class_distribution

    calculated_mean = (sum_pixels / num_pixels).tolist()
    calculated_variance = (sum_sq_pixels / num_pixels) - (torch.tensor(calculated_mean) ** 2)
    calculated_std = torch.sqrt(torch.clamp(calculated_variance, min=1e-5)).tolist()

    print(f"Vypočítaný průměr datasetu (RGB): {calculated_mean}")
    print(f"Směrodatná odchylka datasetu (RGB): {calculated_std}")
    return calculated_mean, calculated_std, class_distribution


# --- 3. Inicializace hardwaru a předtrénovaného modelu ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nNačítám předtrénovaný model ResNet18 z torchvision...")
# Stažení nejlepších dostupných vah trénovaných na ImageNetu
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Úprava poslední (plně propojené) vrstvy pro naše 2 třídy (OK/BAD)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model = model.to(device)
print(f"Model ResNet18 byl úspěšně připraven na zařízení: {device}")

TRAIN_IMAGE_DATA_PATH = './data/vlist_train_images.npy'
TRAIN_LABEL_DATA_PATH = './data/vlist_train_labels.npy'
TEST_IMAGE_DATA_PATH = './data/vlist_test_images.npy'
TEST_LABEL_DATA_PATH = './data/vlist_test_labels.npy'

calculated_mean, calculated_std, class_dist = calculate_dataset_stats(TRAIN_IMAGE_DATA_PATH, TRAIN_LABEL_DATA_PATH)

# Načtení / Vytvoření konfigurace (Výchozí LR pro předtrénovaný model snížen na 0.0001)
if not os.path.exists('train_config.json'):
    with open('train_config.json', 'w') as f:
        json.dump({'epochs': 20, 'lr': 0.0001, 'batch_size': 16, 'use_aug': 'a', 'rot': 8, 'h_flip': 0.5, 'v_flip': 0.5, 'cj': 0.1, 'use_weights': 'n'}, f)
try:
    with open('train_config.json', 'r') as f:
        config = json.load(f)
except Exception:
    config = {'epochs': 20, 'lr': 0.0001, 'batch_size': 16, 'use_aug': 'a', 'rot': 8, 'h_flip': 0.5, 'v_flip': 0.5, 'cj': 0.1, 'use_weights': 'n'}

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

print("\n--- Nastavení parametrů trénování ---")
epochs = int(get_param('Zadejte počet epoch', config.get('epochs', 20), is_int=True))
lr = get_param('Zadejte rychlost optimalizace (LR)', config.get('lr', 0.0001))
batch_size = int(get_param('Zadejte velikost batchu', config.get('batch_size', 16), is_int=True))

use_weights_input = get_param('Chcete penalizovat nevyváženost tříd pomocí vah (Weighted Loss)? (A/N)', config.get('use_weights', 'n'), is_text=True)
use_weighted_loss = use_weights_input == 'a'

if use_weighted_loss and class_dist and len(class_dist) == 2:
    total_samples = sum(class_dist.values())
    w0 = total_samples / (2.0 * class_dist.get(0, 1))
    w1 = total_samples / (2.0 * class_dist.get(1, 1))
    loss_weights = torch.tensor([w0, w1], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    print(f"  -> Nastaveny váhy chybové funkce: Třída 0 (BAD) = {w0:.2f}, Třída 1 (OK) = {w1:.2f}")
else:
    criterion = nn.CrossEntropyLoss()
    print("  -> Použita standardní nevážená CrossEntropyLoss.")

print("\n--- Nastavení Data Augmentace ---")
use_aug = get_param('Chcete použít Data Augmentaci? (A/N)', config.get('use_aug', 'a'), is_text=True)
use_augmentation = use_aug != 'n'

rot, h_flip, v_flip, cj_bright, cj_contrast = 0, 0.0, 0.0, 0.0, 0.0

if use_augmentation:
    rot = int(get_param('  Zadejte max. úhel rotace ve stupních', config.get('rot', 8), is_int=True))
    h_flip = get_param('  Pravděpodobnost horizontálního překlopení (0.0 - 1.0)', config.get('h_flip', 0.5))
    v_flip = get_param('  Pravděpodobnost vertikálního překlopení (0.0 - 1.0)', config.get('v_flip', 0.5))
    cj_val = get_param('  Intenzita ColorJitter jasu/kontrastu (0.0 - 1.0)', config.get('cj', 0.1))
    cj_bright = cj_contrast = cj_val

with open('train_config.json', 'w') as f:
    json.dump({
        'epochs': epochs, 'lr': lr, 'batch_size': batch_size, 
        'use_aug': 'a' if use_augmentation else 'n', 
        'rot': rot, 'h_flip': h_flip, 'v_flip': v_flip, 'cj': cj_bright,
        'use_weights': 'a' if use_weighted_loss else 'n'
    }, f)

train_transform_list = []
if use_augmentation:
    if rot > 0: train_transform_list.append(transforms.RandomRotation(rot))
    if h_flip > 0: train_transform_list.append(transforms.RandomHorizontalFlip(p=h_flip))
    if v_flip > 0: train_transform_list.append(transforms.RandomVerticalFlip(p=v_flip))
    if cj_bright > 0: train_transform_list.append(transforms.ColorJitter(brightness=cj_bright, contrast=cj_contrast))

train_transform_list.append(transforms.ToTensor())
# UPRAVENO: Normalizace nyní bere nuly/seznamy jako n-tice (tuple) pro 3 kanály
train_transform_list.append(transforms.Normalize(tuple(calculated_mean), tuple(calculated_std)))

train_transform = transforms.Compose(train_transform_list)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(tuple(calculated_mean), tuple(calculated_std))
])

print("\n" + "="*40)
print("Konfigurace spuštění:")
print(f"  Model: ResNet18 (Předtrénovaný)")
print(f"  Epochy: {epochs}")
print(f"  Learning Rate: {lr}")
print(f"  Velikost batchu: {batch_size}")
print(f"  Vážená ztráta (Loss): {'ANO' if use_weighted_loss else 'NE'}")
print("="*40 + "\n")

optimizer = optim.Adam(model.parameters(), lr=lr)

print("Načítám připravená data z .npy souborů pro trénink a test...") 

train_dataset = VLISTDataset(image_data_path=TRAIN_IMAGE_DATA_PATH, label_data_path=TRAIN_LABEL_DATA_PATH, transform=train_transform)
test_dataset = VLISTDataset(image_data_path=TEST_IMAGE_DATA_PATH, label_data_path=TEST_LABEL_DATA_PATH, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# --- 5. Trénovací cyklus (Training Loop) ---
def train(model, train_loader, optimizer, criterion, epochs):
    print(f"\nSpouštím trénink na {epochs} epoch...")
    loss_history = []
    
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
            
            # Upraveno zobrazení frekvence dávky, protože při malém datasetu a batch_size=16 
            # proběhne málo dávek za epochu (cca 13). Zobrazujeme log každých 5 dávek.
            if batch_idx % 5 == 4: 
                current_loss = running_loss / 5
                print(f"Epocha: {epoch+1}/{epochs} | Dávka: {batch_idx+1}/{len(train_loader)} | Ztráta (Loss): {current_loss:.4f}")
                
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
    loss_history = train(model, train_loader, optimizer, criterion, epochs=epochs)
    accuracy, correct, total = test(model, test_loader)

    # --- 7. Uložení natrénovaného modelu a statistik do Markdownu ---
    save_decision = get_param('Uložit model? (A/N)', 'a', is_text=True)

    if save_decision == 'a':
        model_name = input("Zadejte jméno souboru pro model (bez přípony, např. 'resnet_v1'): ").strip()

        if model_name:
            base_name = os.path.splitext(model_name)[0]

            model_filename = f"{base_name}.pth"
            md_filename = f"{base_name}.md"

            output_dir = './moje_modely'
            os.makedirs(output_dir, exist_ok=True)

            model_path = os.path.join(output_dir, model_filename)
            checkpoint = {
                'state_dict': model.state_dict(),
                'mean': calculated_mean,
                'std': calculated_std
            }
            torch.save(checkpoint, model_path)
            print(f"Model a normalizační data byly úspěšně uloženy do: {model_path}")

            # Příprava tabulky historie ztrát (Loss History) v Markdownu
            loss_table_rows = []
            for record in loss_history:
                loss_table_rows.append(
                    f"| {record['epoch']}/{epochs} | {record['batch']}/{record['total_batches']} | {record['loss']:.4f} |"
                )
            loss_table_content = "\n".join(loss_table_rows)

            md_path = os.path.join(output_dir, md_filename)
            dist_str = f"Třída 0 (BAD): {class_dist.get(0, 0)}, Třída 1 (OK): {class_dist.get(1, 0)}" if class_dist else "Neuvedeno"

            markdown_content = f"""# Vyhodnocení tréninku modelu: {base_name}

### Trénovací parametry
| Parametr | Hodnota |
| :--- | :--- |
| **Modelová architektura** | ResNet18 (Předtrénovaný - ImageNet) |
| **Počet epoch** | {epochs} |
| **Learning Rate (LR)** | {lr} |
| **Batch Size** | {batch_size} |
| **Použité zařízení** | {device} |
| **Vážená Loss (Imbalance)** | {'ANO' if use_weighted_loss else 'NE'} |
| **Rozložení tříd v datech** | {dist_str} |

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