import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np # Potřebné pro VLISTDataset
from PIL import Image # Potřebné pro VLISTDataset

# --- VLISTDataset pro načítání vlastních dat ---
# Tato třída načítá vaše obrázky a popisky z .npy souborů.
class VLISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_data_path, label_data_path, transform=None):
        self.data = np.load(image_data_path) # Načte obrázky
        self.targets = np.load(label_data_path) # Načte již binární popisky (0/1)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        image = Image.fromarray(image, mode='L') # Převede numpy array na PIL Image (grayscale)

        if self.transform:
            image = self.transform(image)

        # Popisky jsou již binární (0 nebo 1) a jsou typu int64 z numpy, převedeme je na torch.long
        label = torch.tensor(label, dtype=torch.long)
        return image, label

# 1. Nastavení transformací a dat
transform = transforms.Compose([
    transforms.ToTensor(), # Převede obrázky na PyTorch Tensor a škáluje [0, 255] na [0.0, 1.0].
    transforms.Normalize((0.1307,), (0.3081,)) # Normalizace s hodnotami pro MNIST (dobré pro šedotónové obrázky)
])

print("Připravuji VLASTNÍ data (trénovací a testovací VLIST)...")

# --- Použití VLISTDataset pro trénovací data ---
vlist_train_images_path = './data/vlist_train_images.npy'
vlist_train_labels_path = './data/vlist_train_labels.npy'

train_dataset = VLISTDataset(image_data_path=vlist_train_images_path,
                             label_data_path=vlist_train_labels_path,
                             transform=transform)

# --- Použití VLISTDataset pro testovací data ---
vlist_test_images_path = './data/vlist_test_images.npy'
vlist_test_labels_path = './data/vlist_test_labels.npy'

test_dataset = VLISTDataset(image_data_path=vlist_test_images_path,
                            label_data_path=vlist_test_labels_path,
                            transform=transform)


train_loader = DataLoader(train_dataset,
                          batch_size=64,
                          shuffle=True)

test_loader = DataLoader(test_dataset,
                         batch_size=200, # Pro 200 testovacích obrázků můžeme otestovat celou sadu najednou
                         shuffle=False)

# 2. Definice architektury neuronové sítě - CNN
class CislicovaSit(nn.Module):
    def __init__(self):
        super(CislicovaSit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Výpočet velikosti vstupu pro fc1 po konvolucích a pooling
        # Původní MNIST obrázek 28x28
        # Po conv1 (padding 1) -> 28x28
        # Po pool (kernel 2, stride 2) -> 14x14
        # Po conv2 (padding 1) -> 14x14
        # Po pool (kernel 2, stride 2) -> 7x7
        # 64 výstupních kanálů z conv2, proto 64 * 7 * 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # Výstupní vrstva pro binární klasifikaci "BAD" (index 0) / "OK" (index 1)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # Zploštění
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Inicializace modelu, ztrátové funkce a optimalizátoru.
model = CislicovaSit()

# Přesun modelu na GPU, pokud je k dispozici
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Používám zařízení: {device}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=0.01,
                      momentum=0.9)

# 3. Trénovací cyklus (Training Loop)
def train(model, train_loader, optimizer, criterion, epochs=3):
    model.train()
    print(f"\nSpouštím trénink na {epochs} epoch s vlastními VLIST daty...")
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Popisky 'target' jsou již binární (0 nebo 1) díky create_vlist_data.py
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 20 == 19: # Upravený tisk pro menší trénovací sadu (1000 obrázků)
                print(f"Epocha: {epoch+1} | Dávka: {batch_idx+1}/{len(train_loader)} | Ztráta (Loss): {running_loss / 20:.4f}")
                running_loss = 0.0

# 4. Testování úspěšnosti sítě
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    print("\nVyhodnocuji model na vlastní testovací VLIST sadě (OK vs. BAD)...")
    with torch.no_grad():
        for data, target in test_loader:
            # Popisky 'target' jsou již binární (0 nebo 1) díky create_vlist_test_data.py
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # Získá predikci: 0 pro "BAD", 1 pro "OK"
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    print(f"\nVýsledná úspěšnost na testovacích datech (OK vs. BAD): {correct}/{total} ({100. * correct / total:.2f}%)")

if __name__ == "__main__":
    train(model, train_loader, optimizer, criterion, epochs=3)
    test(model, test_loader)