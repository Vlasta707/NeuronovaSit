import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# numpy a PIL už nepotřebujeme, protože nebudeme načítat VLISTDataset
# import numpy as np
# from PIL import Image

# --- Nastavení pro binární klasifikaci (MUSÍ ODPOVÍDAT VÝSTUPU MODELU) ---
# Zde definujeme, která číslice z MNISTu bude považována za "OK".
# Všechny ostatní číslice budou automaticky "BAD".
OK_DIGIT = 5 # Např. číslice 5 je "OK", ostatní "BAD"

# 1. Nastavení transformací a dat
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Standardní normalizace pro MNIST
])

print("Připravuji data (trénovací a testovací MNIST)...")

# --- Použití standardního MNIST datasetu ---
# Načtení trénovacího datasetu MNIST
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               download=True,
                               transform=transform)

# Načtení testovacího datasetu MNIST
test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              download=True,
                              transform=transform)

train_loader = DataLoader(train_dataset,
                          batch_size=64,
                          shuffle=True)

test_loader = DataLoader(test_dataset,
                         batch_size=1000, # Pro testování s MNIST je 1000 v pořádku
                         shuffle=False)

# 2. Definice architektury neuronové sítě - NYNÍ CNN!
class CislicovaSit(nn.Module):
    def __init__(self):
        super(CislicovaSit, self).__init__()
        # První konvoluční vrstva
        # Vstup: 1 kanál (stupně šedi), 32 výstupních kanálů (filtrů)
        # Kernel: 3x3, Padding: 1 (zachovává rozměry obrázku po konvoluci)
        # Výstup z Conv1: (Batch_size, 32, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # První Max Pooling vrstva
        # Zmenší rozměry na polovinu (28x28 -> 14x14)
        # Výstup z Pool1: (Batch_size, 32, 14, 14)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Druhá konvoluční vrstva
        # Vstup: 32 kanálů (z předchozí vrstvy), 64 výstupních kanálů
        # Kernel: 3x3, Padding: 1
        # Výstup z Conv2: (Batch_size, 64, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Druhá Max Pooling vrstva
        # Zmenší rozměry na polovinu (14x14 -> 7x7)
        # Výstup z Pool2: (Batch_size, 64, 7, 7)

        # Plně propojené (Fully Connected) vrstvy pro klasifikaci
        # Vstup: 64 kanálů * 7 * 7 (po zploštění) = 3136 neuronů
        # Výstup: 128 neuronů ve skryté vrstvě
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # Výstupní plně propojená vrstva
        # Vstup: 128 neuronů
        # Výstup: 2 neurony pro binární klasifikaci "BAD" (index 0) / "OK" (index 1)
        self.fc2 = nn.Linear(128, 2)

        # Aktivační funkce ReLU (pro konvoluční i plně propojené vrstvy)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Konvoluce -> ReLU -> Pooling
        x = self.pool(self.relu(self.conv1(x)))
        # Konvoluce -> ReLU -> Pooling
        x = self.pool(self.relu(self.conv2(x)))

        # Zploštění dat pro plně propojené vrstvy
        # x.size(0) je batch_size
        x = x.view(x.size(0), -1) # Tvar (batch_size, 64 * 7 * 7)

        # Plně propojené vrstvy
        x = self.relu(self.fc1(x))
        x = self.fc2(x) # Výstup pro 2 třídy
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
    print(f"\nSpouštím trénink na {epochs} epoch...")
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target_original) in enumerate(train_loader):
            # Převod původního MNIST targetu na binární (0 pro BAD, 1 pro OK)
            target = (target_original == OK_DIGIT).long()

            # Přesun dat na zařízení (CPU/GPU)
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 200 == 199: # Standardní tisk pro 60k MNIST dat
                print(f"Epocha: {epoch+1} | Dávka: {batch_idx+1}/{len(train_loader)} | Ztráta (Loss): {running_loss / 200:.4f}")
                running_loss = 0.0

# 4. Testování úspěšnosti sítě
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target_original in test_loader:
            # Převod původního MNIST targetu na binární (0 pro BAD, 1 pro OK)
            target = (target_original == OK_DIGIT).long()

            # Přesun dat na zařízení (CPU/GPU)
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # Získá predikci: 0 pro "BAD", 1 pro "OK"
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    print(f"\nVýsledná úspěšnost na testovacích datech (OK ({OK_DIGIT}) vs. BAD): {correct}/{total} ({100. * correct / total:.2f}%)")

if __name__ == "__main__":
    train(model, train_loader, optimizer, criterion, epochs=3)
    test(model, test_loader)
