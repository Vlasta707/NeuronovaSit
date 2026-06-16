import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Nastavení transformací a stažení dat
# Obrázky převedeme na Tensory (matice čísel) a normalizujeme je
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("Stahuji a připravuji data...")
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. Definice architektury neuronové sítě
class CislicovaSit(nn.Module):
    def __init__(self):
        super(CislicovaSit, self).__init__()
        # Vstup: 28x28 pixelů = 784 neuronů
        # Skrytá vrstva: 128 neuronů
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        # Výstupní vrstva: 10 neuronů (pro čísla 0-9)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Zploštění obrázku z 2D (28x28) do 1D řádku (784)
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Inicializace modelu, ztrátové funkce a optimalizátoru
model = CislicovaSit()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 3. Trénovací cyklus (Training Loop)
def train(model, train_loader, optimizer, criterion, epochs=3):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()   # Vynulování přechodů (gradientů) z minula
            output = model(data)    # Dopředný chod - předpověď sítě
            loss = criterion(output, target) # Výpočet chyby
            loss.backward()         # Zpětný chod - šíření chyby sítí zpět
            optimizer.step()        # Úprava vah neuronů
            
            running_loss += loss.item()
            if batch_idx % 200 == 199:
                print(f"Epocha: {epoch+1} | Dávka: {batch_idx+1}/{len(train_loader)} | Ztráta (Loss): {running_loss / 200:.4f}")
                running_loss = 0.0

# 4. Testování úspěšnosti sítě
def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad(): # Při testování nepotřebujeme počítat gradienty
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # Vybereme nejpravděpodobnější číslo
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f"\nVýsledná úspěšnost na testovacích datech: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)")

# Spuštění celého procesu
if __name__ == "__main__":
    train(model, train_loader, optimizer, criterion, epochs=3)
    test(model, test_loader)