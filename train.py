import torch # Importuje hlavní knihovnu PyTorch, která poskytuje tenzory a funkce pro hluboké učení.
import torch.nn as nn # Importuje modul pro definování neuronových sítí a jejich vrstev.
import torch.optim as optim # Importuje modul s optimalizátory, které upravují váhy sítě během tréninku.
from torchvision import datasets, transforms # Importuje moduly z torchvision pro práci s obrazovými daty a transformacemi.
from torch.utils.data import DataLoader # Importuje DataLoader pro efektivní načítání a dávkování dat.

import numpy as np # Nový import pro VLISTDataset
from PIL import Image # Nový import pro VLISTDataset

# --- VLISTDataset pro načítání vlastních dat ---
# Tuto třídu můžete buď ponechat zde, nebo ji umístit do samostatného souboru
# a poté ji importovat (např. from my_datasets import VLISTDataset).
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
        image = Image.fromarray(image, mode='L') # 'L' pro grayscale

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        return image, label

# 1. Nastavení transformací a dat
# Obrázky převedeme na Tensory (matice čísel) a normalizujeme je.
# Transformace jsou operace, které se provedou s každým obrázkem před jeho vstupem do sítě.
transform = transforms.Compose([
    transforms.ToTensor(), # Převede obrázky z PIL Image nebo numpy.ndarray na PyTorch Tensor.
                           # Také škáluje hodnoty pixelů z [0, 255 na [0.0, 1.0].
    transforms.Normalize((0.1307,), (0.3081,)) # Normalizuje Tensor s daným průměrem a směrodatnou odchylkou.
                                            # Tyto hodnoty jsou specifické pro dataset MNIST a zlepšují trénink sítě.
                                            # Normalizace posouvá hodnoty pixelů, aby měly průměr 0 a směrodatnou odchylku 1.
])


print("Připravuji data (trénovací a testovací)...")












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


# Vytvoří DataLoader pro trénovací data. DataLoader usnadňuje iteraci přes dataset
# a automaticky rozděluje data do "dávek" (batchů).
train_loader = DataLoader(train_dataset,  # Dataset, ze kterého se budou načítat data.
                          batch_size=64,  # Počet vzorků v jedné dávce. Zde se bude trénovat na 64 obrázcích najednou.
                          shuffle=True)   # Zamíchá data v každé epoše, což pomáhá zabránit přeučení a zlepšuje zobecnění.

# Vytvoří DataLoader pro testovací data.
test_loader = DataLoader(test_dataset,


                         batch_size=200, # Pro 200 testovacích obrázků můžeme použít batch_size 200, aby se otestovaly najednou
                         shuffle=False)   # Testovací data obvykle nemícháme, aby byly výsledky konzistentní.

# 2. Definice architektury neuronové sítě
# Každá neuronová síť v PyTorch dědí z nn.Module.
class CislicovaSit(nn.Module):
    # Konstruktor třídy, zde definujeme vrstvy sítě.
    def __init__(self):
        super(CislicovaSit, self).__init__() # Volá konstruktor rodičovské třídy nn.Module.

        # fc1 je první plně propojená (fully connected) vrstva.
        # Vstup: 28x28 pixelů = 784 neuronů (každý pixel je jeden vstupní neuron).

        # Výstup: 128 neuronů ve skryté vrstvě.
        self.fc1 = nn.Linear(28 * 28, 128) # Původně chyběla, nebo byla omylem přesunuta


        # ReLU (Rectified Linear Unit) je aktivační funkce.
        # Přidává síti nelinearitu, což jí umožňuje učit se složitější vztahy.
        # Bez aktivačních funkcí by síť byla jen posloupností lineárních transformací.
        self.relu = nn.ReLU()

        # fc2 je druhá plně propojená (výstupní) vrstva.
        # Vstup: 128 neuronů ze skryté vrstvy.


        # Výstup: 2 neurony pro binární klasifikaci: 0 pro "BAD", 1 pro "OK".
        self.fc2 = nn.Linear(128, 2) # Dva výstupy pro "BAD" a "OK"



    # Metoda 'forward' definuje, jak data procházejí sítí.
    # 'x' je vstupní tensor (obrázek nebo dávka obrázků).
    def forward(self, x):
        # Zploštění obrázku z 2D (28x28 pixelů) do 1D řádku (784 pixelů).
        # '-1' říká PyTorchi, aby automaticky odvodil velikost dávky (batch size).
        x = x.view(-1, 28 * 28)
        # Data prochází první plně propojenou vrstvou (fc1) a poté aktivační funkcí ReLU.
        x = self.relu(self.fc1(x))
        # Data prochází druhou (výstupní) plně propojenou vrstvou (fc2).

        # Výstupem jsou "logity" – surová skóre pro každou ze 2 tříd.
        x = self.fc2(x)
        return x # Vrátí výstupy sítě.

# Inicializace modelu, ztrátové funkce a optimalizátoru.
model = CislicovaSit() # Vytvoří instanci naší neuronové sítě.

# Definice ztrátové funkce (Loss Function).


# CrossEntropyLoss je běžná ztrátová funkce pro klasifikační problémy s více třídami,
# včetně binární klasifikace (dvě třídy).
criterion = nn.CrossEntropyLoss()

# Definice optimalizátoru. Optimalizátor je algoritmus, který upravuje váhy sítě
# na základě vypočítaných gradientů (sklonů) ztrátové funkce.
# SGD (Stochastic Gradient Descent) je základní optimalizátor.
optimizer = optim.SGD(model.parameters(), # Řekne optimalizátoru, které parametry (váhy a bias) sítě má optimalizovat.
                      lr=0.01,           # 'lr' (learning rate) určuje, jak velkým krokem se budou váhy upravovat.
                                         # Menší lr = pomalejší, ale stabilnější učení; větší lr = rychlejší, ale nestabilní.
                      momentum=0.9)      # 'momentum' pomáhá optimalizátoru překonat lokální minima a zrychluje konvergenci.

# 3. Trénovací cyklus (Training Loop)
# Tato funkce trénuje model po zadaný počet epoch.
def train(model, train_loader, optimizer, criterion, epochs=3):
    model.train() # Nastaví model do trénovacího režimu. To aktivuje například dropout nebo batch normalization, pokud by byly použity.
    print(f"\nSpouštím trénink na {epochs} epoch...")
    for epoch in range(epochs): # Prochází se trénovací data tolikrát, kolik je zadáno 'epochs'.
        running_loss = 0.0 # Proměnná pro sledování kumulované ztráty v aktuální epoše.
        # Iteruje přes všechny dávky (batche) v trénovacím DataLoaderu.
        # 'batch_idx' je index dávky, 'data' jsou obrázky a 'target' jsou správné štítky.
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()   # Vynulování přechodů (gradientů) z minula.

            output = model(data)    # Dopředný chod (forward pass) - data se proženou sítí a získáme predikce (output).
            loss = criterion(output, target) # Výpočet chyby (ztráty) porovnáním predikcí sítě a skutečných štítků.
            loss.backward()         # Zpětný chod (backward pass) - vypočítá gradienty ztráty vzhledem k vahám sítě.

            optimizer.step()        # Úprava vah neuronů na základě vypočítaných gradientů.


            running_loss += loss.item() # Přidá hodnotu ztráty aktuální dávky k celkové ztrátě pro tisk. .item() získá skalární Python číslo z tensoru.


            if batch_idx % 20 == 19: # Tiskne průběžnou ztrátu každých 20 dávek (upraveno pro menší dataset)
                print(f"Epocha: {epoch+1} | Dávka: {batch_idx+1}/{len(train_loader)} | Ztráta (Loss): {running_loss / 20:.4f}")
                running_loss = 0.0 # Resetuje kumulovanou ztrátu pro další interval tisku.

# 4. Testování úspěšnosti sítě
# Tato funkce vyhodnocuje model na testovacím datasetu.
def test(model, test_loader):
    model.eval() # Nastaví model do evaluačního režimu. To deaktivuje chování specifické pro trénink (např. dropout).
    correct = 0 # Počítadlo pro správně klasifikované vzorky.
    total = 0 # Celkový počet testovaných vzorků
    # Kontext 'torch.no_grad()' říká PyTorchi, aby během tohoto bloku nepočítal ani neukládal gradienty.
    # To šetří paměť a zrychluje výpočet, protože gradienty nejsou při testování potřeba.
    with torch.no_grad():
        for data, target in test_loader: # Iteruje přes všechny dávky v testovacím DataLoaderu.
            output = model(data) # Dopředný chod - získá predikce sítě pro testovací data.







            pred = output.argmax(dim=1, keepdim=True) # Získá predikci: 0 pro "BAD", 1 pro "OK"
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0) # Přičtení počtu prvků v aktuální dávce



    # Zde se úspěšnost počítá správně, protože test_dataset nyní také obsahuje
    # binární popisky (0/1) odpovídající tréninku.

    print(f"\nVýsledná úspěšnost na testovacích datech (OK vs. BAD): {correct}/{total} ({100. * correct / total:.2f}%)")

# Spuštění celého procesu
# Toto je standardní Python konstrukce, která zajišťuje, že kód uvnitř bloku
# se spustí pouze tehdy, když je soubor spuštěn přímo (ne když je importován jako modul).
if __name__ == "__main__":
    train(model, train_loader, optimizer, criterion, epochs=3) # Zavolá trénovací funkci. Model se bude učit 3 epochy.
    test(model, test_loader) # Po tréninku zavolá testovací funkci, aby se vyhodnotila úspěšnost modelu.
