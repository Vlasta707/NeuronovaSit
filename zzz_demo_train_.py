import torch # Importuje hlavní knihovnu PyTorch, která poskytuje tenzory a funkce pro hluboké učení.
import torch.nn as nn # Importuje modul pro definování neuronových sítí a jejich vrstev.
import torch.optim as optim # Importuje modul s optimalizátory, které upravují váhy sítě během tréninku.
from torchvision import datasets, transforms # Importuje moduly z torchvision pro práci s obrazovými daty a transformacemi.
from torch.utils.data import DataLoader # Importuje DataLoader pro efektivní načítání a dávkování dat.
# 1. Nastavení transformací a stažení dat
# Obrázky převedeme na Tensory (matice čísel) a normalizujeme je.
# Transformace jsou operace, které se provedou s každým obrázkem před jeho vstupem do sítě.
transform = transforms.Compose([
    transforms.ToTensor(), # Převede obrázky z PIL Image nebo numpy.ndarray na PyTorch Tensor.
                           # Také škáluje hodnoty pixelů z [0, 255 na [0.0, 1.0].
    transforms.Normalize((0.1307,), (0.3081,)) # Normalizuje Tensor s daným průměrem a směrodatnou odchylkou.
                                            # Tyto hodnoty jsou specifické pro dataset MNIST a zlepšují trénink sítě.
                                            # Normalizace posouvá hodnoty pixelů, aby měly průměr 0 a směrodatnou odchylku 1.
])

print("Stahuji a připravuji data...") # Informační zpráva pro uživatele.

# Stáhne trénovací dataset MNIST (nebo ho načte, pokud už je stažený).
# MNIST je dataset ručně psaných číslic (0-9). Každý obrázek má velikost 28x28 pixelů.
train_dataset = datasets.MNIST(root='./data', # 'root' určuje cestu, kam se dataset stáhne/uloží.
                               train=True,    # Určuje, že chceme trénovací část datasetu.
                               download=True, # Pokud dataset není k dispozici, stáhne ho.
                               transform=transform) # Aplikuje definované transformace na každý obrázek.

# Stáhne testovací dataset MNIST. Testovací dataset se používá k ověření výkonu sítě na datech, která nikdy neviděla.
test_dataset = datasets.MNIST(root='./data',
                              train=False,   # Určuje, že chceme testovací část datasetu.
                              download=True,
                              transform=transform)

# Vytvoří DataLoader pro trénovací data. DataLoader usnadňuje iteraci přes dataset
# a automaticky rozděluje data do "dávek" (batchů).
train_loader = DataLoader(train_dataset,  # Dataset, ze kterého se budou načítat data.
                          batch_size=64,  # Počet vzorků v jedné dávce. Zde se bude trénovat na 64 obrázcích najednou.
                          shuffle=True)   # Zamíchá data v každé epoše, což pomáhá zabránit přeučení a zlepšuje zobecnění.

# Vytvoří DataLoader pro testovací data.
test_loader = DataLoader(test_dataset,
                         batch_size=1000, # Větší dávka pro testování je běžná, protože nepotřebujeme zpětnou propagaci.
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
        self.fc1 = nn.Linear(28 * 28, 128) # nn.Linear je lineární transformace: y = xA^T + b.

        # ReLU (Rectified Linear Unit) je aktivační funkce.
        # Přidává síti nelinearitu, což jí umožňuje učit se složitější vztahy.
        # Bez aktivačních funkcí by síť byla jen posloupností lineárních transformací.
        self.relu = nn.ReLU()

        # fc2 je druhá plně propojená (výstupní) vrstva.
        # Vstup: 128 neuronů ze skryté vrstvy.
        # Výstup: 10 neuronů (pro 10 tříd: číslice 0-9).
        # Každý z těchto 10 výstupů představuje "skóre" pro danou číslici.
        self.fc2 = nn.Linear(128, 10)

    # Metoda 'forward' definuje, jak data procházejí sítí.
    # 'x' je vstupní tensor (obrázek nebo dávka obrázků).
    def forward(self, x):
        # Zploštění obrázku z 2D (28x28 pixelů) do 1D řádku (784 pixelů).
        # '-1' říká PyTorchi, aby automaticky odvodil velikost dávky (batch size).
        x = x.view(-1, 28 * 28)
        # Data prochází první plně propojenou vrstvou (fc1) a poté aktivační funkcí ReLU.
        x = self.relu(self.fc1(x))
        # Data prochází druhou (výstupní) plně propojenou vrstvou (fc2).
        # Výstupem jsou "logity" – surová skóre pro každou z 10 tříd.
        x = self.fc2(x)
        return x # Vrátí výstupy sítě.
# Inicializace modelu, ztrátové funkce a optimalizátoru.
model = CislicovaSit() # Vytvoří instanci naší neuronové sítě.

# Definice ztrátové funkce (Loss Function).
# CrossEntropyLoss je běžná ztrátová funkce pro klasifikační problémy s více třídami.
# Měří, jak moc se předpovědi sítě liší od skutečných štítků.
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
    for epoch in range(epochs): # Prochází se trénovací data tolikrát, kolik je zadáno 'epochs'.
        running_loss = 0.0 # Proměnná pro sledování kumulované ztráty v aktuální epoše.
        # Iteruje přes všechny dávky (batche) v trénovacím DataLoaderu.
        # 'batch_idx' je index dávky, 'data' jsou obrázky a 'target' jsou správné štítky.
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()   # Vynulování přechodů (gradientů) z minula.
                                    # Gradienty se kumulují, takže je nutné je vynulovat pro každou novou dávku.
            output = model(data)    # Dopředný chod (forward pass) - data se proženou sítí a získáme predikce (output).
            loss = criterion(output, target) # Výpočet chyby (ztráty) porovnáním predikcí sítě a skutečných štítků.
            loss.backward()         # Zpětný chod (backward pass) - vypočítá gradienty ztráty vzhledem k vahám sítě.
                                    # Tyto gradienty ukazují směr a velikost, jak se mají váhy změnit.
            optimizer.step()        # Úprava vah neuronů na základě vypočítaných gradientů.
                                    # Optimalizátor aplikuje pravidla učení (např. SGD) a aktualizuje parametry modelu.

            running_loss += loss.item() # Přidá hodnotu ztráty aktuální dávky k celkové ztrátě pro tisk. .item() získá skalární Python číslo z tensoru.
            if batch_idx % 200 == 199: # Tiskne průběžnou ztrátu každých 200 dávek.
                print(f"Epocha: {epoch+1} | Dávka: {batch_idx+1}/{len(train_loader)} | Ztráta (Loss): {running_loss / 200:.4f}")
                running_loss = 0.0 # Resetuje kumulovanou ztrátu pro další interval tisku.
# 4. Testování úspěšnosti sítě
# Tato funkce vyhodnocuje model na testovacím datasetu.
def test(model, test_loader):
    model.eval() # Nastaví model do evaluačního režimu. To deaktivuje chování specifické pro trénink (např. dropout).
    correct = 0 # Počítadlo pro správně klasifikované vzorky.
    # Kontext 'torch.no_grad()' říká PyTorchi, aby během tohoto bloku nepočítal ani neukládal gradienty.
    # To šetří paměť a zrychluje výpočet, protože gradienty nejsou při testování potřeba.
    with torch.no_grad():
        for data, target in test_loader: # Iteruje přes všechny dávky v testovacím DataLoaderu.
            output = model(data) # Dopředný chod - získá predikce sítě pro testovací data.
            # pred = output.argmax(dim=1, keepdim=True) # Vybere index nejvyšší hodnoty z výstupů sítě.
                                                    # Tento index odpovídá predikované číslici (0-9).
                                                    # 'dim=1' znamená, že se argmax aplikuje podél dimenze pro třídy.
                                                    # 'keepdim=True' zachová dimenzi výstupu, což je užitečné pro porovnání.
            pred = output.argmax(dim=1, keepdim=True) # Najde index třídy s nejvyšším skóre pro každou predikci v dávce.
                                                    # Např. [0.1, 0.8, 0.05, ...] -> 1 (index 1 má nejvyšší skóre).
            # Porovná predikované číslice s reálnými štítky.
            # 'target.view_as(pred)' zajistí, že mají stejný tvar pro porovnání.
            # '.eq()' provede element-wise porovnání (True, kde se rovnají).
            # '.sum().item()' sečte všechny True hodnoty (převedené na 1) a převede výsledek na Python číslo.
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Vypíše celkovou úspěšnost na testovacích datech.
    print(f"\nVýsledná úspěšnost na testovacích datech: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)")

# Spuštění celého procesu
# Toto je standardní Python konstrukce, která zajišťuje, že kód uvnitř bloku
# se spustí pouze tehdy, když je soubor spuštěn přímo (ne když je importován jako modul).
if __name__ == "__main__":
    train(model, train_loader, optimizer, criterion, epochs=3) # Zavolá trénovací funkci. Model se bude učit 3 epochy.
    test(model, test_loader) # Po tréninku zavolá testovací funkci, aby se vyhodnotila úspěšnost modelu.
