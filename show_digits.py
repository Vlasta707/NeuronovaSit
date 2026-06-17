import matplotlib.pyplot as plt # Import knihovny Matplotlib pro vytváření grafů a zobrazování obrázků
import numpy as np # Import knihovny NumPy pro práci s numerickými daty, zejména poli (arrays)
from tensorflow.keras.datasets import mnist # Import datové sady MNIST z TensorFlow Keras, která obsahuje obrázky ručně psaných číslic
def show_random_digits():
    """
    Načte datovou sadu MNIST, vybere náhodné obrázky číslic a zobrazí je v mřížce
    s příslušnými popisky. Funkce je navržena tak, aby pomohla začátečníkům vizualizovat
    data, se kterými se běžně pracuje v oblasti strojového učení.
    """
    # Načtení dat MNIST
    # Data jsou rozdělena na trénovací a testovací sady.
    # Pro účely zobrazení potřebujeme pouze trénovací data (x_train, y_train).
    # Testovací data (x_test, y_test) ignorujeme pomocí podtržítka (_).
    (x_train, y_train), (_, _) = mnist.load_data()

    # Normalizace pixelových hodnot na rozsah 0-1.
    # Původní hodnoty pixelů jsou v rozsahu 0-255. Dělením 255.0 je převedeme na float
    # v rozsahu 0 až 1. Ačkoliv imshow zvládne i 0-255, normalizace je dobrá praxe
    # pro mnoho modelů strojového učení a někdy může zlepšit vizualizaci.
    x_train = x_train / 255.0

    # Nastavení velikosti mřížky pro zobrazení obrázků.
    num_rows = 3 # Počet řádků v mřížce
    num_cols = 3 # Počet sloupců v mřížce
    num_images_to_show = num_rows * num_cols # Celkový počet obrázků, které se zobrazí
    # Vytvoření subplotů (podgrafů) pro zobrazení obrázků.
    # 'fig' je celý obrázek (okno), 'axes' je pole jednotlivých subplotů.
    # figsize=(8, 8) nastavuje velikost celého okna na 8x8 palců.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 8))
    # Nastavení hlavního nadpisu pro celé okno s obrázky.
    fig.suptitle("Náhodné obrázky z MNIST s popisky", fontsize=16)

    # Vybrat náhodné indexy z trénovací sady.
    # np.random.choice() vybere 'num_images_to_show' unikátních indexů
    # (protože replace=False) z celkového počtu obrázků v x_train.
    random_indices = np.random.choice(x_train.shape[0], num_images_to_show, replace=False)

    # Procházení jednotlivých subplotů a zobrazení obrázků.
    # 'axes.flat' umožňuje iterovat přes všechny subplaty v poli 'axes' jako jednorozměrné pole.
    for i, ax in enumerate(axes.flat):
        # Kontrola, zda je aktuální subplot určen pro zobrazení obrázku.
        # (měli bychom zobrazit pouze 'num_images_to_show' obrázků)
        # Podmínka 'if i < num_images_to_show' je vždy splněna,
        # protože 'num_images_to_show' je nastaveno na celkový počet subplotů.
        # Zbytečný 'else' blok byl odstraněn.
            index = random_indices[i] # Získání náhodně vybraného indexu pro aktuální obrázek
            # Zobrazení obrázku v aktuálním subplotu.
            # 'cmap='gray'' zajistí, že obrázek bude zobrazen ve stupních šedi,
            # což je pro MNIST typické.
            ax.imshow(x_train[index], cmap='gray')
            # Nastavení titulku subplotu s příslušným popiskem (skutečná číslice).
            # y_train[index] obsahuje popisek pro obrázek na daném indexu.
            ax.set_title(f"Popisek: {y_train[index]}")
            # Vypnutí os (číselných označení) pro čistší zobrazení obrázků.
            ax.axis('off')
    # Automatické upravení rozložení subplotů tak, aby se nepřekrývaly.
    # 'rect' definuje obdélník pro obsah subplotů, aby se udělalo místo pro 'fig.suptitle'.
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Zobrazení celého grafu (okna s obrázky).
    plt.show()

# Tato část kódu se spustí pouze tehdy, když je skript spuštěn přímo (ne jako importovaný modul).
if __name__ == "__main__":
    show_random_digits() # Volání funkce pro zobrazení náhodných číslic

