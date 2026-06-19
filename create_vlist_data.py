import numpy as np
from PIL import Image
import os

# --- Nastavení cest ---
IMAGE_DIR = './vlist_data/images/'
LABELS_FILE = './vlist_data/labels.txt'
OUTPUT_DIR = './data/'

# Rozměry obrázku (MNIST je 28x28 pixelů)
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
# Počet obrázků
NUM_IMAGES = 1000

# --- Nastavení pro binární klasifikaci ---
# Zde definujeme, která číslice bude považována za "OK".
# Všechny ostatní číslice budou automaticky "BAD".
OK_DIGIT = 5

# Mapování binárních popisků: 1 pro "OK", 0 pro "BAD"
LABEL_OK = 1
LABEL_BAD = 0

def create_vlist_numpy_arrays():
    """
    Načte JPG obrázky a jejich popisky, převede je na formát MNIST
    a uloží jako NumPy pole do souborů .npy pro binární klasifikaci "OK" vs "BAD".
    """
    print(f"Začínám zpracování {NUM_IMAGES} obrázků a popisků pro klasifikaci 'OK' (číslice {OK_DIGIT}) vs. 'BAD' (ostatní)...")

    # --- Zpracování popisků ---
    print(f"Načítám popisky z '{LABELS_FILE}' a převádím na binární (OK={OK_DIGIT} vs. BAD=ostatní)...")
    try:
        with open(LABELS_FILE, 'r') as f:
            for line in f:
                try:
                    original_label = int(line.strip())
                    if 0 <= original_label <= 9:
                        # Klíčová změna zde: mapování na 'OK' nebo 'BAD'
                        binary_label = LABEL_OK if original_label == OK_DIGIT else LABEL_BAD
                        labels_list.append(binary_label)
                    else:
                        print(f"Upozornění: Původní popisek '{line.strip()}' není platná číslice (0-9). Přeskakuji.")
                except ValueError:
                    print(f"Upozornění: Nepodařilo se převést popisek '{line.strip()}' na celé číslo. Přeskakuji.")

    except FileNotFoundError:
        print(f"Chyba: Soubor s popisky '{LABELS_FILE}' nenalezen. Zkontrolujte cestu.")
        return
    except Exception as e:
        print(f"Chyba při čtení souboru popisků: {e}")
        return

    if len(labels_list) != NUM_IMAGES:
        print(f"Upozornění: Počet načtených popisků ({len(labels_list)}) se neshoduje s očekávaným počtem obrázků ({NUM_IMAGES}).")

    vlist_train_labels = np.array(labels_list, dtype=np.int64)

    # --- Uložení výsledných NumPy polí ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images_output_path = os.path.join(OUTPUT_DIR, 'vlist_train_images.npy')
    labels_output_path = os.path.join(OUTPUT_DIR, 'vlist_train_labels.npy')

    np.save(images_output_path, vlist_train_images)
    np.save(labels_output_path, vlist_train_labels)

    print("\n--- Hotovo ---")
    print(f"Obrázky uloženy do: {images_output_path} s tvarem {vlist_train_images.shape}")
    print(f"Popisky uloženy do: {labels_output_path} s tvarem {vlist_train_labels.shape} (kde {LABEL_OK} = 'OK' (číslice {OK_DIGIT}), {LABEL_BAD} = 'BAD' (ostatní číslice))")
    print("Nyní můžete použít tyto .npy soubory ve vašem PyTorch datasetu.")

if __name__ == "__main__":
    create_vlist_numpy_arrays()