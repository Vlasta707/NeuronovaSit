import numpy as np
from PIL import Image
import os

# --- Společné nastavení rozlišení ---
# Zvětšeno z 28x28 na 256x256, aby byly vidět vady na produktech
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
OUTPUT_DIR = './data/'

def process_dataset(labels_file_path, image_dir_path, output_name):
    """
    Načte JPG obrázky na základě textového souboru s popisky,
    převede je do stupňů šedi, zmenší na 256x256 a uloží jako .npy pole.
    
    Očekávaný formát řádku v labels souboru: nazev_obrazku.jpg,1 (nebo 0)
    kde 1 = OK, 0 = BAD
    """
    if not os.path.exists(labels_file_path):
        print(f"Chyba: Soubor s popisky '{labels_file_path}' nenalezen. Přeskakuji tuto sadu.")
        return

    images_list = []
    labels_list = []

    print(f"Začínám zpracování sady podle seznamu: {labels_file_path}...")
    print(f"Hledám obrázky v adresáři: {image_dir_path}")

    # Čtení souboru řádek po řádku
    with open(labels_file_path, 'r') as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            
            # Přeskočit prázdné řádky nebo řádky bez čárky
            if not line or ',' not in line:
                continue
            
            try:
                # Rozdělení řádku na název souboru a popisek
                filename, label_str = line.split(',')
                filepath = os.path.join(image_dir_path, filename)

                # Kontrola, zda fyzický obrázek existuje na disku
                if not os.path.exists(filepath):
                    print(f"Upozornění (řádek {idx}): Obrázek {filepath} nenalezen. Přeskakuji.")
                    continue

                # --- Načtení a úprava obrázku ---
                img = Image.open(filepath)
                img = img.convert('L') # Převedení na stupně šedi (Grayscale / Luminance)
                img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS) # Vysoce kvalitní zmenšení
                
                # Převod na NumPy pole (hodnoty pixelů 0-255)
                img_array = np.array(img)
                
                images_list.append(img_array)
                labels_list.append(int(label_str))

            except Exception as e:
                print(f"Chyba na řádku {idx} při zpracování souboru: {e}")
                continue

    if not images_list:
        print(f"Chyba: Ze souboru {labels_file_path} se nepodařilo načíst žádné platné obrázky.\n")
        return

    # Převod seznamů na finální NumPy pole
    vlist_images = np.array(images_list, dtype=np.uint8) # uint8 (0 až 255) šetří místo
    vlist_labels = np.array(labels_list, dtype=np.int64) # int64 vyžaduje PyTorch pro CrossEntropyLoss

    # Vytvoření výstupní složky, pokud neexistuje
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Definice cest pro uložení .npy souborů
    images_output_path = os.path.join(OUTPUT_DIR, f'{output_name}_images.npy')
    labels_output_path = os.path.join(OUTPUT_DIR, f'{output_name}_labels.npy')

    # Uložení dat do souborů na disk
    np.save(images_output_path, vlist_images)
    np.save(labels_output_path, vlist_labels)

    print("--- Hotovo ---")
    print(f"Obrázky uloženy do: {images_output_path} s tvarem {vlist_images.shape}")
    print(f"Popisky uloženy do: {labels_output_path} s tvarem {vlist_labels.shape}")
    print(f"Párování: 1 = 'OK', 0 = 'BAD'\n")


if __name__ == "__main__":
    # 1. Zpracování TRÉNOVACÍ sady
    process_dataset(
        labels_file_path='./vlist_data/labels.txt',   # Textový soubor s popisy trénovacích dat
        image_dir_path='./vlist_data/images/',        # Složka s trénovacími .jpg obrázky
        output_name='vlist_train'                     # Vznikne vlist_train_images.npy a vlist_train_labels.npy
    )

    # 2. Zpracování TESTOVACÍ sady
    process_dataset(
        labels_file_path='./vlist_data/test_labels.txt', # Textový soubor s popisy testovacích dat
        image_dir_path='./vlist_data/test_images/',     # Složka s testovacími .jpg obrázky
        output_name='vlist_test'                        # Vznikne vlist_test_images.npy a vlist_test_labels.npy
    )
