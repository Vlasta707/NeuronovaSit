import numpy as np
from PIL import Image
import os

# --- Společné nastavení rozlišení ---
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
OUTPUT_DIR = './data/'

def process_dataset(labels_file_path, image_dir_path, output_name):
    """
    Načte JPG obrázky na základě textového souboru s popisky,
    převede je do stupňů šedi (1 kanál), zmenší na 256x256 a uloží jako .npy pole.
    """
    if not os.path.exists(labels_file_path):
        print(f"Chyba: Soubor s popisky '{labels_file_path}' nenalezen. Přeskakuji tuto sadu.")
        return

    images_list = []
    labels_list = []

    # Počítadla pro kontrolu distribuce tříd
    count_ok = 0
    count_bad = 0

    print(f"Začínám zpracování sady podle seznamu: {labels_file_path}...")
    print(f"Hledám obrázky v adresáři: {image_dir_path}")

    with open(labels_file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            
            if not line or ',' not in line:
                continue
            
            try:
                # Rozdělení řádku a odstranění případných neviditelných mezer
                filename, label_str = line.split(',')
                filename = filename.strip()
                label_str = label_str.strip()
                
                filepath = os.path.join(image_dir_path, filename)

                if not os.path.exists(filepath):
                    print(f"Upozornění (řádek {idx}): Obrázek {filepath} nenalezen. Přeskakuji.")
                    continue

                # --- Načtení a úprava obrázku ---
                img = Image.open(filepath)
                img = img.convert('L') # Ponecháváme 1 kanál (stupně šedi) dle Varianty A
                
                # UPRAVENO: Použito přímo Image.LANCZOS pro lepší kompatibilitu napříč verzemi PIL
                img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS) 
                
                img_array = np.array(img)
                
                images_list.append(img_array)
                
                label_val = int(label_str)
                labels_list.append(label_val)

                # Přičtení do statistiky
                if label_val == 1:
                    count_ok += 1
                elif label_val == 0:
                    count_bad += 1

            except Exception as e:
                print(f"Chyba na řádku {idx} při zpracování souboru: {e}")
                continue

    if not images_list:
        print(f"Chyba: Ze souboru {labels_file_path} se nepodařilo načíst žádné platné obrázky.\n")
        return

    # Převod seznamů na finální NumPy pole
    vlist_images = np.array(images_list, dtype=np.uint8) 
    vlist_labels = np.array(labels_list, dtype=np.int64) 

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images_output_path = os.path.join(OUTPUT_DIR, f'{output_name}_images.npy')
    labels_output_path = os.path.join(OUTPUT_DIR, f'{output_name}_labels.npy')

    np.save(images_output_path, vlist_images)
    np.save(labels_output_path, vlist_labels)

    print("--- Hotovo ---")
    print(f"Obrázky uloženy do: {images_output_path} s tvarem {vlist_images.shape}")
    print(f"Popisky uloženy do: {labels_output_path} s tvarem {vlist_labels.shape}")
    print(f"Statistika sady -> OK (1): {count_ok}x | BAD (0): {count_bad}x")
    print(f"Párování: 1 = 'OK', 0 = 'BAD'\n")


if __name__ == "__main__":
    # 1. Zpracování TRÉNOVACÍ sady
    process_dataset(
        labels_file_path='./vlist_data/labels.txt',   
        image_dir_path='./vlist_data/images/',        
        output_name='vlist_train'                     
    )

    # 2. Zpracování TESTOVACÍ sady
    process_dataset(
        labels_file_path='./vlist_data/test_labels.txt', 
        image_dir_path='./vlist_data/test_images/',     
        output_name='vlist_test'                        
    )