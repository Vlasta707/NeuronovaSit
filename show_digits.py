# Import knihovny Matplotlib pro vytváření grafů a zobrazování obrázků
import matplotlib.pyplot as plt
# Import knihovny NumPy pro práci s numerickými daty, zejména poli (arrays)
import numpy as np
# Import os pro práci se souborovým systémem (seznam souborů, cesty)
import os
# Import Image z knihovny Pillow pro načítání a manipulaci s obrázky
from PIL import Image
# Import random pro náhodný výběr souborů
import random


# Definujte cestu k vašemu adresáři s obrázky
# Ujistěte se, že tato cesta je správná a adresář existuje
IMAGE_SOURCE_DIR = '/home/vlastik/NeuronovaSit/syrova_data/'
# IMAGE_SOURCE_DIR = './syrova_data/' # Alternativně, pokud je adresář v kořeni projektu

def show_random_custom_images():
    """
    Načte náhodné PNG obrázky z předdefinovaného adresáře, zpracuje je
    a zobrazí v mřížce 10x10 s názvy souborů jako popisky.
    """
    num_rows = 4
    num_cols = 4
    num_images_to_show = num_rows * num_cols
    target_display_size = (256, 256) # Cílová velikost pro zobrazení v mřížce (např. 64x64 pixelů)

    # --- 1. Získání seznamu všech PNG souborů z adresáře ---
    all_image_paths = []
    if os.path.exists(IMAGE_SOURCE_DIR):
        for filename in os.listdir(IMAGE_SOURCE_DIR):
            if filename.lower().endswith('.png'):
                all_image_paths.append(os.path.join(IMAGE_SOURCE_DIR, filename))
    else:
        print(f"Chyba: Adresář '{IMAGE_SOURCE_DIR}' nebyl nalezen. Zkontrolujte cestu.")
        return

    if not all_image_paths:
        print(f"Chyba: V adresáři '{IMAGE_SOURCE_DIR}' nebyly nalezeny žádné obrázky .png.")
        return

    # --- 2. Vybrat náhodné obrázky k zobrazení ---
    # Pokud je méně obrázků než num_images_to_show, vybereme všechny dostupné
    images_to_process_count = min(num_images_to_show, len(all_image_paths))
    random_selected_paths = random.sample(all_image_paths, images_to_process_count)

    processed_data = [] # Bude obsahovat tuple (numpy_array_obrazku, nazev_souboru)

    # --- 3. Načtení a předzpracování vybraných obrázků ---
    for img_path in random_selected_paths:
        try:
            # Načte obrázek, převede ho na šedotónový ('L' mode)
            img = Image.open(img_path).convert('L')
            # Změní velikost na cílový rozměr pro zobrazení
            img = img.resize(target_display_size, Image.Resampling.LANCZOS)
            # Převede na NumPy pole a normalizuje hodnoty pixelů na rozsah 0-1
            img_array = np.array(img) / 255.0
            processed_data.append((img_array, os.path.basename(img_path)))
        except Exception as e:
            print(f"Varování: Nelze zpracovat obrázek '{os.path.basename(img_path)}': {e}. Bude přeskočen.")
            continue # Pokračuje na další obrázek

    if not processed_data:
        print("Žádné obrázky nebylo možné úspěšně načíst a zpracovat pro zobrazení.")
        return

    # --- 4. Zobrazení obrázků v mřížce ---
    num_to_display_actual = len(processed_data)
    if num_to_display_actual < num_images_to_show:
        print(f"Poznámka: K dispozici je pouze {num_to_display_actual} obrázků. Mřížka nebude zcela zaplněna.")

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
    fig.suptitle(f"Náhodné obrázky z '{os.path.basename(IMAGE_SOURCE_DIR)}'", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < num_to_display_actual:
            img_array, file_name = processed_data[i]
            ax.imshow(img_array, cmap='gray')
            ax.set_title(file_name, fontsize=6) # Titulek bude název souboru
        else:
            # Pokud nemáme dostatek obrázků, skryjeme zbývající prázdné subplaty
            ax.set_visible(False)
            
        ax.axis('off') # Vypnutí os pro čistší zobrazení

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    show_random_custom_images()