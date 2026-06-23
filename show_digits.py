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
IMAGE_SOURCE_DIR = '/home/vlastik/NeuronovaSit/syrova_data/'

def load_all_image_paths():
    """Načte seznam všech dostupných PNG obrázků jednou na začátku."""
    all_image_paths = []
    if os.path.exists(IMAGE_SOURCE_DIR):
        for filename in os.listdir(IMAGE_SOURCE_DIR):
            if filename.lower().endswith('.png'):
                all_image_paths.append(os.path.join(IMAGE_SOURCE_DIR, filename))
    else:
        print(f"Chyba: Adresář '{IMAGE_SOURCE_DIR}' nebyl nalezen. Zkontrolujte cestu.")
        return None

    if not all_image_paths:
        print(f"Chyba: V adresáři '{IMAGE_SOURCE_DIR}' nebyly nalezeny žádné obrázky .png.")
        return None
        
    return all_image_paths


def show_random_batch(all_image_paths):
    """Vybere náhodnou dávku obrázků a zobrazí ji v mřížce."""
    num_rows = 4
    num_cols = 4
    num_images_to_show = num_rows * num_cols
    target_display_size = (256, 256)

    # --- 1. Vybrat náhodné obrázky k zobrazení ---
    images_to_process_count = min(num_images_to_show, len(all_image_paths))
    random_selected_paths = random.sample(all_image_paths, images_to_process_count)

    processed_data = []

    # --- 2. Načtení a předzpracování vybraných obrázků ---
    for img_path in random_selected_paths:
        try:
            img = Image.open(img_path).convert('L')
            img = img.resize(target_display_size, Image.Resampling.LANCZOS)
            img_array = np.array(img) / 255.0
            processed_data.append((img_array, os.path.basename(img_path)))
        except Exception as e:
            print(f"Varování: Nelze zpracovat obrázek '{os.path.basename(img_path)}': {e}. Bude přeskočen.")
            continue

    if not processed_data:
        print("Žádné obrázky nebylo možné úspěšně načíst a zpracovat pro zobrazení.")
        return

    # --- 3. Zobrazení obrázků v mřížce ---
    num_to_display_actual = len(processed_data)
    
    # Používáme explicitní zavření předchozích oken, aby se nehromadila v paměti
    plt.close('all') 
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
    fig.suptitle(f"Náhodné obrázky z '{os.path.basename(IMAGE_SOURCE_DIR.rstrip('/'))}'", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < num_to_display_actual:
            img_array, file_name = processed_data[i]
            ax.imshow(img_array, cmap='gray')
            ax.set_title(file_name, fontsize=6)
        else:
            ax.set_visible(False)
            
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("Zobrazuji okno s maticí obrázků. Pro pokračování okno zavřete...")
    plt.show()


if __name__ == "__main__":
    # Načteme seznam cest k obrázkům (pouze jednou)
    paths = load_all_image_paths()
    
    if paths:
        # Spuštění interaktivní smyčky
        while True:
            show_random_batch(paths)
            
            # Dotaz v terminálu po zavření grafického okna
            user_choice = input("\nChcete zobrazit další náhodnou matici? (Ano = Enter / Ne = napište cokoliv a Enter): ").strip().lower()
            
            # Pokud uživatel zadá jakýkoliv znak (např. 'n', 'no', 'exit'), skript skončí. 
            # Pokud jen zmáčkne Enter (prázdný řetězec), cyklus pokračuje.
            if user_choice != '':
                print("Ukončuji skript. Na shledanou!")
                break