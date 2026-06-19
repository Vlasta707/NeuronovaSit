import numpy as np
from PIL import Image
import os

# --- Nastavení cest ---
# ... (ostatní cesty a konstanty zůstávají stejné) ...

# Cílová číslice, kterou chceme identifikovat (v tomto případě '5')
TARGET_DIGIT = 5

def create_vlist_numpy_arrays():
    # ... (načítání obrázků - tato část zůstává beze změny) ...

    # --- Zpracování popisků ---
    print(f"Načítám popisky z '{LABELS_FILE}' a převádím na binární (5 vs. ostatní)...")
    try:
        with open(LABELS_FILE, 'r') as f:
            for line in f:
                try:
                    original_label = int(line.strip())
                    if 0 <= original_label <= 9:
                        # Zde je klíčová změna:
                        # Pokud je popisek cílová číslice (např. 5), přiřadíme 1.
                        # Jinak (pro ostatní číslice), přiřadíme 0.
                        binary_label = 1 if original_label == TARGET_DIGIT else 0
                        labels_list.append(binary_label)
                    else:
                        print(f"Upozornění: Původní popisek '{line.strip()}' není platná číslice (0-9). Přeskakuji.")
                except ValueError:
                    print(f"Upozornění: Nepodařilo se převést popisek '{line.strip()}' na celé číslo. Přeskakuji.")

    # ... (zbytek zpracování popisků a ukládání zůstává stejný) ...
    # Závěrečný tisk se mírně upraví pro lepší čitelnost, aby odrážel binární povahu.
    print(f"Popisky uloženy do: {labels_output_path} s tvarem {vlist_train_labels.shape} (kde 1 = '{TARGET_DIGIT}', 0 = 'ostatní')")
    print("Nyní můžete použít tyto .npy soubory ve vašem PyTorch datasetu.")

if __name__ == "__main__":
    create_vlist_numpy_arrays()