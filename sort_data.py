import os
import shutil

ZDROJOVA_SLOZKA = "./vlist_data/images"  # Cesta k tvé složce, kde je vše pohromadě
VYSTUPNI_DATASET = "./dataset"        # Cesta, kam se vytvoří struktura pro nový train.py

# Vytvoření cílových podsložek
os.makedirs(os.path.join(VYSTUPNI_DATASET, "OK"), exist_ok=True)
os.makedirs(os.path.join(VYSTUPNI_DATASET, "BAD"), exist_ok=True)

if not os.path.exists(ZDROJOVA_SLOZKA):
    print(f"Chyba: Složka {ZDROJOVA_SLOZKA} neexistuje!")
    exit()

soubory = [f for f in os.listdir(ZDROJOVA_SLOZKA) if f.lower().endswith('.png')]

for soubor in soubory:
    jmeno_bez_pripony = os.path.splitext(soubor)[0]
    
    # Logika: pokud končí na a, b, c -> jde o BAD. Jinak OK.
    if jmeno_bez_pripony.endswith(('a', 'b', 'c')):
        cil = os.path.join(VYSTUPNI_DATASET, "BAD", soubor)
    else:
        cil = os.path.join(VYSTUPNI_DATASET, "OK", soubor)
        
    shutil.copy(os.path.join(ZDROJOVA_SLOZKA, soubor), cil)

print(f"Hotovo! Obrázky byly zkopírovány a roztříděny do složky '{VYSTUPNI_DATASET}'.")
print(f"OK: {len(os.listdir(os.path.join(VYSTUPNI_DATASET, 'OK')))} obrázků")
print(f"BAD: {len(os.listdir(os.path.join(VYSTUPNI_DATASET, 'BAD')))} obrázků")