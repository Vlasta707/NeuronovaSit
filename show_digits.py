# Import knihovny Matplotlib pro vytváření grafů a zobrazování obrázků
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
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


class InteractiveGrid:
    def __init__(self, all_image_paths):
        self.all_image_paths = all_image_paths
        self.num_rows = 4
        self.num_cols = 4
        self.num_images_to_show = self.num_rows * self.num_cols
        self.target_display_size = (256, 256)
        
        # Inicializace grafického okna (Fig) a mřížky sub-grafů (Axes) POUZE JEDNOU při startu
        self.fig, self.axes = plt.subplots(self.num_rows, self.num_cols, figsize=(13, 13))
        self.fig.suptitle(f"Náhodné obrázky z '{os.path.basename(IMAGE_SOURCE_DIR.rstrip('/'))}'", fontsize=16)
        
        # Skryjeme osy pro všechny sub-grafy hned na začátku
        for ax in self.axes.flat:
            ax.axis('off')
            
        # Registrace zavíracího eventu pro křížek okna
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        # Vytvoření tlačítek pouze jednou na začátku
        self.create_buttons()

    def show_next_batch(self, event=None):
        """Vybere náhodnou dávku obrázků a překreslí okno bez zásahu do rozvržení."""
        images_to_process_count = min(self.num_images_to_show, len(self.all_image_paths))
        random_selected_paths = random.sample(self.all_image_paths, images_to_process_count)

        processed_data = []
        for img_path in random_selected_paths:
            try:
                img = Image.open(img_path).convert('L')
                img = img.resize(self.target_display_size, Image.Resampling.LANCZOS)
                img_array = np.array(img) / 255.0
                processed_data.append((img_array, os.path.basename(img_path)))
            except Exception as e:
                print(f"Varování: Nelze zpracovat obrázek '{os.path.basename(img_path)}': {e}.")
                continue

        if not processed_data:
            print("Žádné obrázky nebylo možné úspěšně načíst.")
            return

        num_to_display_actual = len(processed_data)
        
        # Vykreslení/překreslení dat do již existujících os
        for i, ax in enumerate(self.axes.flat):
            ax.clear()  # Vyčistí stará obrazová data v dané sub-ose
            ax.axis('off') # clear() občas resetuje skrytí os, pro jistotu vypínáme znovu
            
            if i < num_to_display_actual:
                img_array, file_name = processed_data[i]
                ax.imshow(img_array, cmap='gray')
                ax.set_title(file_name, fontsize=10) # Větší font (10) zůstává zachován
            else:
                # Pokud by došlo k nedostatku obrázků, prázdné sloty úplně vyčistíme
                ax.imshow(np.zeros(self.target_display_size), cmap='gray')

        # tight_layout voláme bezpečně, protože se počet os (Axes) v okně nijak nemění
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        self.fig.canvas.draw_idle()

    def create_buttons(self):
        """Vytvoří interaktivní tlačítka pod mřížkou obrázků."""
        # [vlevo, dole, šířka, výška] v relativních souřadnicích okna (0 až 1)
        ax_next = plt.axes([0.30, 0.02, 0.18, 0.04])
        ax_exit = plt.axes([0.52, 0.02, 0.18, 0.04])
        
        # Stylování tlačítek
        self.btn_next = Button(ax_next, 'Další matice', color='#008CBA', hovercolor='#007399')
        self.btn_exit = Button(ax_exit, 'Ukončit', color='#f44336', hovercolor='#d32f2f')
        
        # Nastavení barvy textu na tlačítkách
        self.btn_next.label.set_color('white')
        self.btn_next.label.set_fontsize(11)
        self.btn_exit.label.set_color('white')
        self.btn_exit.label.set_fontsize(11)
        
        # Propojení tlačítek s funkcemi
        self.btn_next.on_clicked(self.show_next_batch)
        self.btn_exit.on_clicked(self.close_script)

    def close_script(self, event):
        """Funkce pro tlačítko Ukončit."""
        print("Ukončuji skript přes tlačítko na obrazovce. Na shledanou!")
        plt.close(self.fig)

    def on_close(self, event):
        """Vyčistí reference při zavření okna křížkem."""
        pass


if __name__ == "__main__":
    paths = load_all_image_paths()
    
    if paths:
        # Inicializace objektu mřížky (zde se okno a tlačítka vybudují poprvé a naposledy)
        grid = InteractiveGrid(paths)
        # Načteme do připraveného okna první sadu obrázků
        grid.show_next_batch()
        # Spuštění hlavní smyčky Matplotlibu
        plt.show()