import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk
import os
import tkinter as tk
from tkinter import messagebox, Toplevel

# --- Konfigurační cesty pro uložení posledního modelu ---
# Konfigurační soubor bude uložen v domovském adresáři uživatele.
CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".prumyslova_sit_app")
LAST_MODEL_FILE = os.path.join(CONFIG_DIR, "last_model.txt")

# Adresář, kde se nacházejí tvé modely.
# Podle tvého skriptu se modely hledají v aktuálním adresáři.
MODELS_DIR = "." # "." znamená aktuální adresář

# --- Funkce pro persistenci cesty k modelu ---
def load_last_model_path():
    """Načte cestu k naposledy použitému modelu z konfiguračního souboru."""
    if os.path.exists(LAST_MODEL_FILE):
        try:
            with open(LAST_MODEL_FILE, 'r') as f:
                path = f.read().strip()
                # Zkontrolujeme, zda cesta k modelu stále existuje
                # a zda je ve stejném adresáři jako tento skript (pokud MODELS_DIR je '.')
                if os.path.exists(path) and os.path.dirname(path) == MODELS_DIR:
                    return path
        except Exception as e:
            print(f"Chyba při načítání cesty k předchozímu modelu: {e}")
    return None

def save_last_model_path(path):
    """Uloží cestu k aktuálně použitému modelu do konfiguračního souboru."""
    os.makedirs(CONFIG_DIR, exist_ok=True) # Zajistí, že adresář existuje
    try:
        with open(LAST_MODEL_FILE, 'w') as f:
            f.write(path)
    except Exception as e:
        print(f"Chyba při ukládání cesty k předchozímu modelu: {e}")


# --- 1. Definice architektury CNN (musí být shodná s tréninkovým modelem) ---
class PrumyslovaSit(nn.Module):
    def __init__(self):
        super(PrumyslovaSit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128) # Po třech poolech
        self.fc2 = nn.Linear(128, 2) # Výstup: 2 třídy (0 = BAD, 1 = OK)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- 2. Nastavení transformací pro vstupní obrázek ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- 3. Funkce pro klasifikaci jednoho obrázku ---
def classify_image(model, image_path, device, transform, class_names):
    if not os.path.exists(image_path):
        print(f"Chyba: Soubor obrázku '{image_path}' nebyl nalezen.")
        return None

    try:
        image = Image.open(image_path).convert('L')
    except Exception as e:
        print(f"Chyba při načítání nebo zpracování obrázku '{image_path}': {e}")
        return None

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_prob, predicted_idx = torch.max(probabilities, 1)

    prediction = class_names[predicted_idx.item()]
    confidence = predicted_prob.item() * 100

    return prediction, confidence

# --- 4. Hlavní část skriptu ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Používám zařízení: {device}")

    class_names = {0: "BAD", 1: "OK"}

    main_tk_root = tk.Tk()
    main_tk_root.withdraw()

    selected_model_container = []
    selected_image_container = []

    # Načteme naposledy použitý model z konfigurace
    last_used_model_path = load_last_model_path()
    # Získáme pouze název souboru pro zobrazení
    last_used_model_filename = os.path.basename(last_used_model_path) if last_used_model_path else None

    # --- 4.1. Grafický výběr modelu pomocí Tkinter (jako Toplevel okno) ---
    model_select_window = Toplevel(main_tk_root)
    model_select_window.title("Výběr modelu sítě")
    model_select_window.geometry("350x300") # Zvětšeno pro nové tlačítko

    label = tk.Label(model_select_window, text="Zvolte soubor natrénovaného modelu:", font=("Arial", 10, "bold"))
    label.pack(pady=10)

    listbox = tk.Listbox(model_select_window, width=40, height=8)
    model_filenames_in_listbox = [] # Ukládáme si názvy souborů pro snadnou kontrolu existence

    models_found = False
    # Procházíme soubory v MODELS_DIR (aktuálním adresáři)
    for i, file in enumerate(sorted(os.listdir(MODELS_DIR))):
        if file.endswith('.pth') or file.endswith('.pt'):
            listbox.insert(tk.END, f"{file} - {i+1}")
            model_filenames_in_listbox.append(file)
            models_found = True
    listbox.pack(pady=5)

    if not models_found:
        messagebox.showerror("Chyba", f"V adresáři '{MODELS_DIR}' nebyly nalezeny žádné modely (.pth/.pt)!")
        model_select_window.destroy()
        main_tk_root.destroy()
        exit()

    # Automaticky vybrat a zvýraznit naposledy použitý model, pokud existuje a je dostupný
    if last_used_model_filename and last_used_model_filename in model_filenames_in_listbox:
        try:
            index_to_select = -1
            for i, item_text in enumerate(listbox.get(0, tk.END)):
                if item_text.startswith(last_used_model_filename):
                    index_to_select = i
                    break
            if index_to_select != -1:
                listbox.selection_set(index_to_select) # Zvýrazní položku
                listbox.activate(index_to_select)     # Nastaví ji jako aktivní
                listbox.see(index_to_select)          # Roluje na ní, pokud není viditelná
        except Exception as e:
            print(f"Chyba při automatickém výběru předchozího modelu: {e}")

    # Funkce pro výběr modelu, která může být volána buď z listboxu nebo z tlačítka "Načíst předchozí"
    def select_model(use_last_model_flag=False):
        model_to_select_filename = None
        if use_last_model_flag and last_used_model_filename and last_used_model_filename in model_filenames_in_listbox:
            # Pokud je voláno tlačítkem "Načíst předchozí" a model je dostupný
            model_to_select_filename = last_used_model_filename
            print(f"Vybírám předchozí model: {model_to_select_filename}")
        else:
            # Jinak se pokusíme vybrat z listboxu
            try:
                selection_index = listbox.curselection()[0]
                model_to_select_filename = listbox.get(selection_index).split(" - ")[0]
                print(f"Vybírám model z listboxu: {model_to_select_filename}")
            except IndexError:
                messagebox.showwarning("Upozornění", "Musíte nejdříve kliknutím vybrat model ze seznamu!")
                return

        if model_to_select_filename:
            selected_model_container.append(model_to_select_filename)
            model_select_window.destroy()

    # Tlačítko pro načtení vybraného modelu z listboxu
    button = tk.Button(model_select_window, text="Načíst vybraný model", command=lambda: select_model(False), bg="#4CAF50", fg="white")
    button.pack(pady=10)

    # Nové tlačítko pro výběr naposledy použitého modelu
    if last_used_model_filename and last_used_model_filename in model_filenames_in_listbox:
        default_button_text = f"Načíst předchozí: {last_used_model_filename}"
        default_button = tk.Button(
            model_select_window, text=default_button_text,
            command=lambda: select_model(True), # Voláme select_model s příznakem pro použití předchozího
            bg="#008CBA", fg="white" # Modrá barva pro odlišení
        )
        default_button.pack(pady=5)
    else:
        # Zobrazíme zprávu, pokud předchozí model není k dispozici
        tk.Label(model_select_window, text="Žádný předchozí model není k dispozici.").pack(pady=5)

    main_tk_root.wait_window(model_select_window)

    # --- 4.2. Načtení zvoleného modelu do PyTorchu ---
    if not selected_model_container:
        print("Výběr modelu byl zrušen. Ukončuji skript.")
        main_tk_root.destroy()
        exit()

    model_name = selected_model_container[0]
    model_path = os.path.join(MODELS_DIR, model_name) # Sestavíme plnou cestu k modelu

    model = PrumyslovaSit().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\nModel '{model_name}' byl úspěšně načten.")
        save_last_model_path(model_path) # Uložíme cestu k právě načtenému modelu jako poslední

        # --- 4.3. Grafický výběr obrázku k klasifikaci (jako Toplevel okno) ---
        image_dir = './syrova_data'

        if not os.path.exists(image_dir):
            messagebox.showerror("Chyba", f"Adresář '{image_dir}' nebyl nalezen. Vytvořte jej a vložte do něj obrázky .png pro klasifikaci.")
            main_tk_root.destroy()
            exit()

        image_select_window = Toplevel(main_tk_root)
        image_select_window.title("Výběr obrázku k klasifikaci")
        image_select_window.geometry("400x300")

        image_label = tk.Label(image_select_window, text=f"Zvolte obrázek .png z adresáře '{image_dir}':", font=("Arial", 10, "bold"))
        image_label.pack(pady=10)

        image_listbox = tk.Listbox(image_select_window, width=50, height=10)

        image_files_found = False
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])

        if image_files:
            for i, file_name in enumerate(image_files):
                image_listbox.insert(tk.END, f"{file_name} - {i+1}")
            image_files_found = True
        image_listbox.pack(pady=5)

        if not image_files_found:
            messagebox.showerror("Chyba", f"V adresáři '{image_dir}' nebyly nalezeny žádné obrázky .png!")
            image_select_window.destroy()
            main_tk_root.destroy()
            exit()

        def select_image():
            try:
                selection_index = image_listbox.curselection()[0]
                selected_file_name = image_listbox.get(selection_index).split(" - ")[0]
                selected_image_container.append(os.path.join(image_dir, selected_file_name))
                image_select_window.destroy()
            except IndexError:
                messagebox.showwarning("Upozornění", "Musíte nejdříve kliknutím vybrat obrázek ze seznamu!")

        image_button = tk.Button(image_select_window, text="Klasifikovat vybraný obrázek", command=select_image, bg="#008CBA", fg="white")
        image_button.pack(pady=10)

        main_tk_root.wait_window(image_select_window)

        if not selected_image_container:
            print("Výběr obrázku byl zrušen. Ukončuji skript.")
            main_tk_root.destroy()
            exit()

        final_image_path = selected_image_container[0]

        # --- Zobrazení vybraného obrázku na obrazovce ---
        print(f"Zobrazení vybraného obrázku: {final_image_path}")
        try:
            img = Image.open(final_image_path)

            max_size = (600, 600)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            img_tk = ImageTk.PhotoImage(img)

            image_display_window = Toplevel(main_tk_root)
            image_display_window.title(f"Vybraný obrázek: {os.path.basename(final_image_path)}")

            image_display_window.protocol("WM_DELETE_WINDOW", image_display_window.destroy)

            panel = tk.Label(image_display_window, image=img_tk)
            panel.image = img_tk
            panel.pack(padx=10, pady=10)

            close_button = tk.Button(image_display_window, text="Zavřít náhled a pokračovat", command=image_display_window.destroy)
            close_button.pack(pady=5)

            image_display_window.update_idletasks()
            screen_width = main_tk_root.winfo_screenwidth()
            screen_height = main_tk_root.winfo_screenheight()
            x = (screen_width // 2) - (image_display_window.winfo_width() // 2)
            y = (screen_height // 2) - (image_display_window.winfo_height() // 2)
            image_display_window.geometry(f"+{x}+{y}")

            main_tk_root.wait_window(image_display_window)

        except Exception as e:
            print(f"Chyba při zobrazení obrázku: {e}")
            messagebox.showerror("Chyba zobrazení", f"Nepodařilo se zobrazit obrázek: {e}")

        # --- 4.4. Spuštění klasifikace ---
        result = classify_image(model, final_image_path, device, transform, class_names)

        if result:
            prediction, confidence = result
            print(f"\nPredikce pro obrázek '{final_image_path}':")
            print(f"Třída: {prediction}")
            print(f"Spolehlivost (Confidence): {confidence:.2f}%")
        else:
            print("Klasifikace obrázku selhala.")

    except Exception as e:
        print(f"Chyba při načítání nebo použití  modelu: {e}")
    finally:
        if main_tk_root.winfo_exists():
            main_tk_root.destroy()