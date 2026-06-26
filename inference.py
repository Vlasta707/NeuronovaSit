import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk
import os
import tkinter as tk
from tkinter import messagebox, Toplevel, ttk

# Importy pro vykreslování grafů v Tkinteru
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# --- Konfigurační cesty pro uložení posledního modelu ---
CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".prumyslova_sit_app")
LAST_MODEL_FILE = os.path.join(CONFIG_DIR, "last_model.txt")

# ZMĚNA: Modely se nyní načítají z podadresáře "moje_modely"
MODELS_DIR = "./moje_modely" 

# Zajištění existence adresáře (prevence chyb při prvním spuštění)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Funkce pro persistenci cesty k modelu ---
def load_last_model_path():
    if os.path.exists(LAST_MODEL_FILE):
        try:
            with open(LAST_MODEL_FILE, 'r') as f:
                path = f.read().strip()
                # TOLERANTNÍ KONTROLA: Pokud stará cesta existuje, nebo pokud model existuje v nové složce
                if os.path.exists(path):
                    return path
                
                # Pokud v souboru byla stará cesta (z rootu), zkusíme se podívat, zda už model není v nové složce
                potential_new_path = os.path.join(MODELS_DIR, os.path.basename(path))
                if os.path.exists(potential_new_path):
                    return potential_new_path
        except Exception as e:
            print(f"Chyba při načítání cesty k předchozímu modelu: {e}")
    return None

def save_last_model_path(path):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    try:
        with open(LAST_MODEL_FILE, 'w') as f:
            f.write(path)
    except Exception as e:
        print(f"Chyba při ukládání cesty k předchozímu modelu: {e}")


# --- 1. Definice architektury CNN ---
class PrumyslovaSit(nn.Module):
    def __init__(self):
        super(PrumyslovaSit, self).__init__()
        # 1. konvoluční blok
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2. konvoluční blok
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # === DOPLNĚNÍ DO INFERENCE: 3. konvoluční blok ===
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Plně propojené vrstvy
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)
        
        self.relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 1. blok (256x256 -> 128x128)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # 2. blok (128x128 -> 64x64)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # === DOPLNĚNÍ DO INFERENCE: 3. blok (64x64 -> 32x32) ===
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Narovnání pro lineární vrstvu
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


# --- Pomocné funkce pro načítání dat z .md souboru ---
def parse_md_file(md_path):
    """Přečte .md soubor, vytáhne textový popis a data pro graf ztráty."""
    text_info = ""
    losses = []
    
    if not os.path.exists(md_path):
        return "K tomuto modelu nebyl nalezen .md soubor s vyhodnocením.", []

    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            # Extrakce dat pro graf z tabulky historie ztrát
            if "|" in line:
                if "Epocha" in line or "---" in line:
                    continue
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 3:
                    try:
                        loss_val = float(parts[2])
                        losses.append(loss_val)
                    except ValueError:
                        pass
            
            # Formátování textu pro zobrazení (očištění od tabulkových znaků)
            if line.startswith("#"):
                text_info += line.replace("#", "").strip() + "\n" + "="*30 + "\n"
            elif "|" in line:
                if "---" in line: continue
                parts = [p.strip() for p in line.split("|") if p.strip()]
                text_info += "  ".join(parts).replace("**", "") + "\n"
            else:
                text_info += line.replace("**", "")
                
    except Exception as e:
        text_info = f"Chyba při čtení statistik: {e}"
        
    return text_info, losses


# --- 4. Hlavní část skriptu ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Používám zařízení: {device}")

    class_names = {0: "BAD", 1: "OK"}

    # JEDINÝ ROOT PRO CELÝ SKRIPT - bude řídit celou aplikaci
    main_tk_root = tk.Tk()
    main_tk_root.withdraw()

    selected_model_container = []
    selected_image_container = []

    last_used_model_path = load_last_model_path()
    last_used_model_filename = os.path.basename(last_used_model_path) if last_used_model_path else None

    # Získání seznamu modelů v adresáři moje_modely
    model_filenames_in_listbox = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith('.pth') or f.endswith('.pt')])

    if not model_filenames_in_listbox:
        messagebox.showerror("Chyba", f"V adresáři '{MODELS_DIR}' nebyly nalezeny žádné modely (.pth/.pt)!")
        main_tk_root.destroy()
        exit()

    # Funkce pro zobrazení okna s grafem a detaily
    def show_model_details_and_confirm(model_filename):
        base_name = os.path.splitext(model_filename)[0]
        md_filename = f"{base_name}.md"
        md_path = os.path.join(MODELS_DIR, md_filename)
        
        # Načtení dat z markdownu
        text_info, loss_data = parse_md_file(md_path)
        
        details_window = Toplevel(main_tk_root)
        details_window.title(f"Detail modelu: {model_filename}")
        
        # Nastavení velikosti okna pro nové proporce
        if loss_data:
            # Širší okno pro zobrazení textu a výrazně širšího grafu
            details_window.geometry("1200x600") 
        else:
            # Užší okno, pokud není graf
            details_window.geometry("500x500") 
        
        # Hlavní rozřazovací rámec s grid layoutem pro kontrolu proporcí
        main_frame = tk.Frame(details_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Levá část: Textové info z MD
        left_frame = tk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5)) # Přidáno vodorovné odsazení
        
        lbl_info = tk.Label(left_frame, text="Statistiky trénování modelu:", font=("Arial", 10, "bold"))
        lbl_info.pack(anchor="w", fill=tk.X) # Fill horizontally
        
        txt_scroll = tk.Scrollbar(left_frame)
        txt_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_area = tk.Text(left_frame, wrap=tk.WORD, yscrollcommand=txt_scroll.set, font=("Consolas", 9))
        text_area.insert(tk.END, text_info)
        text_area.config(state=tk.DISABLED)
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        txt_scroll.config(command=text_area.yview)
        
        # Pravá část: Matplotlib Graf ztráty (pouze pokud máme data)
        if loss_data:
            right_frame = tk.Frame(main_frame)
            right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0)) # Přidáno vodorovné odsazení
            
            # Matplotlib Figure: Upravená figsize pro širší graf
            fig = Figure(figsize=(10, 5), dpi=100) # Např. 1000px široký, 500px vysoký
            ax = fig.add_subplot(111)
            ax.plot(loss_data, marker='.', color='#FF5722', label='Loss')
            ax.set_title("Průběh Ztráty (Loss History)")
            ax.set_xlabel("Průběh trénování (záznamy chyb)")
            ax.set_ylabel("Ztráta")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            
            canvas = FigureCanvasTkAgg(fig, master=right_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Konfigurace vah sloupců pro rozložení textu a grafu
            main_frame.grid_columnconfigure(0, weight=1) # Sloupec pro text (levý)
            main_frame.grid_columnconfigure(1, weight=5) # Sloupec pro graf (pravý) - 5x širší
        else:
            # Pokud není graf, textová část zabere celou šířku
            main_frame.grid_columnconfigure(0, weight=1)
            
        main_frame.grid_rowconfigure(0, weight=1) # Jediný řádek se roztáhne vertikálně
        
        # Spodní část: Tlačítka akcí
        btn_frame = tk.Frame(details_window)
        btn_frame.pack(fill=tk.X, pady=10)
        
        def on_confirm():
            selected_model_container.append(model_filename)
            details_window.destroy()
            main_tk_root.quit()  # Ukončí první mainloop
            
        def on_back():
            details_window.destroy()
            open_model_selection_window()
            
        btn_confirm = tk.Button(btn_frame, text="Ponechat tento model", command=on_confirm, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), padx=10, pady=5)
        btn_confirm.pack(side=tk.RIGHT, padx=20)
        
        btn_back = tk.Button(btn_frame, text="Zpět na volbu modelu", command=on_back, bg="#f44336", fg="white", font=("Arial", 10), padx=10, pady=5)
        btn_back.pack(side=tk.LEFT, padx=20)
        
        details_window.protocol("WM_DELETE_WINDOW", lambda: main_tk_root.destroy())


    # --- 4.1. Grafický rozcestník výběru modelu ---
    def open_model_selection_window():
        model_select_window = Toplevel(main_tk_root)
        model_select_window.title("Výběr modelu sítě")
        model_select_window.geometry("380x320")

        label = tk.Label(model_select_window, text="Zvolte soubor natrénovaného modelu:", font=("Arial", 10, "bold"))
        label.pack(pady=10)

        listbox = tk.Listbox(model_select_window, width=45, height=8)
        for i, file in enumerate(model_filenames_in_listbox):
            listbox.insert(tk.END, f"{file} - {i+1}")
        listbox.pack(pady=5)

        if last_used_model_filename and last_used_model_filename in model_filenames_in_listbox:
            idx = model_filenames_in_listbox.index(last_used_model_filename)
            listbox.selection_set(idx)
            listbox.activate(idx)

        def proceed_with_selection(use_last):
            if use_last:
                chosen = last_used_model_filename
            else:
                try:
                    selection_index = listbox.curselection()[0]
                    chosen = model_filenames_in_listbox[selection_index]
                except IndexError:
                    messagebox.showwarning("Upozornění", "Musíte nejdříve kliknutím vybrat model ze seznamu!")
                    return
            
            model_select_window.destroy()
            show_model_details_and_confirm(chosen)

        button = tk.Button(model_select_window, text="Zobrazit detaily vybraného modelu", command=lambda: proceed_with_selection(False), bg="#008CBA", fg="white")
        button.pack(pady=10)

        if last_used_model_filename and last_used_model_filename in model_filenames_in_listbox:
            default_button = tk.Button(
                model_select_window, text=f"Zobrazit předchozí: {last_used_model_filename}",
                command=lambda: proceed_with_selection(True), bg="#e7e7e7", fg="black"
            )
            default_button.pack(pady=5)
            
        model_select_window.protocol("WM_DELETE_WINDOW", lambda: main_tk_root.destroy())


    # Prvotní start logiky
    if last_used_model_filename and last_used_model_filename in model_filenames_in_listbox:
        show_model_details_and_confirm(last_used_model_filename)
    else:
        open_model_selection_window()

    main_tk_root.mainloop()

    # --- 4.2. Načtení zvoleného modelu do PyTorchu ---
    if not selected_model_container:
        print("Výběr modelu byl zrušen. Ukončuji skript.")
        main_tk_root.destroy()
        exit()

    model_name = selected_model_container[0]
    model_path = os.path.join(MODELS_DIR, model_name)

    model = PrumyslovaSit().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\nModel '{model_name}' byl úspěšně načten.")
        save_last_model_path(model_path)

        # --- 4.3. Grafický výběr obrázku k klasifikaci ---
        image_dir = './syrova_data'

        if not os.path.exists(image_dir):
            messagebox.showerror("Chyba", f"Adresář '{image_dir}' nebyl nalezen. Vytvořte jej a vložte do něj obrázky .png pro klasifikaci.")
            main_tk_root.destroy()
            exit()

        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])
        if not image_files:
            messagebox.showerror("Chyba", f"V adresáři '{image_dir}' nebyly nalezeny žádné obrázky .png!")
            main_tk_root.destroy()
            exit()

        # Otevíráme okno výběru obrázku jako Toplevel nad společným main_tk_root
        image_select_window = Toplevel(main_tk_root)
        image_select_window.title("Výběr obrázku k klasifikaci")
        image_select_window.geometry("400x300")

        image_label = tk.Label(image_select_window, text=f"Zvolte obrázek .png z adresáře '{image_dir}':", font=("Arial", 10, "bold"))
        image_label.pack(pady=10)

        image_listbox = tk.Listbox(image_select_window, width=50, height=10)
        for i, file_name in enumerate(image_files):
            image_listbox.insert(tk.END, f"{file_name} - {i+1}")
        image_listbox.pack(pady=5)

        def select_image():
            try:
                selection_index = image_listbox.curselection()[0]
                selected_file_name = image_files[selection_index]
                selected_image_container.append(os.path.join(image_dir, selected_file_name))
                image_select_window.destroy()
                main_tk_root.quit()  # Ukončí druhý mainloop
            except IndexError:
                messagebox.showwarning("Upozornění", "Musíte nejdříve kliknutím vybrat obrázek ze seznamu!")

        image_button = tk.Button(image_select_window, text="Klasifikovat vybraný obrázek", command=select_image, bg="#008CBA", fg="white")
        image_button.pack(pady=10)

        image_select_window.protocol("WM_DELETE_WINDOW", lambda: main_tk_root.destroy())
        main_tk_root.mainloop()

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

            # Okno náhledu jako Toplevel
            image_display_window = Toplevel(main_tk_root)
            image_display_window.title(f"Vybraný obrázek: {os.path.basename(final_image_path)}")

            img_tk = ImageTk.PhotoImage(img)

            panel = tk.Label(image_display_window, image=img_tk)
            panel.image = img_tk  
            panel.pack(padx=10, pady=10)

            def close_display():
                image_display_window.destroy()
                main_tk_root.quit()  # Ukončí třetí mainloop

            close_button = tk.Button(image_display_window, text="Zavřít náhled a pokračovat", command=close_display, bg="#f44336", fg="white", font=("Arial", 10))
            close_button.pack(pady=10)

            # Centrování okna na obrazovce pomocí tk::PlaceWindow
            # Použijeme update_idletasks, abychom zajistili, že okno má finální rozměry před centrováním
            image_display_window.update_idletasks()
            image_display_window.eval(f'tk::PlaceWindow {str(image_display_window)} center')

            image_display_window.protocol("WM_DELETE_WINDOW", close_display)
            main_tk_root.mainloop()

        except Exception as e:
            print(f"Chyba při zobrazení obrázku: {e}")

        # Úklid hlavního root okna z paměti před samotným výpisem klasifikace
        main_tk_root.destroy()

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
        print(f"Chyba při načítání nebo použití modelu: {e}")
        try:
            main_tk_root.destroy()
        except:
            pass