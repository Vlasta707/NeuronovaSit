import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk # Přidáno ImageTk pro zobrazení obrázků v Tkinteru
import os # Pro kontrolu existence souborů
import tkinter as tk
from tkinter import messagebox, Toplevel # Toplevel pro nové okno

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

    # Vytvoříme jedno hlavní Tkinter root okno, které ale bude skryté.
    # Toto okno poskytne potřebný kontext pro PhotoImage a Toplevel okna.
    main_tk_root = tk.Tk()
    main_tk_root.withdraw() # Skryje hlavní okno

    selected_model_container = []
    selected_image_container = []

    # --- 4.1. Grafický výběr modelu pomocí Tkinter (jako Toplevel okno) ---
    model_select_window = Toplevel(main_tk_root) # Toplevel je dítětem skrytého main_tk_root
    model_select_window.title("Výběr modelu sítě")
    model_select_window.geometry("350x250")

    label = tk.Label(model_select_window, text="Zvolte soubor natrénovaného modelu:", font=("Arial", 10, "bold"))
    label.pack(pady=10)

    listbox = tk.Listbox(model_select_window, width=40, height=8)

    models_found = False
    for i, file in enumerate(os.listdir()):
        if file.endswith('.pth') or file.endswith('.pt'):
            listbox.insert(tk.END, f"{file} - {i+1}")
            models_found = True
    listbox.pack(pady=5)

    if not models_found:
        messagebox.showerror("Chyba", "V aktuálním adresáři nebyly nalezeny žádné modely (.pth/.pt)!")
        model_select_window.destroy()
        main_tk_root.destroy() # Zajistíme, že se vše uklidí
        exit()

    def select_model():
        try:
            selection_index = listbox.curselection()[0]
            selected_file_name = listbox.get(selection_index).split(" - ")[0]
            selected_model_container.append(selected_file_name)
            model_select_window.destroy() # Zavře toto Toplevel okno
        except IndexError:
            messagebox.showwarning("Upozornění", "Musíte nejdříve kliknutím vybrat model ze seznamu!")

    button = tk.Button(model_select_window, text="Načíst vybraný model", command=select_model, bg="#4CAF50", fg="white")
    button.pack(pady=10)

    # Čekáme, dokud se toto Toplevel okno nezavře
    main_tk_root.wait_window(model_select_window)

    # --- 4.2. Načtení zvoleného modelu do PyTorchu ---
    if not selected_model_container:
        print("Výběr modelu byl zrušen. Ukončuji skript.")
        main_tk_root.destroy()
        exit()

    model_name = selected_model_container[0]
    model_path = f'./{model_name}'

    model = PrumyslovaSit().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\nModel '{model_name}' byl úspěšně načten.")

        # --- 4.3. Grafický výběr obrázku k klasifikaci (jako Toplevel okno) ---
        image_dir = './syrova_data'

        if not os.path.exists(image_dir):
            messagebox.showerror("Chyba", f"Adresář '{image_dir}' nebyl nalezen. Vytvořte jej a vložte do něj obrázky .png pro klasifikaci.")
            main_tk_root.destroy()
            exit()

        image_select_window = Toplevel(main_tk_root) # Toplevel je dítětem main_tk_root
        image_select_window.title("Výběr obrázku k klasifikaci")
        image_select_window.geometry("400x300")

        image_label = tk.Label(image_select_window, text=f"Zvolte obrázek .png z adresáře '{image_dir}':", font=("Arial", 10, "bold"))
        image_label.pack(pady=10)

        image_listbox = tk.Listbox(image_select_window, width=50, height=10)

        image_files_found = False
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]

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
                image_select_window.destroy() # Zavře toto Toplevel okno
            except IndexError:
                messagebox.showwarning("Upozornění", "Musíte nejdříve kliknutím vybrat obrázek ze seznamu!")

        image_button = tk.Button(image_select_window, text="Klasifikovat vybraný obrázek", command=select_image, bg="#008CBA", fg="white")
        image_button.pack(pady=10)

        # Čekáme, dokud se toto Toplevel okno nezavře
        main_tk_root.wait_window(image_select_window)

        # Po zavření okna pro výběr obrázku
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

            img_tk = ImageTk.PhotoImage(img) # main_tk_root je stále aktivní pro kontext

            image_display_window = Toplevel(main_tk_root) # Parent je main_tk_root
            image_display_window.title(f"Vybraný obrázek: {os.path.basename(final_image_path)}")

            image_display_window.protocol("WM_DELETE_WINDOW", image_display_window.destroy)

            panel = tk.Label(image_display_window, image=img_tk)
            panel.image = img_tk
            panel.pack(padx=10, pady=10)

            close_button = tk.Button(image_display_window, text="Zavřít náhled a pokračovat", command=image_display_window.destroy)
            close_button.pack(pady=5)

            image_display_window.update_idletasks()
            # Centrování okna na obrazovce (protože main_tk_root je skrytý)
            screen_width = main_tk_root.winfo_screenwidth()
            screen_height = main_tk_root.winfo_screenheight()
            x = (screen_width // 2) - (image_display_window.winfo_width() // 2)
            y = (screen_height // 2) - (image_display_window.winfo_height() // 2)
            image_display_window.geometry(f"+{x}+{y}")

            main_tk_root.wait_window(image_display_window) # Čeká na zavření tohoto Toplevel okna

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
        print(f"Chyba při načítání nebo použití modelu: {e}")
    finally:
        # Zajistíme, že se hlavní Tkinter root okno zničí při ukončení skriptu
        if main_tk_root.winfo_exists():
            main_tk_root.destroy()

