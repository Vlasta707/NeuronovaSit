import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os # Pro kontrolu existence souborů
import tkinter as tk
from tkinter import messagebox # Pro zobrazení případného varování

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

    # --- 4.1. Grafický výběr modelu pomocí Tkinter ---
    selected_model_container = [] # Pomocný list pro uložení názvu souboru z GUI

    root = tk.Tk()
    root.title("Výběr modelu sítě")
    root.geometry("350x250") # Nastavení rozumné velikosti okna

    label = tk.Label(root, text="Zvolte soubor natrénovaného modelu:", font=("Arial", 10, "bold"))
    label.pack(pady=10)

    listbox = tk.Listbox(root, width=40, height=8)
    
    # Načtení souborů z adresáře do seznamu
    models_found = False
    for i, file in enumerate(os.listdir()):
        if file.endswith('.pth') or file.endswith('.pt'):
            listbox.insert(tk.END, f"{file} - {i+1}")
            models_found = True
    listbox.pack(pady=5)

    if not models_found:
        messagebox.showerror("Chyba", "V aktuálním adresáři nebyly nalezeny žádné modely (.pth/.pt)!")
        root.destroy()
        exit()

    def select_model():
        try:
            selection_index = listbox.curselection()[0]
            selected_file_name = listbox.get(selection_index).split(" - ")[0]
            selected_model_container.append(selected_file_name) # Uložíme název ven z funkce
            root.destroy() # Zavřeme okno a pokračujeme v kódu
        except IndexError:
            messagebox.showwarning("Upozornění", "Musíte nejdříve kliknutím vybrat model ze seznamu!")

    button = tk.Button(root, text="Načíst vybraný model", command=select_model, bg="#4CAF50", fg="white")
    button.pack(pady=10)

    root.mainloop()

    # --- 4.2. Načtení zvoleného modelu do PyTorchu ---
    if not selected_model_container:
        print("Výběr modelu byl zrušen. Ukončuji skript.")
        exit()

    model_name = selected_model_container[0]
    model_path = f'./{model_name}'

    model = PrumyslovaSit().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\nModel '{model_name}' byl úspěšně načten.")

        # --- 4.3. Grafický výběr obrázku k klasifikaci ---
        image_dir = './SyrovaData'
        selected_image_container = []

        if not os.path.exists(image_dir):
            messagebox.showerror("Chyba", f"Adresář '{image_dir}' nebyl nalezen. Vytvořte jej a vložte do něj obrázky .png pro klasifikaci.")
            exit()

        image_root = tk.Tk()
        image_root.title("Výběr obrázku k klasifikaci")
        image_root.geometry("400x300")

        image_label = tk.Label(image_root, text=f"Zvolte obrázek .png z adresáře '{image_dir}':", font=("Arial", 10, "bold"))
        image_label.pack(pady=10)

        image_listbox = tk.Listbox(image_root, width=50, height=10)

        image_files_found = False
        # Filtrujeme pouze .png soubory z určeného adresáře
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]

        if image_files:
            for i, file_name in enumerate(image_files):
                image_listbox.insert(tk.END, f"{file_name} - {i+1}")
            image_files_found = True
        image_listbox.pack(pady=5)

        if not image_files_found:
            messagebox.showerror("Chyba", f"V adresáři '{image_dir}' nebyly nalezeny žádné obrázky .png!")
            image_root.destroy()
            exit()

        def select_image():
            try:
                selection_index = image_listbox.curselection()[0]
                selected_file_name = image_listbox.get(selection_index).split(" - ")[0]
                # Uložíme celou cestu k obrázku
                selected_image_container.append(os.path.join(image_dir, selected_file_name))
                image_root.destroy()
            except IndexError:
                messagebox.showwarning("Upozornění", "Musíte nejdříve kliknutím vybrat obrázek ze seznamu!")

        image_button = tk.Button(image_root, text="Klasifikovat vybraný obrázek", command=select_image, bg="#008CBA", fg="white")
        image_button.pack(pady=10)

        image_root.mainloop()

        # Po zavření okna pro výběr obrázku
        if not selected_image_container:
            print("Výběr obrázku byl zrušen. Ukončuji skript.")
            exit()

        final_image_path = selected_image_container[0] # Získáme finální cestu k obrázku

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