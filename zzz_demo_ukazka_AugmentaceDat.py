import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageTk
import os
import tkinter as tk
from tkinter import messagebox, Toplevel

# 1. Definujeme přesně stejnou augmentaci, jakou máš v train.py
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Funkce pro převod Tensoru zpět na zobrazitelný PIL Image
def tensor_to_pil(tensor):
    # Odnormalizování z [-1, 1] zpět na [0, 1]
    tensor = tensor * 0.5 + 0.5
    return transforms.ToPILImage()(tensor)

if __name__ == "__main__":
    # Inicializace skrytého hlavního okna Tkinter (stejně jako v inference.py)
    main_tk_root = tk.Tk()
    main_tk_root.withdraw()

    selected_image_container = []
    image_dir = './syrova_data'

    # Kontrola existence adresáře s obrázky
    if not os.path.exists(image_dir):
        messagebox.showerror("Chyba", f"Adresář '{image_dir}' nebyl nalezen. Vytvořte jej a vložte do něj obrázky .png.")
        main_tk_root.destroy()
        exit()

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])
    if not image_files:
        messagebox.showerror("Chyba", f"V adresáři '{image_dir}' nebyly nalezeny žádné obrázky .png!")
        main_tk_root.destroy()
        exit()

    # Grafické okno výběru obrázku z inference.py
    image_select_window = Toplevel(main_tk_root)
    image_select_window.title("Výběr obrázku pro ukázku augmentace")
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
            main_tk_root.quit()  # Ukončí mainloop a pustí kód dál k vykreslení
        except IndexError:
            messagebox.showwarning("Upozornění", "Musíte nejdříve kliknutím vybrat obrázek ze seznamu!")

    image_button = tk.Button(image_select_window, text="Zobrazit augmentaci pro tento obrázek", command=select_image, bg="#008CBA", fg="white")
    image_button.pack(pady=10)

    image_select_window.protocol("WM_DELETE_WINDOW", lambda: main_tk_root.destroy())
    main_tk_root.mainloop()

    # Pokud uživatel okno zavřel bez výběru
    if not selected_image_container:
        print("Výběr obrázku byl zrušen. Ukončuji skript.")
        try:
            main_tk_root.destroy()
        except:
            pass
        exit()

    final_image_path = selected_image_container[0]

    # Úklid hlavního root okna Tkinter z paměti, než se otevře Matplotlib graf
    main_tk_root.destroy()

    # 2. Načteme vybraný reálný obrázek
    try:
        # Převod na šedotónový ('L') jako u trénování
        original_image = Image.open(final_image_path).convert('L') 
    except Exception as e:
        print(f"Nepodařilo se načíst vybraný obrázek: {e}")
        exit()

    # 3. Vytvoříme mřížku pro 6 obrázků (1 původní + 5 náhodně augmentovaných)
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.ravel()

    # Zobrazení původního obrázku
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title(f"Původní: {os.path.basename(final_image_path)}", fontsize=9, fontweight='bold')
    axes[0].axis('off')

    # Použití augmentace v reálném čase pro dalších 5 pozic
    print(f"Generuji ukázky augmentace v paměti pro: {os.path.basename(final_image_path)}...")
    for i in range(1, 6):
        # Simulace toho, co zažívá DataLoader při trénování
        augmented_tensor = train_transform(original_image)
        augmented_image = tensor_to_pil(augmented_tensor)
        
        axes[i].imshow(augmented_image, cmap='gray')
        axes[i].set_title(f"Augmentace {i}", fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    print("Zobrazuji okno s výsledky...")
    plt.show()