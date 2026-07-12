# gui.py
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
from PIL import Image, ImageTk

# Importy pro vykreslení grafu
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def parse_training_logs(md_path):
    """Vyparsuje z .md logu hodnoty pro vykreslení grafu."""
    epochs, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []
    if not os.path.exists(md_path):
        return None
        
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Hledáme řádky typu: Epoch 1/20 -> Train Loss: 0.25..., Val Loss: 0.31...
                if "Epoch" in line and "Train Loss" in line:
                    ep_match = re.search(r'Epoch\s+(\d+)', line)
                    tl_match = re.search(r'Train Loss:\s*([\d.]+)', line)
                    vl_match = re.search(r'Val Loss:\s*([\d.]+)', line)
                    ta_match = re.search(r'Train Acc:\s*([\d.]+)', line)
                    va_match = re.search(r'Val Acc:\s*([\d.]+)', line)
                    
                    if ep_match and tl_match and vl_match:
                        epochs.append(int(ep_match.group(1)))
                        train_loss.append(float(tl_match.group(1)))
                        val_loss.append(float(vl_match.group(1)))
                        if ta_match: train_acc.append(float(ta_match.group(1)))
                        if va_match: val_acc.append(float(va_match.group(1)))
        return epochs, train_loss, val_loss, train_acc, val_acc
    except Exception as e:
        print(f"[VAROVÁNÍ] Nepodařilo se vyparsovat grafy z logu: {e}")
        return None

def run_init_gui(initial_model_path=""):
    """Vytvoří hlavní okno pro výběr checkpointu, vykreslí graf a zvolí režim."""
    root = tk.Tk()
    root.title("Konfigurace inference - ResNet18")
    root.geometry("600x650") # Zvětšeno, aby se vešel graf
    
    # Vycentrování okna
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    model_path_var = tk.StringVar(value=initial_model_path)
    classify_all_mode = [False]
    image_dir_container = [""]
    selected_image_container = [""]
    
    # Kontejner pro graf, abychom ho mohli překreslovat
    graph_frame_container = [None]
    
    tk.Label(root, text="Cesta k modelu (.pth):", font=("Arial", 10, "bold")).pack(pady=5)
    
    entry_frame = tk.Frame(root)
    entry_frame.pack(fill="x", padx=20)
    
    entry = tk.Entry(entry_frame, textvariable=model_path_var, font=("Arial", 9))
    entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
    
    def update_graph():
        """Překreslí graf podle aktuálně zvoleného modelu."""
        if graph_frame_container[0]:
            graph_frame_container[0].destroy()
            
        current_pth = model_path_var.get()
        current_md = current_pth.replace('.pth', '.md')
        
        data = parse_training_logs(current_md)
        if not data or not data[0]:
            # Pokud log neexistuje, vytvoříme prázdný placeholder frame
            graph_frame_container[0] = tk.Frame(root, height=300)
            graph_frame_container[0].pack(fill="both", expand=True, padx=20, pady=10)
            tk.Label(graph_frame_container[0], text="(Pro tento model nebyl nalezen tréninkový .md log s grafem)", fg="gray").pack(pady=100)
            return
            
        epochs, t_loss, v_loss, t_acc, v_acc = data
        
        # Vytvoření Matplotlib grafu
        graph_frame_container[0] = tk.Frame(root)
        graph_frame_container[0].pack(fill="both", expand=True, padx=20, pady=10)
        
        fig = Figure(figsize=(5, 3.5), dpi=100)
        
        # Graf pro Loss
        ax1 = fig.add_subplot(211)
        ax1.plot(epochs, t_loss, 'r-', label='Train Loss')
        ax1.plot(epochs, v_loss, 'b-', label='Val Loss')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper right', fontsize='small')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_title(f"Historie učení: {os.path.basename(current_pth)}", fontsize=10)
        
        # Graf pro Accuracy (pokud existuje)
        ax2 = fig.add_subplot(212)
        if t_acc and v_acc:
            ax2.plot(epochs, t_acc, 'r--', label='Train Acc')
            ax2.plot(epochs, v_acc, 'b--', label='Val Acc')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_xlabel('Epocha')
            ax2.legend(loc='lower right', fontsize='small')
            ax2.grid(True, linestyle='--', alpha=0.6)
            
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=graph_frame_container[0])
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def browse_model():
        initial_dir = os.path.dirname(model_path_var.get()) if model_path_var.get() else os.getcwd()
        path = filedialog.askopenfilename(initialdir=initial_dir, filetypes=[("PyTorch Model", "*.pth")])
        if path:
            model_path_var.set(path)
            update_graph() # Aktualizujeme graf po výběru nového modelu
            
    tk.Button(entry_frame, text="Procházet...", command=browse_model).pack(side="right")
    
    # První inicializace grafu při startu okna
    update_graph()
    
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=20)
    
    default_img_dir = "syrova_data"
    img_initial_dir = default_img_dir if os.path.exists(default_img_dir) else os.getcwd()
    
    def on_single():
        if not os.path.exists(model_path_var.get()):
            messagebox.showerror("Chyba", "Vybraný model neexistuje!")
            return
        path = filedialog.askopenfilename(
            initialdir=img_initial_dir, 
            filetypes=[
                ("Všechny podporované obrázky", "*.png *.jpg *.jpeg *.bmp"),
                ("PNG obrázky", "*.png"),
                ("JPEG obrázky", "*.jpg *.jpeg"),
                ("Všechny soubory", "*.*")
            ]
        )
        if path:
            selected_image_container[0] = path
            root.quit()
            
    def on_all():
        if not os.path.exists(model_path_var.get()):
            messagebox.showerror("Chyba", "Vybraný model neexistuje!")
            return
        path = filedialog.askdirectory(title="Vyber složku s obrázky pro hromadnou klasifikaci", initialdir=img_initial_dir)
        if path:
            image_dir_container[0] = path
            classify_all_mode[0] = True
            root.quit()
            
    tk.Button(btn_frame, text="Klasifikovat jeden obrázek", command=on_single, bg="#4CAF50", fg="white", font=("Arial", 11, "bold"), padx=10, pady=5).pack(side="left", padx=10)
    tk.Button(btn_frame, text="Klasifikovat celou složku", command=on_all, bg="#008CBA", fg="white", font=("Arial", 11, "bold"), padx=10, pady=5).pack(side="right", padx=10)
    
    root.mainloop()
    
    return root, model_path_var.get(), classify_all_mode[0], image_dir_container[0], selected_image_container[0]

def show_image_preview(main_root, image_path):
    """Zobrazí okno s náhledem vybraného obrázku před inferencí."""
    try:
        img = Image.open(image_path)
        img.thumbnail((600, 600), Image.Resampling.LANCZOS)
        
        preview_win = Toplevel(main_root)
        preview_win.title(f"Náhled: {os.path.basename(image_path)}")
        
        img_tk = ImageTk.PhotoImage(img)
        panel = tk.Label(preview_win, image=img_tk)
        panel.image = img_tk
        panel.pack(padx=10, pady=10)
        
        def close_preview():
            preview_win.destroy()
            main_root.quit()
            
        tk.Button(preview_win, text="Zavřít náhled a spustit klasifikaci", command=close_preview, bg="#f44336", fg="white", font=("Arial", 10)).pack(pady=10)
        
        preview_win.update_idletasks()
        x = (main_root.winfo_screenwidth() // 2) - (preview_win.winfo_width() // 2)
        y = (main_root.winfo_screenheight() // 2) - (preview_win.winfo_height() // 2)
        preview_win.geometry(f"+{x}+{y}")
        
        preview_win.protocol("WM_DELETE_WINDOW", close_preview)
        main_root.mainloop()
    except Exception as e:
        print(f"[VAROVÁNÍ] Selhalo zobrazení náhledu: {e}")