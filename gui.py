# gui.py
import os
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
from PIL import Image, ImageTk

def run_init_gui(initial_model_path=""):
    """Vytvoří hlavní okno pro výběr checkpointu a režimu inference."""
    root = tk.Tk()
    root.title("Konfigurace inference - ResNet18")
    root.geometry("550x300")
    
    # Vycentrování okna
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    model_path_var = tk.StringVar(value=initial_model_path)
    classify_all_mode = [False]
    image_dir_container = [""]
    selected_image_container = [""]
    
    # UI Prvky
    tk.Label(root, text="Cesta k modelu (.pth):", font=("Arial", 10, "bold")).pack(pady=5)
    
    entry_frame = tk.Frame(root)
    entry_frame.pack(fill="x", padx=20)
    
    entry = tk.Entry(entry_frame, textvariable=model_path_var, font=("Arial", 9))
    entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
    
    def browse_model():
        path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if path:
            model_path_var.set(path)
            
    tk.Button(entry_frame, text="Procházet...", command=browse_model).pack(side="right")
    
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=30)
    
    def on_single():
        if not os.path.exists(model_path_var.get()):
            messagebox.showerror("Chyba", "Vybraný model neexistuje!")
            return
        path = filedialog.askopenfilename(filetypes=[("Obrázky", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            selected_image_container[0] = path
            root.quit()
            
    def on_all():
        if not os.path.exists(model_path_var.get()):
            messagebox.showerror("Chyba", "Vybraný model neexistuje!")
            return
        path = filedialog.askdirectory(title="Vyber složku s obrázky pro hromadnou klasifikaci")
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