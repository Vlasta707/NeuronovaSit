import os
import torch
from torchvision import transforms

# Importy našich vlastních modulů
from parser_utils import load_last_model_path, save_last_model_path, parse_md_file
from gui import run_init_gui, show_image_preview
from model_utils import get_resnet18_model, classify_image

def main():
    class_names = ['BAD', 'OK']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Načtení historie a spuštění úvodního GUI
    last_model = load_last_model_path()
    root, model_path, bulk_mode, image_dir, single_image_path = run_init_gui(last_model)
    
    # Pokud uživatel zavřel okno křížkem bez akce
    if not model_path or (not bulk_mode and not single_image_path):
        print("[INFO] Skript ukončen uživatelem.")
        try: root.destroy()
        except: pass
        return

    # Uložení cesty k modelu pro příště
    save_last_model_path(model_path)
    
    # 2. Parsování normalizačních hodnot z MD logu sítě
    md_path = model_path.replace('.pth', '.md')
    parsed_mean, parsed_std = parse_md_file(md_path)
    
    # Ošetření: Pokud parser vrátil pole/seznam pro 3 kanály, vezmeme jen první hodnotu pro černobílý režim
    if isinstance(parsed_mean, (list, tuple)):
        mean_val = parsed_mean[0]
    else:
        mean_val = parsed_mean if parsed_mean is not None else 0.5
        
    if isinstance(parsed_std, (list, tuple)):
        std_val = parsed_std[0]
    else:
        std_val = parsed_std if parsed_std is not None else 0.5

    # Vytvoření 1-kanálové transformace přímo zde (Rozměr 256x256 přesně pro PrumyslovaSit)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((mean_val,), (std_val,))
    ])
    
    print(f"[INFO] Běží na zařízení: {device.type.upper()}")
    print(f"[INFO] Načítám model: {os.path.basename(model_path)}")
    print(f"[INFO] Použitá normalizace (1 kanál) -> Mean: {mean_val}, Std: {std_val}")
    
    # 3. Načtení PyTorch modelu
    try:
        model = get_resnet18_model(num_classes=2)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Přesné ošetření tvé struktury checkpointu
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model = checkpoint
            
        model = model.to(device)
    except Exception as e:
        print(f"[KRITICKÁ CHYBA] Nelze načíst checkpoint modelu: {e}")
        root.destroy()
        return

    # 4. Spuštění samotné inference
    if bulk_mode:
        # --- REŽIM: HROMADNÁ KLASIFIKACE ---
        root.destroy()
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]
        
        if not image_files:
            print(f"[VAROVÁNÍ] Ve složce '{image_dir}' nebyly nalezeny žádné podporované obrázky.")
            return
            
        print(f"\n--- Spouštím hromadnou klasifikaci adresáře '{image_dir}' ---")
        print(f"{'Predikce':<8} | {'Jistota':<10} | {'Název souboru'}")
        print("-" * 50)
        
        for file_name in image_files:
            full_path = os.path.join(image_dir, file_name)
            res = classify_image(model, full_path, device, transform, class_names)
            
            if res:
                pred, conf = res
                print(f"{pred:<8} | {conf:>8.2f}% | {file_name}")
            else:
                print(f"{'CHYBA':<8} | {'-':>10} | {file_name}")
                
        print(f"\n[HOTOVO] Analýza složky '{image_dir}' byla dokončena.")
        
    else:
        # --- REŽIM: JEDEN OBRÁZEK ---
        show_image_preview(root, single_image_path)
        root.destroy()
        
        res = classify_image(model, single_image_path, device, transform, class_names)
        if res:
            pred, conf = res
            print(f"\nVýsledek analýzy pro: {os.path.basename(single_image_path)}")
            print(f"   Verdikt:  {pred}")
            print(f"   Jistota:  {conf:.2f}%")

if __name__ == "__main__":
    main()