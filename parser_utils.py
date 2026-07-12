# parser_utils.py
import os
import re
import torchvision.transforms as transforms

def parse_md_file(file_path):
    """Vyparsuje z markdown logu normalizační parametry (Mean a Std)."""
    mean = [0.485, 0.456, 0.406]  # default ImageNet values
    std = [0.229, 0.224, 0.225]
    
    if not os.path.exists(file_path):
        return mean, std
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        mean_match = re.search(r'Mean:\s*\[?([\d.,\s]+)\]?', content)
        std_match = re.search(r'Std:\s*\[?([\d.,\s]+)\]?', content)
        
        if mean_match:
            mean = [float(x.strip()) for x in mean_match.group(1).split(',')]
        if std_match:
            std = [float(x.strip()) for x in std_match.group(1).split(',')]
    except Exception as e:
        print(f"[VAROVÁNÍ] Chyba při čtení MD souboru, použity defaultní transformace: {e}")
        
    return mean, std

def load_last_model_path(config_file="last_model.txt"):
    """Načte cestu k naposledy použitému checkpointu. Pokud neexistuje, vrátí default."""
    DEFAULT_MODEL_PATH = "checkpoints/best_model.pth"
    
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                path = f.read().strip()
                if path and os.path.exists(path):
                    return path
        except Exception as e:
            print(f"[VAROVÁNÍ] Nepodařilo se načíst konfiguraci: {e}")
            
    # Pokud historie neexistuje, zkusíme vrátit defaultní model, pokud fyzicky existuje
    if os.path.exists(DEFAULT_MODEL_PATH):
        return DEFAULT_MODEL_PATH
        
    return ""

def save_last_model_path(path, config_file="last_model.txt"):
    """Uloží cestu k aktuálně vybranému checkpointu."""
    try:
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(path)
    except Exception as e:
        print(f"[VAROVÁNÍ] Nepodařilo se uložit konfiguraci: {e}")

def get_inference_transforms(mean, std):
    """Vrátí transformační pipeline pro obrázky."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])