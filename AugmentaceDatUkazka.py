import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

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
    # Převod na PIL
    return transforms.ToPILImage()(tensor)

if __name__ == "__main__":
    # 2. Načteme tvoje trénovací data (stačí první obrázek)
    try:
        data = np.load('./data/vlist_train_images.npy')
        # Vezmeme úplně první obrázek (index 0)
        raw_image_array = data[0]
        # Převedeme na PIL Image (stejně jako to dělá tvůj VLISTDataset)
        original_image = Image.fromarray(raw_image_array, mode='L')
    except Exception as e:
        print(f"Nepodařilo se načíst data: {e}")
        exit()

    # 3. Vytvoříme mřížku pro 6 obrázků (1 původní + 5 náhodně augmentovaných)
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.ravel()

    # Zobrazení původního obrázku
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title("Původní obrázek", fontsize=10, fontweight='bold')
    axes[0].axis('off')

    # Použití augmentace v reálném čase pro dalších 5 pozic
    print("Generuji ukázky augmentace v paměti...")
    for i in range(1, 6):
        # Zde probíhá transformace přesně tak, jak ji provádí DataLoader za běhu
        augmented_tensor = train_transform(original_image)
        augmented_image = tensor_to_pil(augmented_tensor)
        
        axes[i].imshow(augmented_image, cmap='gray')
        axes[i].set_title(f"Augmentace {i}", fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    print("Zobrazuji okno s výsledky...")
    plt.show()