import Mydatasetplayersjede
from Mydatasetplayersjede import Mydataset
import torch
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import cv2

import matplotlib.pyplot as plt
import numpy as np


def display_masks_from_dataset(dataset, index):
    """
    Zeigt die Masken eines Bildes aus dem Dataset an.
    Args:
        dataset: Dataset-Objekt.
        index: Index des Bildes im Dataset.
    """
    # Hole das Bild und die Ziel-Daten (inkl. Masken)
    img, target = dataset[index]

    if img is None or target is None:
        print(f"Kein gültiges Bild oder Ziel für Index {index}.")
        return

    # Bild als NumPy Array konvertieren
    img_np = img.numpy().transpose(1, 2, 0)  # Konvertiere von (C, H, W) nach (H, W, C)

    # Masken extrahieren
    masks = target['masks'].numpy()  # Shape: [N, H, W], wobei N die Anzahl der Objekte ist

    # Erstelle eine figure für die Visualisierung
    fig, axes = plt.subplots(1, masks.shape[0] + 1, figsize=(15, 5))

    # Originalbild anzeigen
    axes[0].imshow(img_np.astype(np.uint8))
    axes[0].set_title("Originalbild")
    axes[0].axis("off")
    bool_portray = [True, True, True, True]
    # Jede Maske anzeigen
    for i, mask in enumerate(masks):
        if len(masks[i]) != 0:
            bool_portray[i] = True
        else:
            bool_portray[i] = False
        axes[i + 1].imshow(mask, cmap="gray")
        axes[i + 1].set_title(f"Maske {i + 1}, {i}, {bool_portray[i]}")
        axes[i + 1].axis("off")

    # Zeige die Figuren
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    path = osp.normpath(osp.join(osp.dirname(__file__), "data"))
    dataset = Mydataset(path)
    idx = 17
    print(dataset.imgs[idx])
    print(dataset[idx])
    # Malen
    dataset = Mydataset(None)  # Dataset initialisieren
    display_masks_from_dataset(dataset, 0)  # Index 0 anzeigen
