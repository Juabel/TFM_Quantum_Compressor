import glob
import random
import os
from torch.utils.data import Dataset
import torch
import numpy as np
import funciones_estado


def cargar_por_clases(path, n_per_class, seed=42):
    random.seed(seed)
    
    files = []
    classes = sorted(os.listdir(path))
    
    for c in classes:
        folder = os.path.join(path, c)
        images = glob.glob(os.path.join(folder, "*.png"))
        
        random.shuffle(images)
        selected = images[:n_per_class]
        
        files.extend(selected)
    
    # print(f"[Dataset] {path} -> {len(files)} imágenes")
    return files

def split_dataset(path, n_train, n_test, seed=42):
    random.seed(seed)
    
    train_files = []
    test_files = []
    
    classes = sorted(os.listdir(path))
    
    for c in classes:
        folder = os.path.join(path, c)
        images = glob.glob(os.path.join(folder, "*.png"))
        
        random.shuffle(images)
        
        train_files += images[:n_train]
        test_files += images[n_train:n_train+n_test]
    
    # print(f"[SAR] Train: {len(train_files)} | Test: {len(test_files)}")
    return train_files, test_files

class QuantumImageDataset(Dataset):

    def __init__(self, file_list, resize_dim):

        self.files = file_list
        self.resize_dim = resize_dim

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]

        img = funciones_estado.open_image_safe(path)

        img = img.resize(self.resize_dim)

        img_array = np.array(img, dtype=np.float32)

        # Normalización opcional
        img_array = img_array / 255.0

        img_tensor = torch.tensor(img_array)

        # Añadir canal:
        # [28,28] -> [1,28,28]

        if len(img_tensor.shape) == 2:
            img_tensor = img_tensor.unsqueeze(0)

        return img_tensor, self.files[idx]