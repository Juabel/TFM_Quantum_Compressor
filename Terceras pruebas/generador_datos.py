import glob
import random
import os

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