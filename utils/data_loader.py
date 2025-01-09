from torchvision.transforms import transforms
from PIL import Image
import os
import yaml
import torch

def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, labels_dir, transform=None):
        """
        root_dir: Directorio con las imágenes.
        labels_dir: Directorio con los archivos .txt de los targets.
        transform: Transformaciones para las imágenes.
        """
        self.root_dir = root_dir
        self.labels_dir = labels_dir
        self.transform = transform
        
        # Obtener nombres de archivos de imágenes y targets
        self.images = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith(('.jpg', '.png','.jpeg'))]
        self.labels = [os.path.join(labels_dir, os.path.splitext(os.path.basename(file))[0] + '.txt') for file in os.listdir(root_dir) if file.endswith(('.jpg', '.png','.jpeg'))]
        # Verificar que haya correspondencia entre imágenes y targets
        assert len(self.images) == len(self.labels), "El número de imágenes y archivos .txt no coincide."
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]
        
        # Cargar la imagen
        image = Image.open(img_path).convert('RGB')
        
        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)
        
        # Leer el archivo .txt para el target
        try:
            with open(label_path, 'r') as f:
                target_data = f.read().strip()
            # Parsear los targets
            target = torch.tensor([float(value) for value in target_data.split()], dtype=torch.float32)
        except FileNotFoundError:
            # Si no existe el archivo, asumir máscara vacía
            target = torch.tensor([], dtype=torch.float32)  # Sin segmentación
        
        #print(f"Image: {img_path}, Target: {target}")
        return image, target


def create_custom_loaders(config_path, batch_size=256, workers=1, pin_memory=True, do_normalize=True):
    # Leer configuración YAML
    config = load_yaml_config(config_path)
    train_path = os.path.join(config['path'], config['train'])
    train_labels_path = os.path.join(config['path'], config['train_labels'])  # Agrega esta línea
    val_path = os.path.join(config['path'], config['val'])
    val_labels_path = os.path.join(config['path'], config['val_labels'])      # Agrega esta línea
    
    # Transformaciones
    transformations = [
        transforms.Resize(640),
        transforms.CenterCrop(640),
        transforms.ToTensor()
    ]
    
    if do_normalize:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transformations.append(normalize)
    
    transform = transforms.Compose(transformations)
    
    # Crear datasets personalizados
    #train_dataset = CustomDataset(train_path, train_labels_path, transform)  # Pasa train_labels_path
    val_dataset = CustomDataset(val_path, val_labels_path, transform)        # Pasa val_labels_path
    
    # DataLoaders
    train_loader = []
    """ 
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory
    )
    """
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

def collate_fn(batch):
    images, targets = zip(*batch)
    
    # Apila las imágenes
    images = torch.stack(images, 0)
    
    # Encuentra el tamaño máximo entre todas las etiquetas
    max_len = max(target.size(0) for target in targets)
    
    # Rellena cada etiqueta con ceros para que todas tengan el mismo tamaño
    padded_targets = torch.zeros((len(targets), max_len), dtype=torch.float32)
    for i, target in enumerate(targets):
        padded_targets[i, :target.size(0)] = target
    
    return images, padded_targets
