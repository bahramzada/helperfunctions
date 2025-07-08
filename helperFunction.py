import zipfile
import os

def unzipFile(file_path, target_folder):
    # TargetFolder yoxdursa yarat
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)
    return "Unzip Completed"


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def calculateMeanStd(data_dir, batch_size=32, num_workers=2, image_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(image_size), # Add resizing here
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    mean = 0.
    std = 0.
    total_images_count = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  # (B, C, H*W)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean , std
