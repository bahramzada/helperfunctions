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

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def calculateMeanStd(data_dir, batch_size=32, num_workers=2, image_size=(224, 224)):
    """
    Diskdə yerləşən şəkil datasetinin (ImageFolder strukturu) orta (mean) və standart sapma (std) dəyərlərini hesablayır.
    
    Args:
        data_dir (str): Datasetin yerləşdiyi qovluğun yolu. Qovluq alt-kataloqlar şəklində olmalıdır (hər biri bir class).
        batch_size (int): DataLoader üçün batch ölçüsü.
        num_workers (int): DataLoader üçün paralel işləyəcək işçi sayı.
        image_size (tuple): Şəkillərin ölçüsünü dəyişmək üçün istifadə olunan ölçü (hündürlük, en).

    Returns:
        mean (torch.Tensor): RGB kanalları üçün orta dəyərlər (shape: [3])
        std (torch.Tensor): RGB kanalları üçün standart sapmalar (shape: [3])
    """

    # Şəkilləri əvvəlcədən müəyyən ölçüyə gətirib, tensor formatına çeviririk
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    # Qovluqdakı şəkilləri və label-ları yükləyirik
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    n_channels = 3  # RGB şəkillər üçün 3 kanal
    mean = torch.zeros(n_channels)
    std = torch.zeros(n_channels)
    total_images = 0

    # Bütün batch-lər üzrə dövr (loop)
    for images, _ in loader:
        batch_samples = images.size(0)  # Hər batch-dəki şəkil sayı
        # [batch, channel, height, width] -> [batch, channel, height*width]
        images = images.view(batch_samples, images.size(1), -1)
        # Hər kanal üzrə orta və std hesabla və batch üzrə cəmlə
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    # Ümumi şəkil sayına bölməklə son nəticəni əldə et
    mean /= total_images
    std /= total_images

    return mean, std

import numpy as np
from torchvision import transforms
from tqdm import tqdm

def calculateMeanStdHF(hf_dataset):
    """
    Hugging Face dataset-indəki bütün şəkilləri RGB-yə çevirərək
    RGB kanallar üzrə orta (mean) və standart sapma (std) hesablayır.

    Args:
        hf_dataset: Hugging Face dataset obyektidir (məs: ds["train"])

    Returns:
        mean: np.ndarray, RGB üçün orta dəyərlər (shape: [3])
        std: np.ndarray, RGB üçün std dəyərlər (shape: [3])
    """
    to_tensor = transforms.ToTensor()
    n_images = len(hf_dataset)
    mean = np.zeros(3)
    std = np.zeros(3)
    rgb_count = 0

    for i in tqdm(range(n_images)):
        img = hf_dataset[i]['image']
        img = img.convert("RGB")  # Bütün şəkilləri RGB-yə çeviririk
        img = to_tensor(img)      # [C, H, W] formatına salırıq
        mean += img.mean(dim=(1, 2)).numpy()
        std += img.std(dim=(1, 2)).numpy()
        rgb_count += 1

    mean /= rgb_count
    std /= rgb_count

    return mean, std

# İstifadə nümunəsi:
# mean, std = calculate_mean_std_hfdataset(ds["train"])
# print("Mean:", mean)
# print("Std:", std)
