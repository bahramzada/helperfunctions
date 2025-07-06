import zipfile
import os

def unzipFile(file_path, target_folder):
    # Hedef klasör yoksa oluştur
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)
    return "Unzip Completed"

