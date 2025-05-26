import os
from PIL import Image
from tqdm import tqdm
import csv
import random

#adjust these paths as necessary
base_c2a_dir = 'bdt\\C2A_Dataset\\new_dataset3'
no_disaster_dir = 'bdt\\no_disaster_images'
output_dir = 'bdt\\dataset'
target_size = (224, 224)

splits = ['train', 'val', 'test']

#create output directories
classes = ['fire', 'flood', 'collapsed_building', 'traffic_accident', 'no_disaster']
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

def process_disaster_split(split_name):
    split_image_dir = os.path.join(base_c2a_dir, split_name, 'images')
    label_entries = []
    print(f"Processing {split_name} disaster images..")
    for filename in tqdm(os.listdir(split_image_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            prefix = filename.split('_image')[0].lower()
            if prefix == 'fire':
                label = 'fire'
            elif prefix == 'flood':
                label = 'flood'
            elif prefix == 'collapsed_building':
                label = 'collapsed_building'
            elif prefix == 'traffic_incident':
                label = 'traffic_accident'
            else:
                print(f"Skipping unrecognized file: {filename}")
                continue

            src_path = os.path.join(split_image_dir, filename)
            dst_path = os.path.join(output_dir, split_name, label, filename)
            try:
                img = Image.open(src_path).convert('RGB')
                img_resized = img.resize(target_size)
                img_resized.save(dst_path)
                label_entries.append({'filename': dst_path, 'label': label})
            except Exception as e:
                print(f"Failed processing {filename}: {e}")
    return label_entries

#process each split
all_labels = []
for split in splits:
    all_labels.extend(process_disaster_split(split))

#split no-disaster images into train, val, test
no_disaster_files = [f for f in os.listdir(no_disaster_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.shuffle(no_disaster_files)
n_total = len(no_disaster_files)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)

split_indices = {
    'train': no_disaster_files[:n_train],
    'val': no_disaster_files[n_train:n_train + n_val],
    'test': no_disaster_files[n_train + n_val:]
}

print("Processing no-disaster images..")
for split, files in split_indices.items():
    for filename in tqdm(files):
        src_path = os.path.join(no_disaster_dir, filename)
        dst_path = os.path.join(output_dir, split, 'no_disaster', filename)
        try:
            img = Image.open(src_path).convert('RGB')
            img_resized = img.resize(target_size)
            img_resized.save(dst_path)
            all_labels.append({'filename': dst_path, 'label': 'no_disaster'})
        except Exception as e:
            print(f"Failed processing {filename}: {e}")

#save labels to CSV
labels_csv_path = os.path.join(output_dir, 'labels.csv')
with open(labels_csv_path, 'w', newline='') as csvfile:
    fieldnames = ['filename', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for entry in all_labels:
        writer.writerow(entry)