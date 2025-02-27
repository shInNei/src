import os
import json
import shutil
from PIL import Image
from torchvision import transforms

# Paths
input_root = "dataset"
output_root = "dataset"
os.makedirs(output_root, exist_ok=True)

# Define augmentations
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(10, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=1),
])

# Get the last action number
existing_folders = [f for f in os.listdir(input_root) if f.startswith("action")]
last_action_num = max([int(f[6:]) for f in existing_folders])  # Extract numbers from "actionXXX"

# Loop through each action folder
for action_folder in existing_folders:
    action_path = os.path.join(input_root, action_folder)

    if os.path.isdir(action_path):
        img_path = os.path.join(action_path, "image.jpg")
        json_path = os.path.join(action_path, "info.json")

        # Open image
        img = Image.open(img_path)

        # Create 5 augmented versions
        for i in range(10):
            last_action_num += 1
            new_action_folder = f"action{last_action_num}"
            new_action_path = os.path.join(output_root, new_action_folder)
            os.makedirs(new_action_path, exist_ok=True)

            # Apply augmentation
            augmented_img = augmentations(img)
            augmented_img.save(os.path.join(new_action_path, "image.jpg"))

            # Copy info.json as it is
            shutil.copy(json_path, os.path.join(new_action_path, "info.json"))

print("✅ Augmentation completed! Check 'augmented_dataset' folder.")
