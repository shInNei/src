import os
import json

# Directory where your dataset is located
dataset_dir = 'dataset'

# Function to check if any action folder contains the specified distance
def check_for_distance(dataset_dir, target_distance):
    # Loop through each action folder in the dataset
    for action_index, action_folder in enumerate(os.listdir(dataset_dir)):
        action_folder_path = os.path.join(dataset_dir, action_folder)

        if os.path.isdir(action_folder_path):
            # Look for the JSON file in the action folder
            json_file = None
            for file in os.listdir(action_folder_path):
                if file.endswith('.json'):
                    json_file = os.path.join(action_folder_path, file)
                    break
            
            if json_file is None:
                continue  # Skip if no JSON file is found

            # Read the JSON file and extract the distance value
            with open(json_file, 'r') as f:
                data = json.load(f)
                distance = data.get('distance')  # Replace with the correct key if necessary

            # Check if the distance matches the target distance
            if distance == target_distance:
                print(f"Found target distance {target_distance} in Action {action_index} (Folder: {action_folder})")

# Specify the target distance you are looking for
target_distance = 152

# Run the function to check for the target distance
check_for_distance(dataset_dir, target_distance)
