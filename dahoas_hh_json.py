from datasets import load_dataset
import json
from random import shuffle

# Load the dataset from Hugging Face
dataset = load_dataset("Dahoas/full-hh-rlhf")

# Shuffle the dataset
shuffled_train_data = dataset['train'].shuffle(seed=42) 
shuffled_test_data = dataset['test'].shuffle(seed=42) 

# # Assuming you want to slice the dataset after shuffling
# train_slice = shuffled_data.select(range(17000, 37000))
# val_slice = shuffled_data.select(range(1000))

# Function to save a slice of the dataset to a JSON file
def save_to_json(data_slice, file_path):
    data_to_save = {
        "chosen": data_slice['chosen'],
        "rejected": data_slice['rejected'],
        "prompt": data_slice['prompt'],
        "response":data_slice['response']
    }
    with open(file_path, 'w') as f:
        json.dump(data_to_save, f)

# Save the slices to JSON files
save_to_json(shuffled_train_data, 'train_data.json')
save_to_json(shuffled_test_data, 'val_data.json')

print("Files saved: train_data.json, val_data.json")
