import json
import matplotlib.pyplot as plt

# Step 1: Load the JSON data
file_path = 'your_file_path_here.json'  # Change this to your actual file path
with open(file_path, 'r') as file:
    data = json.load(file)

before = data['before']
after = data['after']

# Step 2: Sort 'before' and reorder 'after' accordingly
sorted_indices = sorted(range(len(before)), key=lambda k: before[k])
sorted_before = [before[i] for i in sorted_indices]
sorted_after = [after[i] for i in sorted_indices]

# Step 3: Plot the sorted pairs
plt.figure(figsize=(10, 5))
plt.plot(sorted_before, sorted_after, marker='o')
plt.xlabel('Before')
plt.ylabel('After')
plt.title('Plot of Before vs. After')
plt.grid(True)
plt.show()
