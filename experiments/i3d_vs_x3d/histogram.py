import json

import matplotlib.pyplot as plt

# Load the JSON data
with open("data/processed/extractor/train_10.json", "r") as f:
    data = json.load(f)

# Extract the video IDs for the "book" class
glosses = [item["gloss"] for item in data]

fig = plt.figure(figsize=(10, 5))
plt.hist(glosses, bins=10, density=True, edgecolor="black", linewidth=1)
plt.savefig("experiments/hist.png")
plt.close()
