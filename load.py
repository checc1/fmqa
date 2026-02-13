import os
import numpy as np

n = 16
file_path = os.path.join(os.getcwd(), "dicts", f"dataframe_model{n}.npz")
data = np.load(file_path, allow_pickle=True)
rows = data["rows"]

print(type(rows))
print(len(rows))
sample = rows[0]

print(type(sample))
print(sample.keys())

print("Class:", sample["Class"])
print("Idx:", sample["Idx"])
print("Image shape:", sample["Image"].shape)
print("FMaps shape:", sample["FMaps"].shape)
print("Alpha:", sample["Grad-contribution"])
print("Feature maps:", sample["FMaps"])