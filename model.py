import pandas as pd
import numpy as np
import torch
import regex as re

# data loading and pre-processing
data = pd.read_csv("datasets/Game_of_Thrones_Script.csv")
data_array = np.array(data.iloc[:, 5])

n = int(0.9 * float(data_array.shape[0]))

data_array = ["<START> " + str(phrase) + " <END>" for phrase in data_array]

"""
pattern = re.compile("'s|'t|'re|'ve|'m|'ll|'d|(?:\s{2,}(?=\S))| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+|\$+")
res = re.compile(pattern)
"""
print(data_array)

train_data = data_array[:n]
val_data = data_array[n:]
 
