import os
import pandas as pd

current_dir = os.curdir
files = []

for dir in os.listdir(current_dir):
    if dir.endswith(".csv"):
        files.append(dir)

        
df1=pd.concat([pd.read_csv(file) for file in files],ignore_index=True)

df1.to_csv('2_layer.csv')

print('success!')