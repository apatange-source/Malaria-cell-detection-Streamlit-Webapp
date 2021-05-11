import os
import pandas as pd

os.chdir(r'cell_images/train')

folders = ['Parasitized', 'Uninfected']

files = []

for folder in folders:
    for file in os.listdir(folder):
        files.append([file, folder])

os.chdir(r'C:\Users\aishw\PycharmProjects\pythonProject\cell_images\train')
pd.DataFrame(files, columns=['files', 'target']).to_csv('train_files_and_targets.csv')