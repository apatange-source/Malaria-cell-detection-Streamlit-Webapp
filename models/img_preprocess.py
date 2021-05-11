import os
import cv2
from PIL import Image
import numpy as np
from sklearn.utils import shuffle

data = []
labels = []
Parasitized = os.listdir("cell_images/Parasitized/")
for parasite in Parasitized:
    image = cv2.imread("cell_images/Parasitized/" + parasite)
    print("cell_images/Parasitized/" + parasite)
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((150, 150))
    data.append(np.array(size_image))
    labels.append(0)


Uninfected = os.listdir("pythonProject/cell_images/Uninfected/")
for uninfect in Uninfected:
    image = cv2.imread("cell_images/Uninfected/" + uninfect)
    print("cell_images/Uninfected/" + uninfect)
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((150, 150))
    data.append(np.array(size_image))
    labels.append(1)

data, labels = shuffle(data, labels)


df = np.array(data)
labels = np.array(labels)
(X_train, X_val, X_test) = df[int(0.2 * len(df)):], df[int(0.1 * len(df)):int(0.2 * len(df))], df[:int(0.1 * len(df))]
(y_train, y_val, y_test) = labels[int(0.2 * len(labels)):], labels[int(0.1 * len(labels)):int(0.2 * len(labels))], labels[:int(0.1 * len(labels))]

np.savez("cell_images/training_arrays.npz", X_train, X_val, y_val, y_train, X_test, y_test)



