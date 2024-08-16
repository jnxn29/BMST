import scipy.io
import numpy as np
from PIL import Image
import os
import shutil


labels = scipy.io.loadmat(r'.\imagelabels.mat')
labels = np.array(labels['labels'][0]) - 1
print("labels:", labels)

setid = scipy.io.loadmat(r'.\setid.mat')
validation = np.array(setid['valid'][0]) - 1
np.random.shuffle(validation)
train = np.array(setid['trnid'][0]) - 1
np.random.shuffle(train)
test = np.array(setid['tstid'][0]) - 1
np.random.shuffle(test)


combined_train = np.concatenate((train, validation))


flower_dir = []
for img in os.listdir(r".\102flowers\jpg"):
    flower_dir.append(os.path.join(r".\102flowers\jpg", img))
flower_dir.sort()


des_folder_train = r".\Flower102\prepare_pic\train"
des_folder_test = r".\Flower102\prepare_pic\test"


label_counts = {}
for label in labels:
    if label not in label_counts:
        label_counts[label] = 1
    else:
        label_counts[label] += 1


for label, count in label_counts.items():
    indices = np.where(labels == label)[0]
    np.random.shuffle(indices)

    
    train_samples = int(0.8 * count)
    test_samples = count - train_samples

    
    new_train = indices[:train_samples]
    new_test = indices[train_samples:]

    
    for tid in new_train:
        img = Image.open(flower_dir[tid])
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        path = flower_dir[tid]
        base_path = os.path.basename(path)
        classes = str(label)
        class_path = os.path.join(des_folder_train, classes)
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        despath = os.path.join(class_path, base_path)
        img.save(despath)

    
    for tid in new_test:
        img = Image.open(flower_dir[tid])
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        path = flower_dir[tid]
        base_path = os.path.basename(path)
        classes = str(label)
        class_path = os.path.join(des_folder_test, classes)
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        despath = os.path.join(class_path, base_path)
        img.save(despath)