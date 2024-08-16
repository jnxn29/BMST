import csv
import glob
import os
import random
import shutil


def concat_csv(csv_list):
    file_with_label = {}
    for csv_path in csv_list:
        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for line in reader:
                label = line[1]
                if label not in file_with_label:
                    file_with_label[label] = []
                file_with_label[label].append(line[0])
    return file_with_label


def split_dataset(file_with_label, root, train_ratio=0.8):
    
    train_dir = os.path.join(root, "../data/train")
    test_dir = os.path.join(root, "../data/test")

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    
    for label, files in file_with_label.items():
        train_subdir = os.path.join(train_dir, str(label))
        test_subdir = os.path.join(test_dir, str(label))

        if not os.path.exists(train_subdir):
            os.makedirs(train_subdir)
        if not os.path.exists(test_subdir):
            os.makedirs(test_subdir)

        
        for file_name in files:
            shutil.move(os.path.join(root, "images", file_name), os.path.join(train_subdir, file_name))

    
    for label in os.listdir(train_dir):
        label_dir = os.path.join(train_dir, label)
        files = os.listdir(label_dir)
        num_files = len(files)
        num_train = int(num_files * train_ratio)

        
        test_samples = random.sample(files, num_files - num_train)

        for file_name in test_samples:
            src = os.path.join(label_dir, file_name)
            dst = os.path.join(test_dir, label, file_name)
            shutil.move(src, dst)

    print("Data set partition complete!")


def main():
    root = "./Mini-ImageNet"  
    csv_list = glob.glob(os.path.join(root, "*.csv"))  
    file_with_label = concat_csv(csv_list)  
    split_dataset(file_with_label, root, 0.8)  


if __name__ == "__main__":
    main()