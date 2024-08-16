import os
import torch
from torchvision import datasets, transforms

def calculate_mean_std(data_dir, batch_size=128):

    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)

    
    for images, _ in data_loader:
        
        images = images.view(-1, 3, 224 * 224)

        
        batch_mean = images.mean(dim=[0, 2])
        batch_std = images.std(dim=[0, 2])

        
        mean += batch_mean
        std += batch_std

    
    mean /= len(data_loader)
    std /= len(data_loader)

    return mean.tolist(), std.tolist()


if __name__ == "__main__":
    data_dir = "./data/"  
    mean, std = calculate_mean_std(data_dir)
    print("Mean:", mean)
    print("Standard Deviation:", std)