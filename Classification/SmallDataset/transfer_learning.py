# Starter code for Part 1 of the Small Data Solutions Project
# 

#Set up image data for train and test

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
#from torchvision import datasets 
from torchvision import transforms as T
from TrainModel import train_model
from TestModel import test_model
from torchvision import models
from torch.optim.lr_scheduler import StepLR,OneCycleLR
import argparse

# use this mean and sd from torchvision transform documentation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

#Set up Transforms (train, val, and test)

#<<<YOUR CODE HERE>>>
def ds_dl():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #Set up Transforms (train, val, and test)

    #<<<YOUR CODE HERE>>>
    train_transforms = T.Compose([
        T.Resize(256),
        T.RandomRotation(30),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomResizedCrop((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    valid_transforms = T.Compose([
        T.Resize((256)),
        T.CenterCrop((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    train_path = "./imagedata-50/train/"
    valid_path = "./imagedata-50/val/"
    test_path = "./imagedata-50/test/"
    train_ds = torchvision.datasets.ImageFolder(root=train_path, transform=train_transforms)   
    valid_ds = torchvision.datasets.ImageFolder(root=valid_path, transform=valid_transforms)
    test_ds = torchvision.datasets.ImageFolder(root=test_path, transform=valid_transforms)
    #Set up DataLoaders (train, val, and test)
    batch_size = 10
    num_workers = 4
    #<<<YOUR CODE HERE>>>
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=1)
    #hint, create a variable that contains the class_names. You can get them from the ImageFolder
    class_names = train_ds.__dict__['classes']
    return train_loader, val_loader, test_loader, class_names


# Using the VGG16 model for transfer learning 
# 1. Get trained model weights
# 2. Freeze layers so they won't all be trained again with our data
# 3. Replace top layer classifier with a classifer for our 3 categories

#<<<YOUR CODE HERE>>>
def net(arch):
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights)
        for params in model.parameters():
            params.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Linear(1280, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
            )
    else:
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for params in model.parameters():
            params.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=4096, out_features=3, bias=True)
            )
    return model
    
    
def model_test(model, test_dl):
    test_accuracy = 0
    test_loss = 0
    ds_len = len(test_dl.dataset)
    criterion = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    with torch.no_grad():
        for images, labels in test_dl:
            batch_size = images.shape[0]
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            loss = criterion(out, labels)
            test_loss += loss.item() * batch_size
            test_accuracy += (torch.argmax(out, dim=1)==labels).sum().item()
    print(f'Test Loss: {round(test_loss/ds_len, 5)}, Test Accuracy: {round(test_accuracy/ds_len, 5)}')
# Train model with these hyperparameters
# 1. num_epochs 
# 2. criterion 
# 3. optimizer 
# 4. train_lr_scheduler 

#<<<YOUR CODE HERE>>>


# When you have all the parameters in place, uncomment these to use the functions imported above
def main(arch, lr):
    model = net(arch)
    num_epochs = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr) #0.0012)
    #train_lr_scheduler = OneCycleLR(optimizer, max_lr=0.05,epochs=num_epochs,steps_per_epoch=len(train_loader))
    train_lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    train_loader, val_loader, test_loader, class_names = ds_dl()
    trained_model = train_model(model, criterion, optimizer, train_lr_scheduler, train_loader, val_loader, num_epochs=num_epochs)
    torch.save(trained_model.state_dict(), arch+'_model.pt')
    model_test(trained_model, test_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transfer Learning")
    parser.add_argument(
        "--arch",
        type=str,
        default="vgg16",
        help="Architecture to use for model training. Default: vgg16",
    )
    parser.add_argument(
            "--lr",
            type=float,
            default=0.001,
            help="Learning Rate",
            )
    args = parser.parse_args()
    main(args.arch, args.lr)
    print("done")
