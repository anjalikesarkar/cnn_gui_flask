import os
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from model import classificationDemo

#Hyperparamters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NO_EPOCHS = 10

#Device 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device used:{DEVICE}')

#Transformations
transformations = transforms.Compose([ transforms.ToTensor(),
                                      transforms.Normalize((0.5,),(0.5,))

])

#Dataset
trainset = datasets.CIFAR10(root='./data',
                            train= True,
                            download=True,
                            transform=transformations)

testset = datasets.CIFAR10(root='./data',
                           train= False,
                           download=True,
                           transform=transformations)

#Dataloader
train_loader = DataLoader(trainset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

test_loader = DataLoader(testset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)

print("Data loaded successfully!")

#Initialize the model 
model = classificationDemo().to(DEVICE)
criterian = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr= LEARNING_RATE)

#Loop for training
for epoch in range(NO_EPOCHS):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for images,labels in (train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        #Forward pass
        output = model(images)
        loss = criterian(output, labels)

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _,prediction = torch.max(output,1)
        correct = (prediction == labels)
        total_correct += correct.sum().item()
        total_samples += labels.size(0)

    training_acc = (total_correct*100)/total_samples
    print(f"Epoch:{epoch+1}/{NO_EPOCHS}|Loss:{loss.item():.4f}| Training_Accuracy:{training_acc:.2f}")
print("Model Trained Successfully!")

#Save model
torch.save(model.state_dict(),"classificationDemo.pth")
print("Model saved!")