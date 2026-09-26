import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import zipfile
from sklearn.model_selection import train_test_split
import torchvision
import PIL
from io import BytesIO
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.optim as optim
from torchvision.transforms import v2
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split



with zipfile.ZipFile(r'C:\Users\диджей арбуз\Downloads\archive.zip') as zf:
    zf.extractall('data')


transform = v2.Compose([
    v2.Resize(256),
    v2.RandomResizedCrop(224),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

p = ImageFolder(root="data/td3_data", transform=transform)


p = ImageFolder(root="data/td3_data", transform=transform)


# (assuming 80/20 split)
train_size, test_size = int(len(p) * 0.8), len(p) - (int(len(p) * 0.8))
    

train_dataset, test_dataset = random_split(p, [train_size, test_size])

batch_train, batch_test = len(train_dataset), len(test_dataset)

#  создание dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_train, shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=batch_test)


model = torchvision.models.resnet18(weights = 'DEFAULT')  #transfer learning(раннее обучен. модели с их весами) 

criterion = nn.CrossEntropyLoss()

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)  


for param in model.parameters():
    param.requires_grad = False #полная заморозка слоев

# размораживаем последний блок 
for param in model.layer4.parameters():
    param.requires_grad = True


optimizer = optim.Adam(model.parameters(), lr=0.001) #learning rate град. спуск
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) 
model = model.to('cuda')#Только теперь на GPU

num_epochs = 10 

for epoch in range(num_epochs):   
    model.train()
    running_loss = 0.0
    
    for images, labels in train_dataloader : 
        images, labels = images.to( 'cuda' ), labels.to( 'cuda' )   # Перемещаем данные на GPU
        
        optimizer.zero_grad()   # Обнуляем буферы градиента(сложение со старым град.)
        outputs = model(images)   # ПРОХОДИМ ПО СЛОЯМ МОДЕЛИ, которые заранее проли проверку 
        loss = criterion(outputs, labels)   # Вычисляем потери(ОШИБКИ)
        loss.backward()   # Обратный проход. Определяет, какой вклад 
        #каждый параметр вносит в ошибку и в каком направлении модель должна двигаться, 
        # чтобы уменьшить её.
        optimizer.step() #Выполняет фактическое движение в сторону меньшего значения потерь.
        running_loss += loss.item()   # Обновляем веса модели
        scheduler.step()  # Обновляем веса оптимизатора

torch.save(model.state_dict(), "model_weights.pth")

model.eval()  # перевод в оценочный режим.
total, correct = 0, 0  

with torch.no_grad():  # отключение градиента(иначек оно опять будет считаться)
    for images, labels in test_dataloader:
        images, labels = images.to('cuda'), labels.to('cuda')  # Move data to GPU
        outputs = model(images)  
        _, predicted = outputs.max(1)  # получение класса с самой выс качеством 
        #(значения(tensor), индексы)
        total += labels.size(0) #общего количества обработанных картинок
        correct += (predicted == labels).sum().item()
        # 1 - сравнение двух тензоров
        # 2 - суммируем (1 или 0)
        #  Пояснение шагов 2 и 3
        # мы получаем после 2 шага ('torch.Tensor' и он на CPU/GPU мы переносим в RAM)
        # 3 - Возвращает значение этого тензора в виде стандартного числа Python


accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%') #79 - 83%


