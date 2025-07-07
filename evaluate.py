import torch
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from model import classificationDemo

from sklearn.metrics import confusion_matrix,recall_score,f1_score,accuracy_score

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device using:{DEVICE}")

transformations = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5,),(0.5,))
                                 ])

test_dataset = datasets.CIFAR10(root='./data',train= False, transform = transformations)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)

model = classificationDemo()
model.load_state_dict(torch.load("classificationDemo.pth"))
model.to(DEVICE)
model.eval()

total_pred =[]
total_labels =[]
total_loss = 0
criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    for images,labels in test_loader:
        images,labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs,labels)
        total_loss += loss.item()*images.size(0)

        _,pred = torch.max(outputs,1)
        #batch tensor to CPU as gpu cant handle and all batches united to build total
        total_pred.extend(pred.cpu().numpy())
        total_labels.extend(labels.cpu().numpy())

#Append array's to np
test_true = np.array(total_labels)
test_pred = np.array(total_pred)

#Calculations
conf_mat = confusion_matrix(test_true,test_pred)
recall = recall_score(test_true,test_pred,average=None)
acc_score = accuracy_score(test_true,test_pred)
avg_loss = total_loss / len(test_loader)

#pd dataframs
conf_df = pd.DataFrame(conf_mat,
                       index = [f"Label_{i+1}" for i in range(10)],
                       columns= [f"Pred_{i+1}" for i in range(10)]
                       )

recall_df = pd.DataFrame([recall],
                         index=["Recall"], 
                         columns= [f"Class_{i+1}" for i in range(10)]
                         )

summary_df = pd.DataFrame({'Metric':['Loss','Accuracy'], 
                           'Values':[avg_loss,acc_score]} )

#Save as CSV
conf_df.to_csv("Confusion_matrix.csv")
recall_df.to_csv("Recall.csv")
summary_df.to_csv("Summary.csv")

print("✅ Evaluation complete. CSV files saved.")

#Plot
plt.figure(figsize=(10,8))
sns.heatmap(conf_mat, 
            annot=True,
            fmt='d',
            xticklabels=class_names,
            yticklabels=class_names
            )

plt.title('Confusion Matrix (CIFAR-10)')
plt.xlabel('Predicate')
plt.ylabel('Labels')
plt.tight_layout()
plt.savefig('Confusion_Matrix_(CIFAR-10).png')
plt.close()

print("Heatmap is saved successfully!")