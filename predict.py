import torch
from torchvision import transforms
from PIL import Image
from model import classificationDemo

#Load the model
model = classificationDemo()
model.load_state_dict(torch.load('classificationDemo.pth'))
model.eval()
print("Model loaded successfully!")

#Class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#Load a image
img = Image.open("./test_images/car.jpg").convert('RGB')

transformations = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,))
                                    ])
img_tensor = transformations(img).unsqueeze(0)
#Predict
with torch.no_grad():
    out = model(img_tensor)
    pred_index = out.argmax(1).item()
    pred_class = class_labels[pred_index]

print(f"Predicated class:{pred_class}")