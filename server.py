from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import classificationDemo

app = Flask(__name__)

model = classificationDemo()
model.load_state_dict(torch.load("classificationDemo.pth", map_location=torch.device("cpu")))
model.eval()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route("/")
def greet():
    return "Helllllllllooooo!"
    
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['image']
    image = Image.open(file).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(1).item()
        pred_class = class_names[pred_idx]

    return jsonify({"prediction": pred_class})

if __name__ == "__main__":
    app.run(debug=True)
