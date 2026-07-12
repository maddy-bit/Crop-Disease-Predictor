import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

with open('models/classes.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

model.load_state_dict(torch.load('models/crop_disease_resnet18.pth', map_location=device))
model = model.to(device)
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device) # unsqueeze adds a batch dimension

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, preds = torch.max(probabilities, 0)
            
        return class_names[preds.item()], conf.item() * 100
    except Exception as e:
        return f"Error processing image: {str(e)}", 0.0

if __name__ == "__main__":
    test_path = "split_data/test/Apple___Apple_scab/1cb869ea-0a4c-47d7-9def-a88c16b72ddc___FREC_Scab 3350.JPG" 
    
    import os
    if os.path.exists(test_path):
        label, confidence = predict_image(test_path)
        print(f"\nPrediction: {label}")
        print(f"Confidence: {confidence:.2f}%")
    else:
        print(f"\nPlease update 'test_path' in predict.py to a real image file path!")