import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

st.set_page_config(page_title="AI Crop Disease Detector", layout="centered")
st.title("🌿 AI-Based Crop Disease Prediction System")
st.write("Upload a clear photo of a crop leaf to identify pathologies in real-time.")

@st.cache_resource # Caches the model in RAM so it only loads ONCE, making the app blazing fast
def load_predictor():
    with open('models/classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    
    model.load_state_dict(torch.load('models/crop_disease_resnet18.pth', map_location='cpu'))
    model.eval()
    return model, classes

model, class_names = load_predictor()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Leaf Image', use_container_width=True)
    
    if st.button("Analyze Leaf Health"):
        with st.spinner('Analyzing patterns...'):
            # Preprocess
            input_tensor = transform(image).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, index = torch.max(probabilities, 0)
            
            result_label = class_names[index.item()]
            clean_name = result_label.replace("___", " - ").replace("_", " ")
            conf_score = confidence.item() * 100
            
            st.write("---")
            if "healthy" in result_label.lower():
                st.success(f"**Status:** The crop appears to be **Healthy**!")
            else:
                st.error(f"**Pathology Detected:** {clean_name}")
                
            st.metric(label="Confidence Score", value=f"{conf_score:.2f}%")