import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from google import genai
from google.genai import types

# --- Page Setup & Styling ---
st.set_page_config(page_title="AgriSight AI Dashboard", layout="wide", page_icon="🌿")

# Custom CSS injection for premium look
st.markdown("""
    <style>
    .metric-box { padding: 15px; border-radius: 10px; background-color: #f0f2f6; border-left: 5px solid #4caf50; }
    .status-text { font-size: 1.2rem; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 AgriSight AI: Intelligent Crop Diagnostic System")
st.write("An advanced multi-modal system merging **Predictive Computer Vision (ResNet18)** with **Generative Agronomy (Gemini 3.5)**.")

# --- Initialize Gen AI Client ---
if "GEMINI_API_KEY" in os.environ:
    client = genai.Client()
else:
    client = None
    st.error("🔑 ERROR: `GEMINI_API_KEY` environment variable missing. Please export your key to enable the Generative AI layer.")

# --- Load Class Labels & Model ---
@st.cache_resource
def load_predictor():
    with open('models/classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load('models/crop_disease_resnet18.pth', map_location='cpu'))
    model.eval()
    return model, classes

model, class_names = load_predictor()

# --- Image Transform Pipeline ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Main Layout ---
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📸 Imagery Ingestion")
    uploaded_file = st.file_uploader("Upload leaf sample for neural network analysis...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Target Sample', use_container_width=True)
        analyze_btn = st.button("🚀 Execute Comprehensive Analysis", use_container_width=True)

with col2:
    st.subheader("📊 Diagnostic Diagnostics & Action Plan")
    
    if uploaded_file is not None and analyze_btn:
        # 1. RUN COMPUTER VISION BACKEND
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, index = torch.max(probabilities, 0)
        
        result_label = class_names[index.item()]
        clean_name = result_label.replace("___", " - ").replace("_", " ")
        conf_score = confidence.item() * 100
        is_healthy = "healthy" in result_label.lower()
        
        # Display Vision Telemetry Metrics
        st.markdown("### 🔌 Model Telemetry")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="Inference Classification", value=clean_name)
        with m_col2:
            st.metric(label="Classification Confidence", value=f"{conf_score:.2f}%")
            
        st.write("---")
        
        # 2. RUN GENERATIVE AI STREAMING ENGINE
        if is_healthy:
            st.success("✨ **Neural Network Verdict:** Leaf sample profile matches normal distribution parameter guidelines. Plant is healthy.")
        elif client:
            st.markdown("### 🤖 GenAI Agronomic Treatment Generation")
            
            # Formulating a production prompt with strict constraints
            prompt = f"""
            You are a Principal Plant Pathologist and senior Agricultural Advisor.
            A crop sample has been evaluated by a computer vision model and confirmed with a {conf_score:.2f}% confidence to have: {clean_name}.
            
            Generate a highly actionable management blueprint formatted beautifully in clean markdown. 
            Use clear bullet points and bold headers. Structure the response strictly into these 4 sections:
            
            ### 1. 🚨 Immediate Containment Actions
            - Give localized tasks (pruning protocols, tool sanitation with % concentrations, debris handling).
            
            ### 2. 🧪 Organic & Biological Remedies
            - Focus on natural remedies, compound application rules (e.g., weather constraints, sulfur/oil conflicts).
            
            ### 3. 🛡️ Conventional Chemical Treatments
            - Include structural compound recommendations using standard FRAC group rotations to avoid pathogen resistance. Add warning indicators.
            
            ### 4. 📅 Proactive Next-Cycle Prevention
            - Provide soil adjustments, canopy spacing, or winter treatment adjustments to break the infection cycle.
            """
            
            # Stream the Gen AI content directly to the UI container word-by-word
            def generate_stream():
                response_stream = client.models.generate_content_stream(
                    model='gemini-3.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2, # Lower temperature for rigorous, factual agronomic responses
                    )
                )
                for chunk in response_stream:
                    yield chunk.text
            
            # Render using Streamlit's built-in streaming container
            st.write_stream(generate_stream)