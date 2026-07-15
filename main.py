import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from google import genai
from google.genai import types
from google.genai.errors import ServerError

# --- Page Setup & Styling ---
st.set_page_config(page_title="AgriSight AI Dashboard", layout="wide", page_icon="🌿")

# Custom CSS injection for premium look
st.markdown("""
    <style>
    .metric-box { padding: 15px; border-radius: 10px; background-color: #f0f2f6; border-left: 5px solid #4caf50; }
    .status-text { font-size: 1.2rem; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 AgriSight AI: Intelligent Multi-Modal Crop Diagnostic System")
st.write("An advanced multi-modal system merging **Predictive Computer Vision (ResNet18)** with **Generative Agronomy Triage (Gemini 3.5)**.")

# --- Initialize Gen AI Client ---
if "GEMINI_API_KEY" in os.environ:
    client = genai.Client()
else:
    client = None
    st.error("🔑 ERROR: `GEMINI_API_KEY` environment variable missing. Please add your key to enable the Generative AI layer.")

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
            st.markdown("### 🤖 GenAI Agronomic Triage & Action Plan")
            
            # Intelligent Triage Routing System Prompt
            prompt = f"""
            You are a Principal Plant Pathologist and Senior Agricultural Extension Officer.
            A crop sample was analyzed by a computer vision network and identified with {conf_score:.2f}% confidence to have: {clean_name}.
            
            First, evaluate whether this specific disease ({clean_name}) is a "Critical Case" (e.g., highly infectious blights like Late Blight, quarantine vectors, or rapid epidemic outbreaks) or a "Manageable Case" (e.g., localized minor fungal spots, mildews, rusts, or cosmetic leaf anomalies).
            
            Based on your triage evaluation, you must output your response in clean Markdown using one of the two strict routing paths below:

            --------------------------------------------------
            PATHWAY A: MANAGEABLE CASE (🟢 Minor/Contained Infection)
            If the infection is manageable, print a badge at the top: `[TRIAGE: MANAGEABLE CASE 🟢]`
            Then provide these sections:
            1. **🚨 Immediate Containment Actions:** Local actions (how to prune, sanitize tools using specific % alcohol/bleach, and safely discard debris).
            2. **🧪 Organic & Biological Remedies:** Non-chemical solutions and application warnings (e.g., temperature rules, sulfur/oil conflicts).
            3. **🛡️ Conventional Chemical Treatments:** Standard chemical remedies utilizing FRAC rotation guidelines to prevent pathogen resistance.

            --------------------------------------------------
            PATHWAY B: CRITICAL CASE (🔴 High-Threat/Epidemic/Quarantine Vector)
            If the disease is highly contagious, destructive, or a quarantine risk, print a warning badge at the top: `[TRIAGE: CRITICAL OUTBREAK ALERT 🔴]`
            Then provide these sections:
            1. **🛑 Emergency Isolation Steps:** Strict physical steps to isolate the infected area immediately to prevent farm-wide or region-wide devastation. Do not recommend basic DIY home remedies.
            2. **📞 Expert Escalation Protocol:** Specific details on WHOM to contact immediately (e.g., local state agricultural extension office, university diagnostic labs, certified crop advisors, or department of agriculture).
            3. **📝 Onsite Diagnostic Checklist:** A quick checklist of observations and historical data the farmer should gather before the expert arrives on-site.
            --------------------------------------------------
            
            Generate the appropriate pathway plan clearly and structure it using distinct markdown titles, bolding, and clean bullet lists. Do not show or discuss the unused pathway. Keep instructions highly practical.
            """
            
            # Stream the Gen AI content directly to the UI container word-by-word with error handling
            def generate_stream():
                try:
                    response_stream = client.models.generate_content_stream(
                        model='gemini-3.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.2, # Lower temperature for rigorous, factual agronomic responses
                        )
                    )
                    for chunk in response_stream:
                        if chunk.text:
                            yield chunk.text
                except ServerError as e:
                    yield f"⚠️ **Gemini API Server Error:** The server is currently experiencing high demand. Please try again in a few moments. \n\n*Details: {str(e)}*"
                except Exception as e:
                    yield f"⚠️ **An unexpected error occurred:** {str(e)}"
            
            # Render using Streamlit's built-in streaming container
            st.write_stream(generate_stream)