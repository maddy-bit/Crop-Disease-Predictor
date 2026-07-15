import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from google import genai
from google.genai import types
from google.genai.errors import ServerError

st.set_page_config(
    page_title="AgriSight AI Dashboard", 
    layout="wide", 
    page_icon="🌿",
    initial_sidebar_state="expanded"
)

# Soft, clean design overrides to make the dashboard look highly polished and readable
st.markdown("""
    <style>
    .reportview-container {
        background-color: #fafbfc;
    }
    .status-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #eef2f6;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1b2a4a;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    </style>
""", unsafe_allow_html=True)

if "GEMINI_API_KEY" in os.environ:
    client = genai.Client()
else:
    client = None

@st.cache_resource
def load_predictor():
    with open('models/classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load('models/crop_disease_resnet18.pth', map_location='cpu'))
    model.eval()
    return model, classes

try:
    model, class_names = load_predictor()
except Exception as e:
    st.error(f"⚠️ Error loading deep learning weights: {str(e)}")
    class_names = []

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

with st.sidebar:
    st.image("https://img.shields.io/badge/System-Active-success?style=for-the-badge&logo=opsgenie", use_container_width=False)
    st.markdown("## 🌿 About AgriSight AI")
    st.write(
        "This diagnostic companion merges **Predictive Deep Learning** with "
        "**Generative Agronomic Triage** to help you identify crop stress and disease instantly."
    )
    
    st.divider()
    st.markdown("### 📋 How to use:")
    st.markdown(
        "1. **Upload** a clear image of an infected leaf in the main panel.\n"
        "2. Click **Run Diagnostics**.\n"
        "3. Review the AI-generated triage severity indicator and step-by-step treatment plan."
    )
    st.divider()
    
    if not client:
        st.warning("🔑 `GEMINI_API_KEY` not found in environment. The Generative triage layer is deactivated.")
    else:
        st.success("🔌 Gemini Triage Gateway connected successfully.")

st.title("🌿 AgriSight AI")
st.caption("Intelligent Multi-Modal Crop Diagnosis & Automated Severity Triage")
st.write("---")

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown("### 📸 Leaf Sample Upload")
    uploaded_file = st.file_uploader(
        "Drag and drop or browse for a leaf sample image...", 
        type=["jpg", "jpeg", "png"],
        help="Make sure the leaf is centered, well-lit, and the infected area is clearly visible."
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Displaying image inside a clean, rounded card layout
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.image(image, caption='Uploaded Target Leaf Sample', use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        analyze_btn = st.button("🚀 Run Diagnostics", type="primary", use_container_width=True)
    else:
        st.info("💡 Upload a leaf sample image to start the analysis pipeline.")
        analyze_btn = False

with col2:
    st.markdown("### 📊 Diagnostic Analysis Results")
    
    if uploaded_file is not None and analyze_btn:
        with st.spinner("Processing leaf sample..."):
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, index = torch.max(probabilities, 0)
            
            result_label = class_names[index.item()]
            clean_name = result_label.replace("___", " - ").replace("_", " ")
            conf_score = confidence.item() * 100
            is_healthy = "healthy" in result_label.lower()
            
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.markdown("⚡ **Deep Learning Inference Complete**")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(f'<div class="metric-label">Detected Issue</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{clean_name}</div>', unsafe_allow_html=True)
        with m_col2:
            st.markdown(f'<div class="metric-label">Model Confidence</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{conf_score:.2f}%</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        if is_healthy:
            st.success("🟢 **Analysis Result:** Leaf matches healthy parameters. No disease symptoms identified.")
        elif client:
            st.markdown("### 🏥 Agronomic Severity Triage")
            
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
            
            def generate_stream():
                try:
                    response_stream = client.models.generate_content_stream(
                        model='gemini-3.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.2,
                        )
                    )
                    for chunk in response_stream:
                        if chunk.text:
                            yield chunk.text
                except ServerError as e:
                    yield f"⚠️ **Gemini API Server Error:** The server is currently experiencing high demand. Please try again in a few moments. \n\n*Details: {str(e)}*"
                except Exception as e:
                    yield f"⚠️ **An unexpected error occurred:** {str(e)}"
            
            st.write_stream(generate_stream)