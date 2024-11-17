import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import uuid
import os
import numpy as np

# Global statistics
stats = {"total_reports": 0, "abnormal_reports": 0, "healthy_reports": 0}

# Load the CheXNet model using pretrained weights from PyTorch
@st.cache_resource
def load_chexnet():
    try:
        # Initialize the DenseNet-121 model with pretrained weights
        model = models.densenet121(pretrained=True)
        
        # Modify the classifier to match your number of output classes (14 for CheXNet)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, 14)

        # Set the model to evaluation mode
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess the input image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = image.convert('RGB')
    return transform(image).unsqueeze(0)

# Predict diseases using the model
def predict_diseases(image, model, threshold=0.5):
    try:
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.sigmoid(outputs).squeeze().numpy()
        classes = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
            "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
        ]
        return {
            cls: float(prob)
            for cls, prob in zip(classes, probabilities)
            if prob >= threshold
        }
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return {}

# Generate a report summary
def generate_report_summary(predictions):
    thresholds = {
        "Hernia": {"threshold": 0.72, "severity": "high"},
        "Pleural_Thickening": {"threshold": 0.61, "severity": "high"},
        "Fibrosis": {"threshold": 0.73, "severity": "high"},
        "Emphysema": {"threshold": 0.72, "severity": "moderate"},
        "Edema": {"threshold": 0.72, "severity": "high"},
        "Consolidation": {"threshold": 0.67, "severity": "moderate"},
        "Pneumothorax": {"threshold": 0.80, "severity": "critical"},
        "Pneumonia": {"threshold": 0.69, "severity": "high"},
        "Nodule": {"threshold": 0.68, "severity": "moderate"},
        "Mass": {"threshold": 0.78, "severity": "high"},
        "Infiltration": {"threshold": 0.69, "severity": "moderate"},
        "Effusion": {"threshold": 0.51, "severity": "moderate"},
        "Atelectasis": {"threshold": 0.58, "severity": "moderate"},
        "Cardiomegaly": {"threshold": 0.65, "severity": "moderate"}
    }

    diseases = []
    severity_level = "normal"
    severity_priority = {"critical": 4, "high": 3, "moderate": 2, "normal": 1}

    for condition, prob in predictions.items():
        threshold_info = thresholds.get(condition, {"threshold": 0.5, "severity": "moderate"})
        if prob >= threshold_info["threshold"]:
            diseases.append({
                "name": condition,
                "confidence": prob,
                "severity": threshold_info["severity"]
            })
            if severity_priority[threshold_info["severity"]] > severity_priority[severity_level]:
                severity_level = threshold_info["severity"]

    if not diseases:
        condition = "Healthy Lungs"
        recommendations = "No abnormalities detected. Maintain regular health check-ups."
        is_healthy = True
    else:
        diseases.sort(key=lambda x: x["confidence"], reverse=True)
        condition = ", ".join([d["name"] for d in diseases])
        recommendations_map = {
            "critical": "URGENT: Immediate medical attention required.",
            "high": "Consult a healthcare provider as soon as possible.",
            "moderate": "Schedule a follow-up with your healthcare provider."
        }
        recommendations = recommendations_map.get(severity_level, "")
        is_healthy = False

    stats["total_reports"] += 1
    if is_healthy:
        stats["healthy_reports"] += 1
    else:
        stats["abnormal_reports"] += 1

    return condition, recommendations, is_healthy, severity_level, diseases

# Generate charts for predictions
def generate_charts(predictions, diseases):
    plt.figure(figsize=(12, 6))

    # Plot all condition probabilities
    plt.subplot(1, 2, 1)
    all_conditions = list(predictions.keys())
    all_probs = list(predictions.values())
    plt.barh(all_conditions, all_probs, color='skyblue')
    plt.xlabel("Probability")
    plt.title("All Conditions Probabilities")

    # Plot detected conditions with severity
    if diseases:
        plt.subplot(1, 2, 2)
        detected_names = [d["name"] for d in diseases]
        detected_probs = [d["confidence"] for d in diseases]
        colors = ['red' if d["severity"] == "critical"
                  else 'orange' if d["severity"] == "high"
                  else 'yellow' for d in diseases]
        plt.barh(detected_names, detected_probs, color=colors)
        plt.xlabel("Probability")
        plt.title("Detected Conditions by Severity")

    plt.tight_layout()
    st.pyplot(plt)

def hide_streamlit_footer():
    hide_footer_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(hide_footer_style, unsafe_allow_html=True)


def set_gradient_background():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #2b5876, #4e4376);
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def auto_scroll_to_results():
    js = '''
    <script>
        // Wait for content to load
        setTimeout(function() {
            // Target the Streamlit root container
            const element = window.parent.document.querySelector('.main.css-k1vhr4.egzxvld3');
            if (element) {
                // Use smooth scrolling
                element.scrollTo({
                    top: element.scrollHeight,
                    behavior: 'smooth'
                });
            }
        }, 1000);  // 1 second delay
    </script>
    '''
    
    # Inject the JavaScript with required height to ensure it runs
    st.components.v1.html(js, height=0)

# Main application
def main():
    # st.title("Health Report - Chest X-ray Classification")
    # st.markdown("""
    # Upload a chest X-ray image to generate a detailed health report using CheXNet.
    # The results include disease probabilities and recommendations.
    # """)
    
    set_gradient_background()
    hide_streamlit_footer()

    # Center-align and style title and header
    st.markdown(
        """
        <style>
        .title {
            font-size: 3rem;
            color: #ffffff;
            text-align: center;
            font-weight: bold;
            margin-bottom: 20px;
            margin-top: 20px;
        }
         .title2 {
            font-size: 2rem;
            color: #ffffff;
            text-align: center;
            font-weight: bold;
            margin-bottom: 20px;
            margin-top: 20px;
        }
        .header {
            font-size: 1.5rem;
            color: #e6e6e6;
            text-align: center;
            margin-bottom: 30px;
        }
        .result-box {
            background-color: rgba(0, 0, 0, 0.7);
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            color: #ffffff;
            text-align: center;
        }
        .result-title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #ffa726;
        }
        .result-score {
            font-size: 1.8rem;
            color: #81c784;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="title">🩺 LungScope 🩺</div>', unsafe_allow_html=True)

    # Display title
    st.markdown('<div class="title2">Health Report - Chest X-ray Classification</div>', unsafe_allow_html=True)

    # Display header
    st.markdown('<div class="header">Please upload a chest X-ray image</div>', unsafe_allow_html=True)


    # File uploader
    file = st.file_uploader("Upload Chest X-ray Image", type=["jpeg", "jpg", "png", "jfif"])

    if file is not None:
    # Read and display the image
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Process the image with loading spinner
        with st.spinner('Processing image... Please wait.'):
            model = load_chexnet()
            if model is not None:
                predictions = predict_diseases(image, model)
                if predictions:
                    condition, recommendations, is_healthy, severity_level, diseases = generate_report_summary(predictions)

                    # Display results
                    severity_icons = {
                        "critical": "🔴",
                        "high": "🟠",
                        "moderate": "🟡",
                        "normal": "🟢"
                    }
                    
                    st.success(f"### Status: {severity_icons.get(severity_level, '')} {condition}")
                    st.markdown(f"<h4>Severity Level: {severity_level.upper()}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<h5>Recommendations: {recommendations}</h3>", unsafe_allow_html=True)
                    # Detailed findings
                    if not is_healthy:
                        st.markdown("### Detailed Findings")
                        for disease in diseases:
                            st.write(f"- {disease['name']}: {disease['confidence']:.2%} ({disease['severity']})")
                        generate_charts(predictions, diseases)
                    
                    # Auto-scroll to results
                    auto_scroll_to_results()
            else:
                st.error("Model could not be loaded.")

if __name__ == "__main__":
    main()
