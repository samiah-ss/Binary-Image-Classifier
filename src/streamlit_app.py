import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding = 1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = None

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        if self.fc is None:
            self.fc = nn.Linear(x.size(1), 2).to(x.device)
        return self.fc(x)

@st.cache_resource
def load_pytorch_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'benign_model.pth')
    
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    net = CNN()
    
    # Safely match your dynamic layer keys from Colab
    # Checks if your model saved it as 'fc.weight' or left it unmapped
    if 'fc.weight' in state_dict:
        in_features = state_dict['fc.weight'].shape[1]
        net.fc = nn.Linear(in_features, 2)
        
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    return net

model = load_pytorch_model()

st.title("Malware/Benign Binary Image Classifier")
st.write("Received a fishy email? Suspect a PDF to be malicious? Change the format to a malware binary visualization image. Then upload it here to check if it is benign or malicious.")

uploaded_file = st.file_uploader("Choose a PNG, JPEG, or JPG image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open the image directly in its native upload state
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Binary Image.', use_container_width=True)
    st.write("Processing and classifying...")

    # Mirror exact training transforms without any extra conversions
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim = 1)
        _, predicted_class = torch.max(outputs, 1)
    
    class_idx = predicted_class.item()
    confidence = probabilities[0][class_idx].item() * 100

    # Retain the correct classes
    labels = {0: "Benign", 1: "Malware"} 
    result_label = labels[class_idx]

    HIGH_CONFIDENCE_THRESHOLD = 80.0
    LOW_CONFIDENCE_THRESHOLD = 60.0

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            "**Potential Obfuscation Detected**\n\n"
            "Our system flagged this file because its structures are highly scrambled. "
            "Modern malware deliberately mimics benign files to dodge detection algorithms. "
            "Treat this file as a high risk until verified."
        )
    elif confidence < HIGH_CONFIDENCE_THRESHOLD:
        st.warning(
            f"**Suspicious Anomalies Detected**\n\n"
            f"The model leans toward **{result_label}**, but detected conflicting structural "
            f"signals. This overlap is standard when analyzing files that employ evasion tactics."
            f"Treat this file as a risk until verified."
        )
    else:
        if result_label == "Malware":
            st.error(f"🔴 **Malware Detected!** (Certainty: {confidence:.2f}%)")
        else:
            st.success(f"🟢 **Seems Safe!** (Certainty: {confidence:.2f}%)")

st.write("Made by Samiah Siddiqua")