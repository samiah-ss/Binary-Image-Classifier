import streamlit as st

st.set_option("server.enableXsrfProtection", False)
st.set_option("server.enableCORS", False)

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os  # Added to safely track path locations across the server

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

# Use a modified loading function to handle the dynamic fc layer safely
@st.cache_resource
def load_pytorch_model():
    # Automatically finds the folder where this specific script lives inside the container
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'benign_model.pth')
    
    # Load the raw state weights dict using the bulletproof path
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    net = CNN()
    
    # Extract the dynamic fc dimensions straight from the saved weights file
    fc_weight_shape = state_dict['fc.weight'].shape
    in_features = fc_weight_shape[1]                # 8192
    
    # Manually initialize fc so load_state_dict doesn't throw a key error
    net.fc = nn.Linear(in_features, 2)
    
    # Load the weights into the structure cleanly
    net.load_state_dict(state_dict)
    net.eval()
    return net

model = load_pytorch_model()

st.title("Malware Binary Image Classifier")
st.write("Received a fishy email? Suspect a PDF to be malicious? Change the format to a malware binary visualization image. Then upload it here to check if it is benign or malicious.")

uploaded_file = st.file_uploader("Choose a PNG or JPG image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Binary Image.', use_container_width=True)
    st.write("Processing and classifying...")

    # MATCHED TO YOUR NOTEBOOK PIPELINE + FORCED SHARP PIXELS
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.NEAREST), 
        transforms.ToTensor(),
    ])
    
    # Sent raw image directly into the corrected pipeline
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted_class = torch.max(outputs, 1)
        
    class_idx = predicted_class.item()
    confidence = probabilities[0][class_idx].item() * 100

    labels = {0: "Benign", 1: "Malware"} 
    result_label = labels[class_idx]

    if result_label == "Malware":
        st.error(f"**Malware Detected!** (Confidence: {confidence:.2f}%)")
        st.write("Have more images? You can scroll back up to upload them!")
    else:
        st.success(f"**Seems Safe!** (Confidence: {confidence:.2f}%)")
        st.write("Have more images? You can scroll back up to upload them!")
