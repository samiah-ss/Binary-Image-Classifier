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
    
    fc_weight_shape = state_dict['fc.weight'].shape
    in_features = fc_weight_shape[1]                
    
    net.fc = nn.Linear(in_features, 2)
    net.load_state_dict(state_dict)
    net.eval()
    return net

model = load_pytorch_model()

st.title("Malware Binary Image Classifier")
st.write("Received a fishy email? Suspect a PDF to be malicious? Change the format to a malware binary visualization image. Then upload it here to check if it is benign or malicious.")

uploaded_file = st.file_uploader("Choose a PNG or JPG image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    st.image(raw_image, caption='Uploaded Binary Image.', use_container_width=True)
    st.write("Processing and classifying...")

    # 1. FORCE THE IMAGE TO CONVERT TO RGB FIRST
    # This strips away web transparency channels that corrupt the grayscale math!
    clean_image = raw_image.convert("RGB")

    # 2. RUN YOUR EXACT NOTEBOOK TRANSFORMS ON CLEAN DATA
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.NEAREST), 
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(clean_image)
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
