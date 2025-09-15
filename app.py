import os
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

# Import your model architecture
from xray.ml.model.arch import Net

# Model path
MODEL_PATH = "model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and load weights
model = Net().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Define classes
CLASS_NAMES = ['COVID19', 'NORMAL', 'PNEUMONIA']

# Define image transforms (must match training!)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Prediction function
def predict(image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        prediction = CLASS_NAMES[predicted.item()]
        confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item() * 100
    return prediction, confidence

# --- Streamlit UI ---
st.title("Lung Disease Prediction App")

# Upload section
uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

# Sample section
st.subheader("📂 Or choose from local samples")
SAMPLES_DIR = "samples"
sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

selected_sample = None
if sample_files:
    choice = st.selectbox("Select a sample image:", sample_files)
    if st.button("Use Selected Sample"):
        selected_sample = os.path.join(SAMPLES_DIR, choice)
else:
    st.info("No images found in `samples/` folder. Please add some.")

# Handle image selection
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", width=300)
elif selected_sample:
    image = Image.open(selected_sample).convert("RGB")
    st.image(image, caption=f"Sample X-ray: {os.path.basename(selected_sample)}", width=300)

# Prediction
if image is not None:
    prediction, confidence = predict(image)
    st.success(f"**Prediction:** {prediction}")
    st.info(f"**Confidence:** {confidence:.2f}%")