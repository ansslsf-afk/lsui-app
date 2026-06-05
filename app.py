import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import segmentation_models_pytorch as smp

st.set_page_config(page_title="LSUI Enhancer", page_icon="🌊", layout="centered")
st.title("🌊 Underwater Image Enhancer")
st.write("Upload gambar bawah air yang buram atau kehijauan, dan model AI (U-Net + EfficientNet-B0) akan memperbaiki warnanya!")

@st.cache_resource
def load_model():
    model = smp.Unet(
        encoder_name='efficientnet-b0',
        encoder_weights=None,
        in_channels=3,
        classes=3,
        activation='sigmoid'
    )

    model.load_state_dict(torch.load('model_smp_weights.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

uploaded_file = st.file_uploader("choose picture", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(raw_image, caption="Gambar Asli (Input)", use_column_width=True)
        
    with st.spinner('AI sedang memproses gambar...'):
      
        input_tensor = transform(raw_image).unsqueeze(0)
        
    
        with torch.no_grad():
            output_tensor = model(input_tensor)
            
        output_image = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
        output_image = (output_image * 255).clip(0, 255).astype(np.uint8)
        
    with col2:
        st.image(output_image, caption="Hasil Enhancement (Output)", use_column_width=True)