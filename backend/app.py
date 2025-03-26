import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

# Streamlit UI
st.title("AI-Powered Storyboard Creator 🎬")
st.sidebar.header("Customize Storyboard")

# Load the Stable Diffusion model (forcing CPU mode & fixing dtype issue)
model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
model.to("cpu")  # Ensure model is running on CPU
model.float()  # Convert model weights to float32 (fix dtype issue)

# Sidebar Inputs
prompt = st.text_area("Enter Storyboard Prompt", "A sci-fi spaceship flying through a nebula")
num_frames = st.slider("Number of Frames", min_value=1, max_value=5, value=3)
image_quality = st.radio("Image Quality", ["Low", "Medium", "High"], index=1)
black_and_white = st.checkbox("Black and White")

# Output Directory
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

# Function to generate images
def generate_image(prompt, black_and_white):
    image_prompt = f"{prompt}, storyboard frame"
    if black_and_white:
        image_prompt += ", black and white"
    
    image = model(image_prompt).images[0]  # Generate image
    image_path = os.path.join(output_dir, f"{prompt.replace(' ', '_')}.png")
    image.save(image_path)
    return image_path

# Generate and Display Images
if st.button("Generate Storyboard"):
    storyboard_images = []
    for _ in range(num_frames):
        img_path = generate_image(prompt, black_and_white)
        storyboard_images.append(img_path)

    st.subheader("Generated Storyboard")
    cols = st.columns(len(storyboard_images))
    for col, img_path in zip(cols, storyboard_images):
        col.image(Image.open(img_path), use_column_width=True)
