import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import threading
import time
import pkg_resources

# Streamlit UI
st.title("AI-Powered Storyboard Creator 🎬")
st.sidebar.header("Customize Storyboard")

# Log package versions for debugging
st.sidebar.write("Debug Info:")
st.sidebar.write(f"Python Version: {pkg_resources.get_distribution('python').version}")
st.sidebar.write(f"Streamlit Version: {pkg_resources.get_distribution('streamlit').version}")
st.sidebar.write(f"Torch Version: {pkg_resources.get_distribution('torch').version}")
st.sidebar.write(f"Diffusers Version: {pkg_resources.get_distribution('diffusers').version}")

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Running on: {device.upper()}")

# Load the Stable Diffusion model
@st.cache_resource
def load_model():
    try:
        model = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float32,
            use_auth_token=False
        )
        model.to(device)
        return model
    except Exception as e:
        st.error(f"Failed to load the model: {str(e)}")
        return None

# Load the model (cached to avoid reloading on every run)
model = load_model()

if model is None:
    st.stop()  # Stop the app if the model fails to load

# Sidebar Inputs
prompt = st.text_area("Enter Storyboard Prompt", "A sci-fi spaceship flying through a nebula")
num_frames = st.slider("Number of Frames", min_value=1, max_value=5, value=3)
image_quality = st.radio("Image Quality", ["Low", "Medium", "High"], index=1)
black_and_white = st.checkbox("Black and White")

# Map image quality to inference steps (more steps = better quality, but slower)
quality_to_steps = {"Low": 20, "Medium": 50, "High": 50}  # Reduced High to 50 for testing
num_inference_steps = quality_to_steps[image_quality]

# Output Directory
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

# Function to generate a single image
def generate_image(prompt, black_and_white, frame_idx, total_frames, progress_placeholder):
    try:
        image_prompt = f"{prompt}, storyboard frame"
        if black_and_white:
            image_prompt += ", black and white"

        # Update progress
        progress_placeholder.progress((frame_idx + 1) / total_frames, f"Generating frame {frame_idx + 1}/{total_frames}...")

        # Generate image using the pipeline
        image = model(
            image_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5
        ).images[0]

        # Save the image
        image_path = os.path.join(output_dir, f"frame_{frame_idx}_{prompt.replace(' ', '_')}.png")
        image.save(image_path)
        return image_path
    except Exception as e:
        st.error(f"Error generating frame {frame_idx + 1}: {str(e)}")
        return None

# Function to run image generation in a separate thread
def generate_in_thread(prompt, black_and_white, frame_idx, total_frames, storyboard_images, progress_placeholder):
    img_path = generate_image(prompt, black_and_white, frame_idx, total_frames, progress_placeholder)
    if img_path:
        storyboard_images[frame_idx] = img_path

# Generate and Display Images
if st.button("Generate Storyboard"):
    if not prompt.strip():
        st.error("Please enter a valid prompt.")
    else:
        # Initialize progress bar
        progress_placeholder = st.progress(0, "Starting generation...")

        # List to store image paths (using a list to maintain order)
        storyboard_images = [None] * num_frames
        threads = []

        # Start a thread for each frame
        for i in range(num_frames):
            thread = threading.Thread(
                target=generate_in_thread,
                args=(prompt, black_and_white, i, num_frames, storyboard_images, progress_placeholder)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check if all images were generated successfully
        if all(img_path is not None for img_path in storyboard_images):
            st.subheader("Generated Storyboard")
            cols = st.columns(len(storyboard_images))
            for col, img_path in zip(cols, storyboard_images):
                col.image(Image.open(img_path), use_column_width=True)
        else:
            st.error("Some frames failed to generate. Please check the errors above and try again.")

        # Clear the progress bar
        progress_placeholder.empty()
