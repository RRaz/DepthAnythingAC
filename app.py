import gradio as gr
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import tempfile
import io

from depth_anything.dpt import DepthAnything_AC


def normalize_depth(disparity_tensor):
    """Standard normalization method to convert disparity to depth"""
    eps = 1e-6
    disparity_min = disparity_tensor.min()
    disparity_max = disparity_tensor.max()
    normalized_disparity = (disparity_tensor - disparity_min) / (disparity_max - disparity_min + eps)
    return normalized_disparity


def load_model(model_path='checkpoints/depth_anything_AC_vits.pth', encoder='vits'):
    """Load trained depth estimation model"""
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'version': 'v2'},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'version': 'v2'},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'version': 'v2'}
    }
    
    model = DepthAnything_AC(model_configs[encoder])
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
    else:
        print(f"Warning: Model file {model_path} not found")
        
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    
    return model


def preprocess_image(image, target_size=518):
    """Preprocess input image"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        pass
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    
    image = image.astype(np.float32) / 255.0
    h, w = image.shape[:2]
    scale = target_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    new_h = ((new_h + 13) // 14) * 14
    new_w = ((new_w + 13) // 14) * 14
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    image = torch.from_numpy(image.transpose(2, 0, 1)).float()
    image = image.unsqueeze(0)
    
    return image, (h, w)


def postprocess_depth(depth_tensor, original_size):
    """Post-process depth map"""
    if depth_tensor.dim() == 3:
        depth_tensor = depth_tensor.unsqueeze(1)
    elif depth_tensor.dim() == 2:
        depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(1)
    
    h, w = original_size
    depth = F.interpolate(depth_tensor, size=(h, w), mode='bilinear', align_corners=True)
    depth = depth.squeeze().cpu().numpy()
    
    return depth


def create_colored_depth_map(depth, colormap='spectral'):
    """Create colored depth map"""
    if colormap == 'inferno':
        depth_colored = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    elif colormap == 'spectral':
        from matplotlib import cm
        spectral_cmap = cm.get_cmap('Spectral_r')
        depth_colored = (spectral_cmap(depth) * 255).astype(np.uint8)
        depth_colored = depth_colored[:, :, :3]
    else:
        depth_colored = (depth * 255).astype(np.uint8)
        depth_colored = np.stack([depth_colored] * 3, axis=2)
    
    return depth_colored


print("Loading model...")
model = load_model()
print("Model loaded successfully!")


def predict_depth(input_image, colormap_choice):
    """Main depth prediction function"""
    try:
        # Handle case when no image is provided
        if input_image is None:
            return None
            
        image_tensor, original_size = preprocess_image(input_image)
        
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        
        with torch.no_grad():
            prediction = model(image_tensor)
            disparity_tensor = prediction['out']
            depth_tensor = normalize_depth(disparity_tensor)
        
        depth = postprocess_depth(depth_tensor, original_size)
        
        depth_colored = create_colored_depth_map(depth, colormap_choice.lower())
        
        return Image.fromarray(depth_colored)
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None


def capture_and_predict(camera_image, colormap_choice):
    """Capture image from camera and predict depth"""
    return predict_depth(camera_image, colormap_choice)


with gr.Blocks(title="Depth Anything AC - Depth Estimation Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌊 Depth Anything AC - Depth Estimation Demo
    
    Upload an image or use your camera to generate corresponding depth maps! Different colors in the depth map represent different distances, allowing you to see the three-dimensional structure of the image.
    
    ## How to Use
    1. **Upload Mode**: Click the upload area to select an image file
    2. **Camera Mode**: Use your camera to capture a live image
    3. Choose your preferred colormap style
    4. Click the "Generate Depth Map" button
    5. View the results and download
    """)
    
    # Input controls row
    with gr.Row():
        input_source = gr.Radio(
            choices=["Upload Image", "Use Camera"],
            value="Upload Image",
            label="Input Source"
        )
        colormap_choice = gr.Dropdown(
            choices=["Spectral", "Inferno", "Gray"],
            value="Spectral",
            label="Colormap Style"
        )
        submit_btn = gr.Button(
            "🎯 Generate Depth Map",
            variant="primary",
            size="lg"
        )
    
    # Images row - perfectly aligned
    with gr.Row():
        with gr.Column(scale=1, min_width=400):
            # Upload image component
            upload_image = gr.Image(
                label="Input Image",
                type="pil",
                height=400,
                visible=True
            )
            
            # Camera component
            camera_image = gr.Image(
                label="Camera Input",
                type="pil", 
                sources=["webcam"],
                height=400,
                visible=False
            )
            
        with gr.Column(scale=1, min_width=400):
            output_image = gr.Image(
                label="Depth Map Result",
                type="pil",
                height=400
            )
            
            # Add download button to match the height of upload controls
            download_btn = gr.DownloadButton(
                label="📥 Download Depth Map",
                variant="secondary",
                size="sm",
                visible=False
            )
    
    # Function to switch between upload and camera input
    def switch_input_source(source):
        if source == "Upload Image":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)
    
    # Update visibility based on input source selection
    input_source.change(
        fn=switch_input_source,
        inputs=[input_source],
        outputs=[upload_image, camera_image]
    )
    
    # Function to handle both input sources
    def handle_prediction(input_source, upload_img, camera_img, colormap):
        if input_source == "Upload Image":
            result = predict_depth(upload_img, colormap)
        else:
            result = predict_depth(camera_img, colormap)
        
        # Show download button when depth map is generated
        if result is not None:
            # Save result to temporary file for download
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            result.save(temp_file.name)
            return result, gr.update(visible=True, value=temp_file.name)
        else:
            return None, gr.update(visible=False)
    
    # Examples section
    gr.Examples(
        examples=[
            ["toyset/1.png", "Spectral"],
            ["toyset/2.png", "Spectral"],
            ["toyset/good.png", "Spectral"],
        ] if os.path.exists("toyset") else [],
        inputs=[upload_image, colormap_choice],
        outputs=output_image,
        fn=predict_depth,
        cache_examples=False,
        label="Try these example images"
    )
    
    # Submit button click handler
    submit_btn.click(
        fn=handle_prediction,
        inputs=[input_source, upload_image, camera_image, colormap_choice],
        outputs=[output_image, download_btn],
        show_progress=True
    )
    
    gr.Markdown("""
    ## 📝 Color Map Descriptions
    - **Spectral**: Rainbow spectrum with distinct near-far contrast
    - **Inferno**: Flame spectrum with warm tones  
    - **Gray**: Classic grayscale depth representation
    
    ## 📷 Camera Tips
    - Make sure to allow camera access when prompted
    - Click the camera button to capture the current frame
    - The captured image will be used as input for depth estimation
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    ) 