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


with gr.Blocks(title="Depth Anything AC - Depth Estimation Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌊 Depth Anything AC - Depth Estimation Demo
    
    Upload an image and AI will generate the corresponding depth map! Different colors in the depth map represent different distances, allowing you to see the three-dimensional structure of the image.
    
    ## How to Use
    1. Click the upload area to select an image
    2. Choose your preferred colormap style
    3. Click the "Generate Depth Map" button
    4. View the results and download
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Upload Image",
                type="pil",
                height=400
            )
            
            colormap_choice = gr.Dropdown(
                choices=["Spectral", "Inferno", "Gray"],
                value="Spectral",
                label="Colormap"
            )
            
            submit_btn = gr.Button(
                "🎯 Generate Depth Map",
                variant="primary",
                size="lg"
            )
            
        with gr.Column():
            output_image = gr.Image(
                label="Depth Map Result",
                type="pil",
                height=400
            )
    
    gr.Examples(
        examples=[
            ["toyset/1.png", "Spectral"],
            ["toyset/2.png", "Spectral"],
            ["toyset/good.png", "Spectral"],
        ] if os.path.exists("toyset") else [],
        inputs=[input_image, colormap_choice],
        outputs=output_image,
        fn=predict_depth,
        cache_examples=False,
        label="Try these example images"
    )
    
    submit_btn.click(
        fn=predict_depth,
        inputs=[input_image, colormap_choice],
        outputs=output_image,
        show_progress=True
    )
    
    gr.Markdown("""
    ## 📝 Notes
    - **Spectral**: Rainbow spectrum with distinct near-far contrast
    - **Inferno**: Flame spectrum with warm tones
    - **Gray**: Grayscale with classic effect
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    ) 