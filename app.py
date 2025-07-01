import gradio as gr
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import tempfile
import io
from tqdm import tqdm

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


def preprocess_image_from_array(image_array, target_size=518):
    """Preprocess input image from numpy array (for video frames)"""
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Convert BGR to RGB if needed
        image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        image = image_array.astype(np.float32) / 255.0
    
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


def is_video_file(filepath):
    """Check if the given file is a video file based on its extension"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    _, ext = os.path.splitext(filepath.lower())
    return ext in video_extensions


print("Loading model...")
model = load_model()
print("Model loaded successfully!")


def predict_depth(input_image, colormap_choice):
    """Main depth prediction function for images"""
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
        print(f"Error during image inference: {str(e)}")
        return None


def predict_video_depth(input_video, colormap_choice, progress=gr.Progress()):
    """Main depth prediction function for videos"""
    if input_video is None:
        return None
        
    try:
        print(f"Starting video processing: {input_video}")
        
        # Open video file
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print(f"Error: Cannot open video file: {input_video}")
            return None
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {total_frames} frames, {input_fps} FPS, {width}x{height}")
        
        # Create temporary output video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        # Set video encoder
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, input_fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Cannot create output video: {output_path}")
            cap.release()
            return None
        
        frame_count = 0
        
        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            progress_percent = frame_count / total_frames
            progress(progress_percent, desc=f"Processing frame {frame_count}/{total_frames}")
            
            try:
                # Preprocess current frame
                image_tensor, original_size = preprocess_image_from_array(frame)
                if torch.cuda.is_available():
                    image_tensor = image_tensor.cuda()
                
                # Perform depth estimation
                with torch.no_grad():
                    prediction = model(image_tensor)
                    disparity_tensor = prediction['out']
                    depth_tensor = normalize_depth(disparity_tensor)
                
                # Postprocess depth map
                depth = postprocess_depth(depth_tensor, original_size)
                
                # Handle failed processing
                if depth is None:
                    if depth_tensor.dim() == 1:
                        h, w = original_size
                        expected_size = h * w
                        if depth_tensor.shape[0] == expected_size:
                            depth_tensor = depth_tensor.view(1, 1, h, w)
                        else:
                            import math
                            side_length = int(math.sqrt(depth_tensor.shape[0]))
                            if side_length * side_length == depth_tensor.shape[0]:
                                depth_tensor = depth_tensor.view(1, 1, side_length, side_length)
                    depth = postprocess_depth(depth_tensor, original_size)
                
                # Generate colored depth map
                if depth is None:
                    print(f"Warning: Failed to process frame {frame_count}, using black frame")
                    depth_frame = np.zeros((height, width, 3), dtype=np.uint8)
                else:
                    if colormap_choice.lower() == 'inferno':
                        depth_frame = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                    elif colormap_choice.lower() == 'spectral':
                        from matplotlib import cm
                        spectral_cmap = cm.get_cmap('Spectral_r')
                        depth_frame = (spectral_cmap(depth) * 255).astype(np.uint8)
                        depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_RGBA2BGR)
                    else:  # gray
                        depth_frame = (depth * 255).astype(np.uint8)
                        depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)
                
                # Write to output video
                out.write(depth_frame)
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {str(e)}")
                # Write black frame
                black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                out.write(black_frame)
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"Video processing completed! Output saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error during video inference: {str(e)}")
        return None


with gr.Blocks(title="Depth Anything AC - Depth Estimation Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌊 Depth Anything AC - Depth Estimation Demo
    
    Upload an image or video and AI will generate the corresponding depth map! Different colors in the depth map represent different distances, allowing you to see the three-dimensional structure of the scene.
    
    ## How to Use
    1. Choose image or video tab
    2. Upload your file
    3. Select your preferred colormap style
    4. Click the "Generate Depth Map" button
    5. View results and download
    """)
    
    with gr.Tabs():
        # Image processing tab
        with gr.TabItem("📷 Image Depth Estimation"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        label="Upload Image",
                        type="pil",
                        height=400
                    )
                    
                    image_colormap_choice = gr.Dropdown(
                        choices=["Spectral", "Inferno", "Gray"],
                        value="Spectral",
                        label="Colormap"
                    )
                    
                    image_submit_btn = gr.Button(
                        "🎯 Generate Image Depth Map",
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
                inputs=[input_image, image_colormap_choice],
                outputs=output_image,
                fn=predict_depth,
                cache_examples=False,
                label="Try these example images"
            )
        
        # Video processing tab
        with gr.TabItem("🎬 Video Depth Estimation"):
            with gr.Row():
                with gr.Column():
                    input_video = gr.Video(
                        label="Upload Video",
                        height=400
                    )
                    
                    video_colormap_choice = gr.Dropdown(
                        choices=["Spectral", "Inferno", "Gray"],
                        value="Spectral",
                        label="Colormap"
                    )
                    
                    video_submit_btn = gr.Button(
                        "🎯 Generate Video Depth Map",
                        variant="primary",
                        size="lg"
                    )
                    
                with gr.Column():
                    output_video = gr.Video(
                        label="Depth Map Video Result",
                        height=400
                    )
            
            gr.Examples(
                examples=[
                    ["toyset/fog.mp4", "Spectral"],
                    ["toyset/snow.mp4", "Spectral"],
                ] if os.path.exists("toyset/fog.mp4") and os.path.exists("toyset/snow.mp4") else [],
                inputs=[input_video, video_colormap_choice],
                outputs=output_video,
                fn=predict_video_depth,
                cache_examples=False,
                label="Try these example videos"
            )
    
    # Event bindings
    image_submit_btn.click(
        fn=predict_depth,
        inputs=[input_image, image_colormap_choice],
        outputs=output_image,
        show_progress=True
    )
    
    video_submit_btn.click(
        fn=predict_video_depth,
        inputs=[input_video, video_colormap_choice],
        outputs=output_video,
        show_progress=True
    )
    
    gr.Markdown("""
    ## 📝 Notes
    - **Spectral**: Rainbow spectrum with distinct near-far contrast
    - **Inferno**: Flame spectrum with warm tones
    - **Gray**: Grayscale with classic effect
    
    ## 💡 Tips
    - Image processing is fast, suitable for quick preview of single images
    - Video processing may take longer time, please be patient
    - GPU is recommended for faster processing speed
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    ) 