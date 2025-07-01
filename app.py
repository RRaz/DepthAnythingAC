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


def is_video_file(filepath):
    """Check if the given file is a video file based on its extension"""
    if filepath is None:
        return False
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    _, ext = os.path.splitext(filepath.lower())
    return ext in video_extensions


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
    """Preprocess input image (supports both PIL Image and numpy array)"""
    if isinstance(image, str):
        raw_image = cv2.imread(image)
        if raw_image is None:
            raise ValueError(f"Cannot read image: {image}")
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    elif isinstance(image, Image.Image):
        image = np.array(image)
        image = image.astype(np.float32) / 255.0
    elif isinstance(image, np.ndarray):
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        pass
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    
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


def process_video(video_path, colormap_choice, progress=gr.Progress()):
    """Process video file for depth estimation"""
    try:
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {total_frames} frames, {input_fps} FPS, {width}x{height}")
        
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = temp_output.name
        temp_output.close()
        
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, input_fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            raise ValueError("Cannot create output video file")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    image_tensor, original_size = preprocess_image(frame_rgb)
                    
                    if torch.cuda.is_available():
                        image_tensor = image_tensor.cuda()
                    
                    with torch.no_grad():
                        prediction = model(image_tensor)
                        disparity_tensor = prediction['out']
                        depth_tensor = normalize_depth(disparity_tensor)
                    
                    depth = postprocess_depth(depth_tensor, original_size)
                    
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
                    
                    if depth is None:
                        print(f"Warning: Frame {frame_count} processing failed, using black frame")
                        depth_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    else:
                        if colormap_choice.lower() == 'inferno':
                            depth_frame = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                        elif colormap_choice.lower() == 'spectral':
                            from matplotlib import cm
                            spectral_cmap = cm.get_cmap('Spectral_r')
                            depth_frame = (spectral_cmap(depth) * 255).astype(np.uint8)
                            depth_frame = depth_frame[:, :, :3]
                            depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_RGB2BGR)
                        else:
                            depth_frame = (depth * 255).astype(np.uint8)
                            depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)
                    
                    out.write(depth_frame)
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {str(e)}")
                    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    out.write(black_frame)
                
                progress((frame_count / total_frames), f"Processing progress: {frame_count}/{total_frames} frames")
                
        except Exception as e:
            print(f"Unexpected error during video processing: {str(e)}")
        finally:
            cap.release()
            out.release()
        
        print(f"Video processing completed! Output saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Video processing failed: {str(e)}")
        return None


print("Loading model...")
model = load_model()
print("Model loaded successfully!")


def predict_depth(input_file, colormap_choice):
    """Main depth prediction function for both images and videos"""
    try:
        if input_file is None:
            return None, gr.update(visible=False)
            
        if is_video_file(input_file):
            output_path = process_video(input_file, colormap_choice)
            if output_path:
                return output_path, gr.update(visible=True, value=output_path)
            else:
                return None, gr.update(visible=False)
        else:
            if isinstance(input_file, str):
                input_image = Image.open(input_file)
            else:
                input_image = input_file
                
            image_tensor, original_size = preprocess_image(input_image)
            
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
            
            with torch.no_grad():
                prediction = model(image_tensor)
                disparity_tensor = prediction['out']
                depth_tensor = normalize_depth(disparity_tensor)
            
            depth = postprocess_depth(depth_tensor, original_size)
            depth_colored = create_colored_depth_map(depth, colormap_choice.lower())
            
            result = Image.fromarray(depth_colored)
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            result.save(temp_file.name)
            
            return result, gr.update(visible=True, value=temp_file.name)
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None


def capture_and_predict(camera_image, colormap_choice):
    """Capture image from camera and predict depth"""
    return predict_depth(camera_image, colormap_choice)


with gr.Blocks(title="Depth Anything AC - Depth Estimation Demo", theme=gr.themes.Soft(), css="""
    .image-container { 
        display: flex !important; 
        align-items: flex-start !important; 
        justify-content: center !important; 
    }
    .gradio-image { 
        vertical-align: top !important; 
    }
""") as demo:
    gr.Markdown("""
    # 🌊 Depth Anything AC - Depth Estimation Demo
    
    Upload an image or use your camera to generate corresponding depth maps! Different colors in the depth map represent different distances, allowing you to see the three-dimensional structure of the image.
    
    ## How to Use
    1. **Upload Mode**: Click the upload area to select an image or video file
    2. **Camera Mode**: Use your camera to capture a live image
    3. Choose your preferred colormap style
    4. Click the "Generate Depth Map" button
    5. View the results and download
    """)
    
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
    
    with gr.Row():
        gr.HTML("<h3 style='text-align: center; margin: 10px;'>📷 Input Image</h3>")
        gr.HTML("<h3 style='text-align: center; margin: 10px;'>🌊 Depth Map Result</h3>")
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            upload_file = gr.File(
                file_types=["image", "video"],
                height=450,
                visible=True,
                show_label=False,
                container=False,
                label="Upload Image or Video"
            )
            
            # Camera component
            camera_image = gr.Image(
                type="pil", 
                sources=["webcam"],
                height=450,
                visible=False,
                show_label=False,
                container=False
            )
            
        with gr.Column(scale=1):
            output_file = gr.File(
                height=450,
                show_label=False,
                container=False,
                visible=False
            )
            
            output_image = gr.Image(
                type="pil",
                height=450,
                show_label=False,
                container=False,
                visible=True
            )
            
            download_btn = gr.DownloadButton(
                label="📥 Download Result",
                variant="secondary",
                size="sm",
                visible=False
            )
    
    def switch_input_source(source):
        if source == "Upload Image":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)
    
    input_source.change(
        fn=switch_input_source,
        inputs=[input_source],
        outputs=[upload_file, camera_image]
    )
    
    def handle_prediction(input_source, upload_file_path, camera_img, colormap):
        if input_source == "Upload Image":
            if upload_file_path is None:
                return None, None, gr.update(visible=False), gr.update(visible=False)
            
            result, download_update = predict_depth(upload_file_path, colormap)
            
            if isinstance(result, str) and is_video_file(result):
                return None, result, gr.update(visible=False), download_update
            else:
                return result, None, gr.update(visible=True), download_update
        else:
            result, download_update = predict_depth(camera_img, colormap)
            return result, None, gr.update(visible=True), download_update
    
    example_files = []
    if os.path.exists("toyset"):
        for img_file in ["1.png", "2.png", "good.png"]:
            if os.path.exists(f"toyset/{img_file}"):
                example_files.append([f"toyset/{img_file}", "Spectral"])
        
        for vid_file in ["fog_2_processed_1s-6s_1.0x.mp4", "snow_processed_1s-6s_1.0x.mp4"]:
            if os.path.exists(f"toyset/{vid_file}"):
                example_files.append([f"toyset/{vid_file}", "Spectral"])
    
    if example_files:
        gr.Examples(
            examples=example_files,
            inputs=[upload_file, colormap_choice],
            outputs=[output_image, output_file],
            fn=lambda file_path, colormap: predict_depth(file_path, colormap),
            cache_examples=False,
            label="Try these example files"
        )
    
    submit_btn.click(
        fn=handle_prediction,
        inputs=[input_source, upload_file, camera_image, colormap_choice],
        outputs=[output_image, output_file, output_image, download_btn],
        show_progress=True
    )
    
    gr.Markdown("""
    ## 📝 Colormap Description
    - **Spectral**: Rainbow spectrum, with clear contrast between near and far
    - **Inferno**: Fire spectrum, warm tones
    - **Gray**: Classic grayscale depth representation
    
    ## 📷 Camera Usage Tips
    - Ensure camera access is allowed when prompted
    - Click the camera button to capture the current frame
    - The captured image will be used as input for depth estimation
    
    ## 🎬 Video Processing Tips
    - Supports multiple video formats (MP4, AVI, MOV, etc.)
    - Video processing may take some time, please be patient
    - Processing progress will be displayed in real-time
    - The output video will maintain the same frame rate as the input
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    ) 