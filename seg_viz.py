import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import torch
import torch.nn.functional as F

def preprocess_image_and_mask(image, seg_source):
    image = image.squeeze(0).cpu()  # Remove batch dimension
    mask = seg_source["segmentation"]
    
    if mask.shape[2:] != image.shape[1:]:
        mask = F.interpolate(mask, size=image.shape[1:], mode='bilinear')
    
    return image, mask

def apply_hard_edges(mask):
    return (torch.max(mask, dim=1, keepdim=True)[0] == mask).float()

def create_color_mask(mask, colormap='gist_rainbow'):
    colormap = plt.get_cmap(colormap)
    num_segments = mask.shape[1]
    mask = mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
    color_mask = np.zeros((256, 256, 3), dtype=np.float32)
    patches = []
    
    for i in range(num_segments):
        color = np.array(colormap((i - 1) / (num_segments - 1)))[:3] if i != 0 else np.array((0, 0, 0))
        patches.append(mpatches.Patch(color=color, label=str(i)))
        color_mask += mask[..., i:(i+1)] * color.reshape(1, 1, 3)
    
    return color_mask, patches

def overlay_mask(image, color_mask, alpha=0.6):
    image_np = image.permute(1, 2, 0).numpy()
    if image_np.max() > 1:
        image_np = image_np / 255.0
    
    overlaid_image = image_np * (1 - alpha) + color_mask * alpha
    return np.clip(overlaid_image, 0, 1)

def create_figure(overlaid_image, patches):
    fig = Figure(figsize=(10, 10))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    
    ax.imshow(overlaid_image)
    ax.axis('off')
    ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')
    
    fig.tight_layout(pad=0)
    return fig, canvas

def render_figure_to_tensor(canvas):
    canvas.draw()
    result_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    result_image = result_image.reshape(canvas.get_width_height()[::-1] + (3,))
    result_tensor = torch.from_numpy(result_image).unsqueeze(0).float() / 255.0
    return result_tensor

def visualize_frame(image, seg_source, hard_edges=True, colormap='gist_rainbow', alpha=0.6):
    image, mask = preprocess_image_and_mask(image, seg_source)
    
    if hard_edges:
        mask = apply_hard_edges(mask)
    
    color_mask, patches = create_color_mask(mask, colormap)
    overlaid_image = overlay_mask(image, color_mask, alpha)
    fig, canvas = create_figure(overlaid_image, patches)
    result_tensor = render_figure_to_tensor(canvas)
    
    return result_tensor

# def visualize_frame(image, seg_source, hard_edges=True, colormap='gist_rainbow', alpha=0.6):
#     # Ensure image is the right shape and on CPU
#     image = image.squeeze(0).cpu()  # Remove batch dimension
    
#     # Extract segmentation from seg_source dictionary
#     mask = seg_source["segmentation"]
    
#     # Interpolate mask to match image size if necessary
#     if mask.shape[2:] != image.shape[1:]:
#         mask = F.interpolate(mask, size=image.shape[1:], mode='bilinear')
    
#     if hard_edges:
#         mask = (torch.max(mask, dim=1, keepdim=True)[0] == mask).float()
    
#     colormap = plt.get_cmap(colormap)
#     num_segments = mask.shape[1]
#     mask = mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
#     color_mask = np.zeros((256, 256, 3), dtype=np.float32)
#     patches = []
#     for i in range(num_segments):
#         if i != 0:
#             color = np.array(colormap((i - 1) / (num_segments - 1)))[:3]
#         else:
#             color = np.array((0, 0, 0))
#         patches.append(mpatches.Patch(color=color, label=str(i)))
#         color_mask += mask[..., i:(i+1)] * color.reshape(1, 1, 3)
    
#     # Convert image to numpy and normalize to [0, 1] range if necessary
#     image_np = image.permute(1, 2, 0).numpy()
#     if image_np.max() > 1:
#         image_np = image_np / 255.0
    
#     # Overlay the color mask on the original image
#     overlaid_image = image_np * (1 - alpha) + color_mask * alpha
    
#     # Ensure all values are in [0, 1] range
#     overlaid_image = np.clip(overlaid_image, 0, 1)
    
#     # Create a figure and axis
#     fig = Figure(figsize=(10, 10))
#     canvas = FigureCanvasAgg(fig)
#     ax = fig.add_subplot(111)
    
#     # Display the image
#     ax.imshow(overlaid_image)
#     ax.axis('off')
    
#     # Add legend
#     ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')
    
#     # Render the figure to numpy array
#     fig.tight_layout(pad=0)
#     canvas.draw()
#     result_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
#     result_image = result_image.reshape(canvas.get_width_height()[::-1] + (3,))
    
#     # Convert to torch tensor and add batch dimension to get [1, 256, 256, 3]
#     result_tensor = torch.from_numpy(result_image).unsqueeze(0).float() / 255.0
    
#     return result_tensor