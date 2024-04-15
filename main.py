import gradio as gr
import cv2
import numpy as np
from imagesFunctions import *


# Define preprocessing function choices (must be before using in Dropdown)
preprocessing_functions = [
    ("Grayscale", grayscale),
    ("Blur", blur),
    ("Edge Detection", edge_detection),
    ("Invert Colors", invert_colors),
    ("Threshold", threshold),
    ("Gray Level Transform", gray_level_transform),
    ("Negative Transform", negative_transform),
    ("Log Transform", log_transform),
    ("Power Law Transform", power_law_transform),
    ("Contrast Stretching", contrast_stretching),
    ("intensity slicing", intensity_slicing),
    ("histogram equalization", histogram_equalization),
    ("mean filter", mean_filter),
    ("gaussian filter", gaussian_filter),
    ("sobel filter", sobel_filter),
    ("laplacian filter", laplacian_filter),
    ("min max filter", min_max_filter), 
    ("median filter", median_filter),
]

input_image = gr.components.Image(label="Upload Image")
function_selector = gr.components.Dropdown(choices=[func[0] for func in preprocessing_functions], label="Select Preprocessing Function")

# Define slider for alpha value
alpha_slider = gr.components.Slider(minimum=-100, maximum=100, label="alpha")
alpha_slider.default = 0  # Set default value for alpha

# Define slider for beta value
beta_slider = gr.components.Slider(minimum=0.1, maximum=3.0, label="beta")
beta_slider.default = 1.0  # Set default value for beta

# Define slider for c_log value
c_log_slider = gr.components.Slider(minimum=0.1, maximum=3.0, label="c_log")
c_log_slider.default = 1.0  # Set default value for c_log

# Define slider for gamma value
gamma_slider = gr.components.Slider(minimum=0.1, maximum=3.0, label="gamma")
gamma_slider.default = 1.0  # Set default value for gamma

# Define slider for slicing_threshold value
slicing_threshold_slider = gr.components.Slider(minimum=0, maximum=255, label="slicing threshold")
slicing_threshold_slider.default = 125.0  # Set default value for slicing_threshold

# Define slider for kernel size value
kernel_size_slider = gr.components.Slider(minimum=2, maximum=5, label="kernel size")
kernel_size_slider.default = 3  # Set default value for kernel size

# Define slider for kernel size value
sigma_slider = gr.components.Slider(minimum=2, maximum=5, label="sigma")
sigma_slider.default = 1  # Set default value for kernel size

def apply_preprocessing(image, selected_function, alpha, beta, c_log, gamma, slicing_threshold, kernel_size, sigma):
    # Find the actual function based on the user-friendly name
    selected_function_obj = None
    for func_name, func_obj in preprocessing_functions:
        if func_name == selected_function:
            selected_function_obj = func_obj
            break
    if selected_function_obj is None:
        raise ValueError("Selected function not found.")
    # For gray level transformation, pass beta and gamma values
    if selected_function == "Gray Level Transform":
        processed_image = selected_function_obj(image, alpha=alpha, beta=beta)
    elif selected_function == "Log Transform":
        processed_image = selected_function_obj(image, c=c_log)    
    elif selected_function == "Power Law Transform":
        processed_image = selected_function_obj(image, gamma=gamma)     
    elif selected_function == "intensity slicing":
        processed_image = selected_function_obj(image, threshold=slicing_threshold)
    elif selected_function == "mean filter":
        processed_image = selected_function_obj(image, kernel_size=kernel_size)
    elif selected_function == "gaussian filter":
        processed_image = selected_function_obj(image, kernel_size=kernel_size, sigma=sigma)  
    elif selected_function == "gaussian filter":
        processed_image = selected_function_obj(image, kernel_size=kernel_size, sigma=sigma)
    elif selected_function == "min max filter":
        processed_image = selected_function_obj(image, kernel_size=kernel_size)
    elif selected_function == "median filter":
        processed_image = selected_function_obj(image, kernel_size=kernel_size)                    
    else:
        print(selected_function_obj)
        processed_image = selected_function_obj(image)
    return processed_image

output_image = gr.components.Image(label="Processed Image")

# Create Gradio interface
gr.Interface(
    fn=apply_preprocessing,
    inputs=[input_image, function_selector, alpha_slider, beta_slider, c_log_slider, gamma_slider, slicing_threshold_slider, kernel_size_slider, sigma_slider],
    outputs=output_image,
    title="Elza3ama studio",
    description="Upload an image and select a preprocessing function."
).launch()