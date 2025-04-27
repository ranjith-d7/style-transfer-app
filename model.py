# model.py
import tensorflow as tf
import tensorflow_hub as hub

# Load the TensorFlow Hub style transfer model
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
model = hub.load(hub_handle)

def apply_style_transfer(content_tensor, style_tensor):
    """
    Run the TF-Hub model to stylize content_tensor with style_tensor.
    Both inputs should be 4D tensors: [1, height, width, 3], float32, values [0,1].
    """
    outputs = model(content_tensor, style_tensor)
    stylized_image = outputs[0]
    return stylized_image


