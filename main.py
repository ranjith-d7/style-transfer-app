# main.py
from fastapi import FastAPI
import tensorflow as tf

app = FastAPI()

from fastapi.staticfiles import StaticFiles

# Mount the "static" directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

from fastapi.responses import FileResponse, StreamingResponse

@app.get("/")
async def read_index():
    return FileResponse('templates/index.html', media_type='text/html')



from fastapi import UploadFile, File
from io import BytesIO
from PIL import Image
import numpy as np
from model import apply_style_transfer

@app.post("/style-transfer/")
async def style_transfer(content_image: UploadFile = File(...), 
                         style_image: UploadFile = File(...)):
    """
    Receive uploaded content and style images, apply style transfer,
    and return the stylized image as a binary response.
    """
    # Read the uploaded files as bytes
    content_bytes = await content_image.read()
    style_bytes = await style_image.read()

    # Load images with PIL and convert to RGB numpy arrays
    content_pil = Image.open(BytesIO(content_bytes)).convert('RGB')
    style_pil = Image.open(BytesIO(style_bytes)).convert('RGB')

    # Resize or preprocess images as needed (e.g., style to 256x256, content to desired size)
    style_pil = style_pil.resize((256, 256))
    # (Optionally resize content or keep original)

    # Convert to float32 numpy arrays and normalize to [0,1]
    content_array = np.array(content_pil) / 255.0
    style_array = np.array(style_pil) / 255.0

    # Add batch dimension: [1, height, width, 3]
    content_tensor = tf.convert_to_tensor(content_array, dtype=tf.float32)
    content_tensor = tf.expand_dims(content_tensor, 0)
    style_tensor = tf.convert_to_tensor(style_array, dtype=tf.float32)
    style_tensor = tf.expand_dims(style_tensor, 0)

    # Apply the style transfer model
    stylized_tensor = apply_style_transfer(content_tensor, style_tensor)

    # Convert the tensor back to an image (as bytes) for the response
    stylized_array = np.squeeze(stylized_tensor.numpy()) * 255
    stylized_image = Image.fromarray(np.uint8(stylized_array))
    buf = BytesIO()
    stylized_image.save(buf, format='PNG')
    buf.seek(0)

    # Return the stylized image as a StreamingResponse
    return StreamingResponse(buf, media_type="image/png")

