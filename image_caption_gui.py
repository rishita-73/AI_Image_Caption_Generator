# image_caption_gui.py

import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image, ImageDraw, ImageFont
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

# Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Global variables
current_image_path = None
current_caption = ""

def generate_caption(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def upload_image():
    global current_image_path, current_caption
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
    if file_path:
        current_image_path = file_path
        img = Image.open(file_path)
        img = img.resize((300, 300))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        
        current_caption = generate_caption(file_path)
        caption_label.config(text="Caption: " + current_caption)

def save_image_with_caption():
    if not current_image_path or not current_caption:
        return

    # Open original image
    img = Image.open(current_image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", size=24)
    except:
        font = ImageFont.load_default()

    # Calculate position
    text_position = (10, img.height - 40)  # Near bottom left
    text_color = (255, 255, 255)  # White color

    # Draw text
    draw.text(text_position, current_caption, fill=text_color, font=font)

    # Save to new file
    save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
    if save_path:
        img.save(save_path)

# GUI Setup
root = tk.Tk()
root.title("AI Image Caption Generator")
root.geometry("500x650")

upload_btn = tk.Button(root, text="Upload Image", command=upload_image, font=("Arial", 14))
upload_btn.pack(pady=20)

panel = tk.Label(root)
panel.pack()

caption_label = tk.Label(root, text="", wraplength=400, font=("Arial", 12))
caption_label.pack(pady=20)

save_btn = tk.Button(root, text="Save Image with Caption", command=save_image_with_caption, font=("Arial", 14))
save_btn.pack(pady=10)

root.mainloop()
