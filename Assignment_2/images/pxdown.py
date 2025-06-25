from PIL import Image

# Load your image
image_path = "Assignment_2/images/barn2.png"
img = Image.open(image_path)

# Resize the image to 90x90 pixels
resized_img = img.resize((90, 90), Image.LANCZOS)

# Save the resized image (overwrite or save as a new file)
resized_img.save("Assignment_2/images/barn2.png")
print("✅ Image resized to 90x90 and saved.")
