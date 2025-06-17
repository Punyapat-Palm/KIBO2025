import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=r"/home/palm/kibo/keeped/objectDT/final_icon_classifier_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Automatically get input shape from model
input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]  # Typically (1, H, W, C)
print(height, width)

# Class names
class_names = ['coin', 'compass', 'coral', 'crystal', 'diamond', 'emerald', 'fossil', 'key', 'letter', 'shell', 'treasure_box']

# Folder path containing test images
test_folder = "/home/palm/kibo/keeped/crop_output"
count = 0
score = 0
y = ["crystal", "shell", "shell", "shell", "shell", "treasure_box", "crystal", "compass", "crystal", "shell", "compass", "shell"]

# Loop through all image files in the folder
for filename in sorted(os.listdir(test_folder)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(test_folder, filename)

        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((width, height))
        input_data = np.expand_dims(img, axis=0)
        input_data = np.array(input_data, dtype=np.float32)  # or uint8 depending on model

        # If model expects uint8, convert
        if input_details[0]['dtype'] == np.uint8:
            input_data = np.array(input_data, dtype=np.uint8)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Get prediction
        predictions = output_data[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]

        try:
                    # Print result
            if class_names[predicted_class] == y[count]:
                score += 1
                print(f"[{filename}] Predicted: {class_names[predicted_class]}, Confidence: {confidence:.4f} ")
            else:
                print(f"[{filename}] Predicted: {class_names[predicted_class]} (Expected: {y[count]}), Confidence: {confidence:.4f} ❌")
        except IndexError:
            print(f"[{filename}] Predicted: {class_names[predicted_class]}, Confidence: {confidence:.4f}  (IndexError)")
        count += 1

# Final score
print(f"Score: {score}/{count} ({(score / count) * 100:.2f}%)")
