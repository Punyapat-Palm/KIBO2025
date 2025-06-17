import tensorflow as tf
import numpy as np
import os

# Parameters
# data_dir = r'/home/palm/kibo/keeped/objectDT/dataset'
data_dir = r'/home/palm/kibo/keeped/objectDT/cropped_dataset'
x = 96
img_size = (x, x)
batch_size = 32
seed = 42
best_overall_epoch = 116  # Use your best validated epoch

# Load dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    image_size=img_size,
    label_mode='categorical',
    shuffle=True,
    seed=seed
)

# ✅ Get class names BEFORE transformations
class_names = dataset.class_names
num_classes = len(class_names)

# Transform
AUTOTUNE = tf.data.AUTOTUNE
dataset = dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

# Build model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=img_size + (3,))
x = tf.keras.layers.Rescaling(1./255)(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(dataset, epochs=best_overall_epoch)

# ✅ Save in SavedModel format for TFLite
model.export("final_icon_classifier_model")  # << This is correct for Keras 3

# ✅ Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("final_icon_classifier_model")
tflite_model = converter.convert()

with open("/home/palm/kibo/keeped/objectDT/final_icon_classifier_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model trained and exported as 'final_icon_classifier_model.tflite'")
