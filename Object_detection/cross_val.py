import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2

# Set seed for reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Paths
data_dir = r'/home/palm/kibo/keeped/objectDT/cropped_dataset'
img_size = (96, 96)
num_folds = 11
epochs = 120
batch_size = 32

# Step 1: Load all data into memory
image_paths = []
labels = []
class_names = sorted(os.listdir(data_dir))

for i, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    for fname in os.listdir(class_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(class_dir, fname))
            labels.append(i)

# Convert labels to numpy array
labels = np.array(labels)
num_classes = len(class_names)

# Load images
def load_image(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=img_size)
    img = tf.keras.preprocessing.image.img_to_array(img)
    return img / 255.0

images = np.array([load_image(p) for p in image_paths])
labels_cat = to_categorical(labels, num_classes)

# Step 2: K-Fold Cross Validation
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
epoch_val_acc = np.zeros((num_folds, epochs))

for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
    print(f"\n📂 Fold {fold+1}/{num_folds} ({class_names[fold]})")
    x_train, x_val = images[train_idx], images[val_idx]
    y_train, y_val = labels_cat[train_idx], labels_cat[val_idx]

    # Define model
    base_model = MobileNetV2(
        input_shape=img_size + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=img_size + (3,)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=0
    )

    epoch_val_acc[fold] = history.history['val_accuracy']
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    best_acc = np.max(history.history['val_accuracy'])
    print(f"✅ Best epoch for fold {fold+1}: {best_epoch} (val_acc={best_acc:.4f})")

# Step 3: Determine the best epoch across folds
mean_acc_per_epoch = np.mean(epoch_val_acc, axis=0)
best_overall_epoch = np.argmax(mean_acc_per_epoch) + 1

print("\n📊 Cross-validation complete.")
print(f"🔥 Best average epoch: {best_overall_epoch} (avg val acc={mean_acc_per_epoch[best_overall_epoch-1]:.4f})")

# Optional: Save results
np.save("/home/palm/kibo/keeped/objectDT/epoch_val_acc.npy", epoch_val_acc)
