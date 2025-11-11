# ==============================================================
# Programming Assignment: Flower Classification using Transfer Learning & Data Augmentation
# Part C — Model Development
# ==============================================================

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------
DATA_DIRECTORY = "E:/Ai_ml_models/flowers"    # or your subset folder, e.g. subset_100
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42

# --------------------------------------------------------------
# 1️⃣ Load dataset (train/validation split)
# --------------------------------------------------------------
print("📁 Loading dataset from:", DATA_DIRECTORY)



# Load training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIRECTORY,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIRECTORY,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# ✅ Save class names before prefetching (important fix)


class_names = train_ds.class_names
num_classes = len(class_names)

print(f"\n✅ Loaded {num_classes} classes: {class_names}")

# --------------------------------------------------------------
# 2️⃣ Optimize dataset pipeline
# --------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


#  Building  a simple CNN 

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=IMG_SIZE + (3,)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')

    #num_classes = len(class_names)  
    
])

model.summary()

# --------------------------------------------------------------
# 4️⃣ Model Summary
# --------------------------------------------------------------
print("\n🧠 Model Summary:")
model.summary()

# --------------------------------------------------------------
# 5️⃣ Compile Model
# --------------------------------------------------------------


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # integer labels
    metrics=['accuracy']
)




# Tranning  Model 
print("\n Training model...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Plot Training vs Validation Accuracy curve
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

