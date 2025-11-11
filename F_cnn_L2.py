import tensorflow as tf
from tensorflow.keras import layers, models, regularizers  
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------
DATA_DIR = "E:/Ai_ml_models/flowers"
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42

# --------------------------------------------------------------
# Load dataset (train/validation split)
# --------------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\n✅ Loaded {num_classes} classes: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# CNN with L2 regularization + Dropout

l2_reg = regularizers.l2(0.001)  

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=IMG_SIZE + (3,)),
    
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2_reg),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),   

    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2_reg),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),   

    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2_reg),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),   

    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=l2_reg),
    layers.Dropout(0.5),   
    layers.Dense(num_classes, activation='softmax')
])

# --------------------------------------------------------------
# 4️⃣ Compile & Train
# --------------------------------------------------------------
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# --------------------------------------------------------------
# 5️⃣ Plot accuracy curves
# --------------------------------------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training vs Validation Accuracy (L2 Regularization + Dropout)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()



