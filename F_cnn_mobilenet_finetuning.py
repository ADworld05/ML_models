import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------
DATA_DIR = "E:/Ai_ml_models/flowers"
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
INITIAL_EPOCHS = 15        # feature extraction
FINE_TUNE_EPOCHS = 5     # fine-tuning phase
SEED = 42

# --------------------------------------------------------------
#  Load dataset
# --------------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
class_names = train_ds.class_names
num_classes = len(class_names)
print(f" Loaded {num_classes} classes: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# --------------------------------------------------------------
#  Data Augmentation
# --------------------------------------------------------------
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2)
])

# --------------------------------------------------------------
#  Load Pretrained MobileNetV2
# --------------------------------------------------------------
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False
print("\n Base MobileNetV2 frozen")

# --------------------------------------------------------------
#  Build Model
# --------------------------------------------------------------
l2_reg = regularizers.l2(1e-4)

model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu', kernel_regularizer=l2_reg),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# --------------------------------------------------------------
#  Compile & Train (Feature Extraction)
# --------------------------------------------------------------
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nTraining classifier layers (feature extraction)...")
history = model.fit(train_ds, validation_data=val_ds, epochs=INITIAL_EPOCHS)

# --------------------------------------------------------------
#  Fine-Tuning Phase
# --------------------------------------------------------------
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 40  # unfreeze last 40 layers

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"\n🔧 Fine-tuning last {len(base_model.layers) - fine_tune_at} layers...")

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=history.epoch[-1],
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS
)

# --------------------------------------------------------------
#  Evaluate & Save
# --------------------------------------------------------------
train_acc = history_fine.history['accuracy'][-1] * 100
val_acc = history_fine.history['val_accuracy'][-1] * 100

model.save("mobilenetv2_finetuned_final.h5")
model_size_mb = os.path.getsize("mobilenetv2_finetuned_final.h5") / (1024 * 1024)

print("\n" + "="*55)
print(" FINAL FINE-TUNED MODEL RESULTS (MobileNetV2)")
print("="*55)
print(f"Training Accuracy:   {train_acc:.2f}%")
print(f"Validation Accuracy: {val_acc:.2f}%")
print(f"Model Size:          {model_size_mb:.2f} MB")
print("="*55)

# --------------------------------------------------------------
#  Accuracy Graph (Combined Phases)
# --------------------------------------------------------------
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
epochs_range = range(len(acc))

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, acc, label='Training Accuracy', linewidth=2)
plt.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2)
plt.axvline(x=INITIAL_EPOCHS - 1, color='gray', linestyle='--', label='Fine-tuning start')
plt.title('Fine-Tuned MobileNetV2 Accuracy Plot')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

