import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------
DATA_DIR = "E:/Ai_ml_models/flowers"
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 20   # Usually fewer epochs needed for transfer learning
SEED = 42

# --------------------------------------------------------------
#  Load dataset
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
print(f"\n Loaded {num_classes} classes: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --------------------------------------------------------------
#  Data Augmentation block
# --------------------------------------------------------------
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
], name="data_augmentation")


#  Load Pretrained MobileNetV2 (Frozen Base)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,     
    weights='imagenet'
)
base_model.trainable = False   #  Freeze base model

#  Build Transfer Learning Model 

model = models.Sequential([
    data_augmentation,
    layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input), 
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])


# --------------------------------------------------------------
#  Compile & Train
# --------------------------------------------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n Training model with Frozen MobileNetV2 base...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# --------------------------------------------------------------
#  Plot Training vs Validation Accuracy
# --------------------------------------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Accuracy plot for MobileNetV2 Frozen Base')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
