import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


# Configuration

IMG_SIZE = (180, 180)
num_classes = 5
EPOCHS = 30


# Build a simple CNN

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
    layers.Dense(num_classes, activation='softmax')
])


# Model Summary

print("\n Model Summary:")
model.summary()


# Compile Model

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Train Model

print("\n Training model...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)


# Evaluate and Display Confusion Matrix

print("\n Generating confusion matrix...")

# Get true labels and predictions
y_true = np.concatenate([y for x, y in val_ds], axis=0)
y_pred = np.argmax(model.predict(val_ds), axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_ds.class_names)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

print("\n✅ Model training and evaluation complete!")
