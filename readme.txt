# Project: Flower Classification with CNNs and Transfer Learning

# Core Deep Learning Libraries
tensorflow==2.16.1
numpy==1.26.4
pandas==2.2.2

# Visualization and Evaluation
matplotlib==3.9.0
seaborn==0.13.2
scikit-learn==1.5.0

# Image Handling and Augmentation
Pillow==10.3.0

# Utilities
pickle5==0.0.11
jupyter==1.0.0

# Optional: for GPU users
# tensorflow-gpu==2.16.1


# description 


| CNN |                          Basic 3-layer convolutional neural network |
| CNN + L2 |                     Adds L2 weight regularization |
| CNN + Dropout + L2 |           Combines dropout and L2 for better generalization |
| MobileNetV2 (Frozen Base) |    Uses pretrained MobileNetV2 with base layers frozen |
| MobileNetV2 (Fine-Tuned) |     Unfreezes top layers of MobileNetV2 for fine-tuning |


local directory <E:\Ai_ml_models>

data set directory
flowers/
├── daisy/
├── dandelion/
├── roses/
├── sunflowers/
└── tulips/