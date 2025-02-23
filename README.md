# Collision Prediction 🚗💥

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-orange)](https://pytorch.org/)

Welcome to the **Collision Prediction** project! This repository contains a deep learning pipeline built with PyTorch that predicts collision events from video sequences. It leverages a ResNet-50 backbone for feature extraction and applies various data augmentations to boost performance.

---

## Overview ✨

In this project, we:
- **Process video data** with custom windowed sampling.
- **Augment the data** using techniques like random gamma adjustment, color jitter, rotation, and Gaussian noise.
- **Extract features** using a pretrained ResNet-50 model.
- **Train a collision prediction model** with a custom training loop.
- **Evaluate performance** using standard metrics, with checkpointing based on the F1 score.

---

## Dataset 📊

This project uses the [Nexar Collision Prediction dataset](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) from Hugging Face. 
Ensure you download and set up the dataset according to your system's paths.

---

## Model Architecture 🧠



---

## Installation 🛠️

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/Diana1800/collision_prediction.git
    cd collision_prediction
    ```

---

## Usage 🚀

1. **Prepare the Dataset:**
   - Download the dataset from [Hugging Face](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction).
   - Place the CSV file and video files in the appropriate directories (update paths in the code if needed).

2. **Run the Training Script:**
    ```bash
    python nexar_collision_prediction.py
    ```

The script will:
- Load and preprocess the video data.
- Initialize and display the model architecture.
- Train the model over 40 epochs.
- Validate and save the best model based on the F1 score.

---

## Training Parameters ⚙️

- **Epochs:** 40
- **Batch Size:** 8
- **Learning Rate:** 1e-4
- **Frames per Video:** 32

---

## Current Results 🏆

- Achieved an **80% F1 score** on the validation set.

---

## Future Improvements & Ideas 💡

- **Experiment with Different Pretrained Models:**  
  Try alternatives to ResNet-50 such as EfficientNet, DenseNet, or Vision Transformers to potentially improve feature extraction.
  
- **Advanced Data Augmentation:**  
  Incorporate augmentations that simulate varied real-world conditions such as night scenes, rain, fog, or motion blur to enrich the diversity of training samples.
  
- **Enhanced Data Handling Approaches:**  
  Explore different strategies for handling video data:
  - Use temporal attention mechanisms.
  - Experiment with 3D convolutional architectures.
  - Consider recurrent layers (e.g., LSTM or GRU) to better capture temporal dependencies.
  
- **Hyperparameter Tuning:**  
  Perform extensive tuning of hyperparameters such as learning rate, dropout rates, and batch size. Automated tools like Optuna can be helpful.
  
- **Ensemble Methods:**  
  Combine predictions from multiple models to increase robustness and overall performance.
  
- **Advanced Training Techniques:**  
  Utilize learning rate scheduling.

---

## Contributing 🤝

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

---

## License 📄

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Happy coding! 😄🚗💥
