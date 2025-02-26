import os
import random
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from torchinfo import summary

from sklearn.model_selection import train_test_split

import mlflow
import mlflow.pytorch

import optuna

# custom training function
from DLfunctions import Train_model


# Augmentation
class RandomGamma(object):
    """Randomly adjust gamma of a PIL image."""
    def __init__(self, gamma_range=(0.8, 1.2), p=0.5):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            img_np = np.array(img).astype(np.float32) / 255.0
            img_np = np.power(img_np, gamma)
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(img_np)
        return img

class AddGaussianNoise(object):
    """Add Gaussian noise to a tensor."""
    def __init__(self, mean=0.0, std=0.05, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            noise = torch.randn(tensor.size()) * self.std + self.mean
            return tensor + noise
        return tensor

# Transform Pipelines
train_transform = T.Compose([
    T.Resize((128, 128)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    RandomGamma(gamma_range=(0.8, 1.2), p=0.5),
    T.RandomRotation(degrees=10),
    T.ToTensor(),
    AddGaussianNoise(mean=0.0, std=0.05, p=0.5),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
val_transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


# Dataset Classes with Windowed Sampling
class NexarCollisionDatasetTrain(Dataset):
    def __init__(self, data, video_dir, transform=train_transform, num_frames=16, file_ext='.mp4', window_duration=3):

        if isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            self.df = pd.read_csv(data)
        self.video_dir = video_dir
        self.transform = transform
        self.num_frames = num_frames
        self.file_ext = file_ext
        self.window_duration = window_duration
        self.fps = 30  # Adjust if needed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = f"{int(row['id']):05d}"
        target = int(row['target'])
        time_of_alert = row['time_of_alert'] if (target == 1 and 'time_of_alert' in row) else None
        video_path = os.path.join(self.video_dir, video_id + self.file_ext)
        frames = self._load_video_frames(video_path, self.num_frames, target, time_of_alert, fps=self.fps, window_duration=self.window_duration)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames)
        return frames, torch.tensor([target], dtype=torch.float32)

    def _load_video_frames(self, video_path, num_frames, target, time_of_alert=None, fps=30, window_duration=3):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} not found.")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video file {video_path} cannot be opened.")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"Video {video_path} has no frames.")

        # Determine window start: for target==1, use time_of_alert; else, start at 0.
        if target == 1 and time_of_alert is not None:
            expected_start = int(float(time_of_alert) * fps)
        else:
            expected_start = 0
        start_index = min(expected_start, total_frames - 1)
        # Define window: fixed window_duration seconds
        end_index = min(start_index + int(window_duration * fps) - 1, total_frames - 1)
        available_frames = end_index - start_index + 1

        # Uniformly sample indices from the window
        if available_frames < num_frames:
            indices = np.linspace(start_index, end_index, available_frames, dtype=int)
        else:
            indices = np.linspace(start_index, end_index, num_frames, dtype=int)

        if target == 1 and time_of_alert is not None:
            if indices[0] != expected_start:
                print(f"Warning: Video {video_path} (target 1) expected to start at {expected_start} but starts at {indices[0]}.")
            # else:
            #     print(f"Video {video_path} (target 1) correctly starts at {indices[0]} from time_of_alert {time_of_alert}.")

        frames = []
        current_frame = 0
        retrieved = 0
        while retrieved < len(indices) and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(frame)
                retrieved += 1
            current_frame += 1
        cap.release()
        while len(frames) < num_frames:
            frames.append(frames[-1].copy())
        return frames

class NexarCollisionDatasetVal(Dataset):
    def __init__(self, data, video_dir, transform=val_transform, num_frames=16, file_ext='.mp4', window_duration=3):
        if isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            self.df = pd.read_csv(data)
        self.video_dir = video_dir
        self.transform = transform
        self.num_frames = num_frames
        self.file_ext = file_ext
        self.window_duration = window_duration
        self.fps = 30

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = f"{int(row['id']):05d}"
        target = int(row['target'])
        time_of_alert = row['time_of_alert'] if (target == 1 and 'time_of_alert' in row) else None
        video_path = os.path.join(self.video_dir, video_id + self.file_ext)
        frames = self._load_video_frames(video_path, self.num_frames, target, time_of_alert, fps=self.fps, window_duration=self.window_duration)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames)
        return frames, torch.tensor([target], dtype=torch.float32)

    def _load_video_frames(self, video_path, num_frames, target, time_of_alert=None, fps=30, window_duration=3):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} not found.")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video file {video_path} cannot be opened.")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"Video {video_path} has no frames.")

        if target == 1 and time_of_alert is not None:
            expected_start = int(float(time_of_alert) * fps)
        else:
            expected_start = 0
        start_index = min(expected_start, total_frames - 1)
        end_index = min(start_index + int(window_duration * fps) - 1, total_frames - 1)
        available_frames = end_index - start_index + 1

        if available_frames < num_frames:
            indices = np.linspace(start_index, end_index, available_frames, dtype=int)
        else:
            indices = np.linspace(start_index, end_index, num_frames, dtype=int)

        if target == 1 and time_of_alert is not None:
            if indices[0] != expected_start:
                print(f"Warning (VAL): Video {video_path} expected to start at {expected_start} but starts at {indices[0]}.")
            # else:
            #     print(f"Video {video_path} (VAL) correctly starts at {indices[0]} from time_of_alert {time_of_alert}.")

        frames = []
        current_frame = 0
        retrieved = 0
        while retrieved < len(indices) and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(frame)
                retrieved += 1
            current_frame += 1
        cap.release()
        while len(frames) < num_frames:
            frames.append(frames[-1].copy())
        return frames


# Model 
class CollisionPredictionModel(nn.Module):
    def __init__(self, num_frames=16, num_classes=1):
        super(CollisionPredictionModel, self).__init__()
        self.num_frames = num_frames
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)  # (B*T, 2048, 1, 1)
        features = features.view(B, T, -1)      # (B, T, 2048)
        pooled = features.mean(dim=1)           # (B, 2048)
        pooled = self.dropout(pooled)
        out = self.fc(pooled)                   # (B, num_classes)
        return out


# Data Splitting, DataLoaders, Model, and Training
train_csv_path = '/home/diana/Downloads/nexar-collision-prediction/train.csv'
train_video_dir = '/home/diana/Downloads/nexar-collision-prediction/train'

df_full = pd.read_csv(train_csv_path)
train_df, val_df = train_test_split(df_full, test_size=0.2, random_state=42, stratify=df_full['target'])

print("Train Class distribution (target counts):")
print(train_df['target'].value_counts())

num_epochs = 40
batch_size = 8
learning_rate = 1e-4
num_frames = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = NexarCollisionDatasetTrain(data=train_df,
                                            video_dir=train_video_dir,
                                            transform=train_transform,
                                            num_frames=num_frames,
                                            file_ext='.mp4',
                                            window_duration=3)
val_dataset = NexarCollisionDatasetVal(data=val_df,
                                        video_dir=train_video_dir,
                                        transform=val_transform,
                                        num_frames=num_frames,
                                        file_ext='.mp4',
                                        window_duration=3)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

model = CollisionPredictionModel(num_frames=num_frames, num_classes=1).to(device)

dummy_input = torch.randn(batch_size, num_frames, 3, 128, 128, device=device)
summary(model, input_data=dummy_input, device=device)


#Training

#MLflow 
mlflow.set_tracking_uri("http://localhost:5000")  
experiment_name = "Experiment"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id

mlflow.set_experiment(experiment_name) 

# Optuna 
def objective(trial):
    # hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    num_frames = trial.suggest_int("num_frames", 40, 70)

    criterion = nn.BCEWithLogitsLoss()
    model = CollisionPredictionModel()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = None  

    with mlflow.start_run(experiment_id=experiment_id, nested=True):  
        print(f"Starting Optuna Trial {trial.number}...")

        # Log hyperparameters
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_frames", num_frames)

        # Train the model
        best_model, history = Train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=10, 
            device=device,
            is_binary=True,
            save_metric='f1'
        )

        # Get best validation F1 score
        best_val_f1 = max(history["val_f1"]) if "val_f1" in history else 0
        mlflow.log_metric("best_val_f1", best_val_f1)

        mlflow.pytorch.log_model(model, f"model_trial_{trial.number}")

        print(f"Trial {trial.number} Completed: F1 Score = {best_val_f1}")

    return best_val_f1  # Optuna will maximize this score

# Run Optuna
study = optuna.create_study(direction="maximize")  
study.optimize(objective, n_trials=20) 

# Log best parameters to MLflow
best_params = study.best_params
mlflow.log_params(best_params)

print(f"Best hyperparameters found: {best_params}")


