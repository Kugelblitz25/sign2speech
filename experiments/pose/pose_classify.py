import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

train_data = pd.read_csv(r"features_train.csv")
val_data = pd.read_csv(r"features_val.csv")
test_data = pd.read_csv(r"features_test.csv")
train_pose_data = pd.read_csv(r"pose_features_train_10.csv")
val_pose_data = pd.read_csv(r"pose_features_val_10.csv")
test_pose_data = pd.read_csv(r"pose_features_test_10.csv")

train = train_data.merge(
    train_pose_data,
    left_on="Video file",
    right_on="video",
    how="inner",
    suffixes=("_data", "_pose"),
)
val = val_data.merge(
    val_pose_data,
    left_on="Video file",
    right_on="video",
    how="inner",
    suffixes=("_data", "_pose"),
)
test = test_data.merge(
    test_pose_data,
    left_on="Video file",
    right_on="video",
    how="inner",
    suffixes=("_data", "_pose"),
)

X_train = train.drop(columns=["Video file", "video", "Gloss"])
y_train = train["Gloss"]
X_val = val.drop(columns=["Video file", "video", "Gloss"])
y_val = val["Gloss"]
X_test = test.drop(columns=["Video file", "video", "Gloss"])
y_test = test["Gloss"]

scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_val_normalized = scaler.transform(X_val)
X_test_normalized = scaler.transform(X_test)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)

X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_normalized, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 5000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(5000, 2500),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2500, 1250),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1250, 625),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(625, 312),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(312, 156),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(156, 78),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(78, num_classes),
        )

    def forward(self, x):
        return self.model(x)


input_size = X_train.shape[1]
num_classes = len(label_encoder.classes_)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

early_stopping_patience = 5
best_val_loss = float("inf")
patience_counter = 0
best_model_path = "best_model.pth"

for epoch in range(100):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in tqdm(train_loader, desc=f"Training {epoch + 1}/100"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    train_acc = correct / total

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(val_loader, desc=f"Validating {epoch + 1}/100"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_correct += (predicted == y_batch).sum().item()
            val_total += y_batch.size(0)

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    print(
        f"Epoch {epoch + 1}/{100}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f},"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print("Best model saved!")
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break
