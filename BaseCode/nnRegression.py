import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data
data = pd.read_csv("BaseCode/input/housing.csv")

# Data preprocessing
def preprocess_data(data):
    columns_to_drop = ["housing_median_age", "households", "total_bedrooms", "longitude", 
                       "latitude", "total_rooms", "population"]
    data = data.drop(columns=columns_to_drop)
    
    X = data.drop("median_house_value", axis=1)
    y = data["median_house_value"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

# Define the Feedforward Neural Network Model
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.output = nn.Linear(100, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.01)
    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc3(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

# Train the model
def train_model(model, X_train, y_train, num_epochs=10000, learning_rate=0.001, patience=50, improvement_threshold=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")
        
        if loss.item() < best_loss - improvement_threshold:
            best_loss = loss.item()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}. No improvement in loss.")
            break
    
    model = model.to('cpu')
    return model


def evaluate_model(model, X_test, y_true):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy()
        y_true = y_true.numpy()
        
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse) 
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print("####################################")
    print(f"MSE: {mse:.4f} \nRMSE: {rmse:.4f} \nMAE: {mae:.4f} \nR²: {r2:.4f}")
    print("####################################")
    
    return y_pred


def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    sns.residplot(x=y_pred, y=residuals, lowess=True, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plt.show()


def main():
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = FeedForwardNN(X_train.shape[1])
    model = train_model(model, X_train, y_train,10000)
    y_pred = evaluate_model(model, X_test, y_test)
    plot_residuals(y_test.numpy(), y_pred)

if __name__ == "__main__":
    main()
