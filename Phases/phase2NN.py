import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv("BaseCode/input/housing.csv")


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
    
    def load_pretrained_model(self, pretrained_path):
        self.load_state_dict(torch.load(pretrained_path,weights_only=True))
        print("Pretrained model loaded successfully!")



def train_model(model:FeedForwardNN, X_train, y_train, num_epochs=10000, learning_rate=0.001,pretrained_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    
    if pretrained_path:
        model = model.load_pretrained_model(pretrained_path)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")
    
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

def seq_base_case():
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = FeedForwardNN(X_train.shape[1])  
    model = train_model(model, X_train, y_train, 10000)
    y_pred = evaluate_model(model, X_test, y_test)

def split_data(X_train, y_train, num_clients, server_id=None):
    data_splits = np.array_split(range(X_train.shape[0]), num_clients)
    partitions = [(X_train[idx], y_train[idx]) for idx in data_splits]

    if server_id is not None:    
        server_value = X_train.shape[1]  
        partitions.insert(server_id, server_value)
    
    return partitions

def federated_train(partitions, input_dim, num_epochs=2000, learning_rate=0.001, pretrained_path=None):
    models = []
    for i, (X_partition, y_partition) in enumerate(partitions):
        model = FeedForwardNN(input_dim)
        if pretrained_path:
            model.load_pretrained_model(pretrained_path)
            
        print(f"Training model for client {i+1}")
        trained_model = train_model(model, X_partition, y_partition, num_epochs, learning_rate)
        models.append(trained_model)
    return models

def aggregate_models(models, input_dim):
    
    global_model = FeedForwardNN(input_dim)
    global_state = global_model.state_dict()
    for key in global_state:
        global_state[key] = torch.stack([model.state_dict()[key] for model in models]).mean(dim=0)
    global_model.load_state_dict(global_state)
    return global_model

def seq_horizontal_federated():
    X_train, X_test, y_train, y_test = preprocess_data(data)
    input_dim = X_train.shape[1]
    partitions = split_data(X_train, y_train, num_clients=2)
    pretrained_path = "/home/pavle/FederalisationExample/phases/starting_model.pth"  
    models = federated_train(partitions, input_dim, pretrained_path=pretrained_path)
    global_model = aggregate_models(models, input_dim)
    evaluate_model(global_model, X_test, y_test)


if __name__ == "__main__":
    #seq_base_case()
    seq_horizontal_federated()
    

