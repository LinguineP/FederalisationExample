import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys

from ptbfla_pkg.ptbfla import *

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
    def extract_model(self):
        state_dict=self.state_dict()
        return {key:value.cpu().numpy().tolist() for key , value in state_dict.items()}

    def load_model(self,model_dict):
        keys = [key for key in model_dict]
        #print(model_dict)
        #print(type(model_dict))
        state_dict={key:torch.tensor(value) for key, value in model_dict.items()}
        self.load_state_dict(state_dict)


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
        
        
        # Insert the server value at the specified position
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

def fl_cent_client_processing(localData, privateData, msg):

    X_partition = privateData[0]
    y_partition = privateData[1]
    input_dim = X_partition.shape[1]
    model=FeedForwardNN(input_dim)
    model.load_model(msg)
    num_epochs=1000
    learning_rate=0.001
    
    model=train_model(model, X_partition, y_partition,num_epochs,learning_rate)
    return model.extract_model() 

def fl_cent_server_processing(privateData, msgs):

    input_dim=privateData
    models=[]
    for msg in msgs:
        model=FeedForwardNN(input_dim)
        model.load_model(msg)
        models.append(model)
    
    global_model=aggregate_models(models,privateData)
    
    return global_model.extract_model()

def seq_horizontal_federated_with_callbacks():
    X_train, X_test, y_train, y_test = preprocess_data(data)
    input_dim = X_train.shape[1]
    partitions = split_data(X_train, y_train, num_clients=2)
    pretrained_path = "/home/pavle/FederalisationExample/phases/starting_model.pth"  

    
    models = []
    for pData in partitions:
        model=FeedForwardNN(input_dim)
        model.load_pretrained_model(pretrained_path)
        lData=model.extract_model()    
        trained_model = fl_cent_client_processing(lData,pData,lData)
        models.append(trained_model)
    
    
    lData,pData=input_dim,models
    global_model_parameters = fl_cent_server_processing(lData,pData)
    
    model.load_model(global_model_parameters)
    
    evaluate_model(model, X_test, y_test)





# PTB-FLA Code
def main():
    if len(sys.argv) != 4:
        # Args: noNodes nodeId flSrvId
        #   noNodes - number of nodes, nodeId - id of a node, flSrvId - id of the FL server
        print('Program usage: python example4_logistic_regression.py noNodes nodeId flSrvId')
        print('Example: noNodes==3, nodeId=0..2, flSrvId==2, i.e. 3 nodes (id=0,1,2), server is node 2:')
        print('python example4_logistic_regression.py 3 0 2',
            '\npython example4_logistic_regression.py 3 1 2\npython example4_logistic_regression.py 3 2 2')
        exit()
    
    # Process command line arguments
    noNodes = int( sys.argv[1] )
    nodeId = int( sys.argv[2] )
    flSrvId = int( sys.argv[3] )
    print(noNodes, nodeId, flSrvId)
    
    
    
    X_train, X_test, y_train, y_test = preprocess_data(data)
    input_dim = X_train.shape[1]
    print(input_dim)
    pretrained_path = "/home/pavle/FederalisationExample/phases/starting_model.pth"  

    # Initialize PTB-FLA
    ptb = PtbFla(noNodes, nodeId,flSrvId)  
    
    partitioned_data = split_data(X_train, y_train, num_clients=2,server_id=flSrvId)  
    

    pData = partitioned_data[nodeId]
        
    
    model=FeedForwardNN(input_dim)
    model.load_pretrained_model(pretrained_path)
    
    lData=model.extract_model()

    
    global_model_parameters = ptb.fl_centralized(fl_cent_server_processing, fl_cent_client_processing, lData,pData ,2)
    
    model.load_model(global_model_parameters)
    
    role="Client"
    if(flSrvId==nodeId):
        role="Server"
    
    print("####################################")
    
    print(role)
    
    print(noNodes, nodeId, flSrvId)
    
    evaluate_model(model, X_test, y_test)
    
    del ptb
    
    
    pkey = input('press any key to continue...')


if __name__ == "__main__":
    #seq_base_case()
    #seq_horizontal_federated()
    #seq_horizontal_federated_with_callbacks()
    main()
