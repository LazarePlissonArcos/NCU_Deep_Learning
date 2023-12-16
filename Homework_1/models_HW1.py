"""deep learning assigment 1 Lazare Plisson

"""

# @title Task 1 :  Implement a Feedforward Neural Network for Regression


#------------1. Python code for loading and preprocessing the dataset.------#



# importing libraries
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#---------------------------Loading the dataset-----------------------------#
dataset = pd.read_csv('TheBostonHousingDataset.csv')

#---------------------------Preprocessing the dataset-----------------------#
# separate input and output components
X = dataset.drop(columns=['MEDV'])  # input (Features)
Y = dataset['MEDV']  # output (Target)


# Standardize the features (mean=0, std=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# separate training and testing components
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=11)

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test.values, dtype=torch.float32)

batch_size = 64  # You can specify a different batch size if needed
trainloader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
testloader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size)




#----2. Python code implementing your chosen neural network architecture.------------#




class regression(nn.Module):
    def __init__(self, input_size):
        super(regression, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
# 3 fully connected layers
# ReLU is an activation fonction.

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

input_size = X_train.shape[1]  # The number of features
model = regression(input_size)

criterion = nn.MSELoss()  # Mean Squared Error
optimizer = optim.Adam(model.parameters(), lr=0.01)
# lr = learning rate


#----------3. Training and validation curves plotted over epochs.------------#


num_epoch = 10
training_loss, validation_loss = [], []

#------------------Training---------------------#

for epoch in range(num_epoch):
    model.train()
    epoch_training_loss = 0
    for inputs, targets in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_training_loss += loss.item() * inputs.size(0)


    epoch_training_loss /= len(trainloader.dataset)
    training_loss.append(epoch_training_loss)

    #-----------Validation------------#
    model.eval()
    epoch_validation_loss = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_validation_loss += loss.item() * inputs.size(0)

    epoch_validation_loss /= len(testloader.dataset)
    validation_loss.append(epoch_validation_loss)


# Ploting the curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epoch + 1), training_loss, label='Training Losses', color='b', marker='o')
plt.plot(range(1, num_epoch + 1), validation_loss, label='Validation Losses', color='r', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Curves (Boston)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(range(1, num_epoch + 1))
plt.tight_layout()
plt.legend(['Training Losses', 'Validation Losses'])
plt.grid(True, linestyle='--', alpha=0.8)
plt.scatter(range(1, num_epoch + 1), training_loss, color='b', marker='o', label='Training Losses')
plt.scatter(range(1, num_epoch + 1), validation_loss, color='r', marker='s', label='Validation Losses')
plt.show()




# @title Task 2: Breast Cancer Dataset

# importing datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


#------------1. Python code for loading and preprocessing the dataset.------#


# hyperparameters definition
num_epoch = 20
size_batch = 40

#---------------------------Loading the dataset-----------------------------#
data_task2 = load_breast_cancer()

#---------------------------Preprocessing the dataset-----------------------#

# standardization of features (mean = 0, std = 1)
scaler = StandardScaler()
X = scaler.fit_transform(data_task2.data)

# separate training and testing components
X_train,X_test,Y_train,Y_test=train_test_split(data_task2.data,data_task2.target,test_size=0.3,random_state=11)

X_train = torch.tensor(X_train)
Y_train = torch.tensor(Y_train).unsqueeze(1).to(torch.float64)
X_test = torch.tensor(X_test)
Y_test = torch.tensor(Y_test).unsqueeze(1).to(torch.float64)



trainloader_breast = DataLoader(TensorDataset(X_train,Y_train),batch_size=size_batch)
testloader_breast = DataLoader(TensorDataset(X_test,Y_test),batch_size=size_batch)


#----2. Python code implementing your chosen neural network architecture.------------#



class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.fc1 = nn.Linear(30, 10).to(torch.float64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 1).to(torch.float64)
        self.sigmoid = nn.Sigmoid().to(torch.float64)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

cancer_model = Classification()

criterion_cancer = nn.BCELoss()  #Binary Cross Entropy Loss
optimizer = optim.Adam(cancer_model.parameters(), lr=0.001)

#----------3. Training and validation curves plotted over epochs.------------#
train_losses = []
validation_losses = []


#------------------Training---------------------#

for epoch in range(num_epoch):
    running_loss = 0
    for inputs, targets in trainloader_breast:
        optimizer.zero_grad()
        outputs = cancer_model(inputs)
        loss = criterion_cancer(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    average_train_loss = running_loss * size_batch/len(trainloader_breast)
    train_losses.append(average_train_loss)



    #-----------Validation------------#
    cancer_model.eval()
    running_loss = 0
    epoch_validation_loss = 0
    with torch.no_grad():
        for inputs, targets in testloader_breast:
              outputs = cancer_model(inputs)
              loss = criterion_cancer(outputs, targets)
              epoch_validation_loss += loss.item() * inputs.size(0)

    epoch_validation_loss = epoch_validation_loss* size_batch /len(testloader_breast.dataset)
    validation_losses.append(epoch_validation_loss)




plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epoch + 1), train_losses, label='Training Losses', color='b', marker='o')
plt.plot(range(1, num_epoch + 1), validation_losses, label='Validation Losses', color='r', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Curves (Breast Cancer)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(range(1, num_epoch + 1))
plt.tight_layout()
plt.legend(['Training Losses', 'Validation Losses'])
plt.grid(True, linestyle='--', alpha=0.8)
plt.scatter(range(1, num_epoch + 1), train_losses, color='b', marker='o', label='Training Losses')
plt.scatter(range(1, num_epoch + 1), validation_losses, color='r', marker='s', label='Validation Losses')
plt.show()



# @title Task 3:  Fashion-MNIST Dataset


# importing libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


#------------1. Python code for loading and preprocessing the dataset.------#

# hyperparameters definition
num_epoch = 20



#---------------------------Loading the dataset-----------------------------#
data_task3 = keras.datasets.fashion_mnist
(X_train, Y_train), (X_test, Y_test) = data_task3.load_data()

#---------------------------Preprocessing the dataset-----------------------#
X_train = X_train / 255.0
X_test = X_test / 255.0


# Define a learning rate schedule
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

#----2. Python code implementing your chosen neural network architecture.------------#
model_task3 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(64, activation='relu'),  # an hidden layeer 64 units and ReLU fonction
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax') # output layer 10 units, softmax activation
])

#----------3. Training and validation curves plotted over epochs.------------#
# Compile the model
model_task3.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Implement early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,  # Number of epochs with no improvement to wait before stopping
    restore_best_weights=True
)

# Train the model
history_task3 = model_task3.fit(X_train, Y_train, epochs=num_epoch, validation_split=0.2)

# Plot training and validation curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_task3.history['accuracy'], label='Training Accuracy', color='b', marker='o')
plt.plot(history_task3.history['val_accuracy'], label='Validation Accuracy', color='r', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_task3.history['loss'], label='Training Loss', color='b', marker='o')
plt.plot(history_task3.history['val_loss'], label='Validation Loss', color='r', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()