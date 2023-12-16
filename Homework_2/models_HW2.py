
# @title Task 1 :  Human Activity Classification with LSTM

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import joblib  # For saving models and encoders
import zipfile
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau



#------------Python code for loading the dataset------#

file_path = 'Homework 2/UCI HAR Dataset'

# Function to load feature names from the dataset
def load_features(file_path):
    with open(file_path) as f:
        return [line.strip() for line in f.readlines()]

# Function to load and merge data, labels, and subject information
def load_dataset(data_folder, feature_names):
    data_path = f'UCI HAR Dataset/UCI HAR Dataset/{data_folder}/X_{data_folder}.txt'
    labels_path = f'UCI HAR Dataset/UCI HAR Dataset/{data_folder}/y_{data_folder}.txt'
    subject_path = f'UCI HAR Dataset/UCI HAR Dataset/{data_folder}/subject_{data_folder}.txt'

    data = pd.read_csv(data_path, delim_whitespace=True, names=feature_names)
    labels = pd.read_csv(labels_path, header=None, names=['Activity'])
    subject = pd.read_csv(subject_path, header=None, names=['Subject'])

    return pd.concat([subject, data, labels], axis=1)

# Load activity labels and feature names
activity_labels = pd.read_csv('UCI HAR Dataset/UCI HAR Dataset/activity_labels.txt', header=None, delim_whitespace=True, names=['ID', 'Activity'])
feature_names = load_features('UCI HAR Dataset/UCI HAR Dataset/features.txt')

# Load training and testing data
train_data = load_dataset('train', feature_names)
test_data = load_dataset('test', feature_names)

# Map numerical activity labels to their names for better interpretation
activity_mapping = activity_labels.set_index('ID')['Activity']
train_data['Activity'] = train_data['Activity'].map(activity_mapping)
test_data['Activity'] = test_data['Activity'].map(activity_mapping)

# Inspect the first few rows of the dataset
print("First few rows of training data:\n", train_data.head())

# Identify the feature columns (excluding 'Subject' and 'Activity')
feature_columns = train_data.columns.difference(['Subject', 'Activity'])



#----1. Python code for preprocessing the data by normalizing and encoding the activities----#

# Visualization before normalization (For Data Preprocessing Evaluation)
# Histograms and Boxplots
for col in feature_columns[:20]:  # Plotting a subset of features to avoid clutter
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(train_data[col], kde=True)
    plt.title(f'Histogram of {col}')

    plt.subplot(1, 2, 2)
    sns.boxplot(y=train_data[col])
    plt.title(f'Boxplot of {col}')

    plt.show()

# Correlation Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(train_data[feature_columns[:30]].corr(), annot=False, fmt=".1f", cmap="coolwarm")  # Reduced the number of features for clarity
plt.title('Correlation Matrix of Features')
plt.show()

# Data Normalization
scaler = StandardScaler()
train_data[feature_columns] = scaler.fit_transform(train_data[feature_columns])
test_data[feature_columns] = scaler.transform(test_data[feature_columns])

# Encoding Activity Labels
encoder = LabelEncoder()
train_data['Activity'] = encoder.fit_transform(train_data['Activity'])
test_data['Activity'] = encoder.transform(test_data['Activity'])

# Save the encoder for future reference
joblib.dump(encoder, 'activity_encoder.joblib')



#-----------2. Python code implementing your chosen neural network architecture---------#

# I fixed the random state to 11, to ensure reproducibility.
X_train, X_val, y_train, y_val = train_test_split(train_data[feature_columns], train_data['Activity'], test_size=0.2, random_state=11)

# Define and compile LSTM Model Architecture
model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.5))
model.add(LSTM(32))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(np.unique(y_train)), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=30,  # Adjust epochs depending on early stopping
    batch_size=64,  # Adjust batch size according to your data and GPU capability
    validation_data=(X_val, y_val),
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    ]
)



#------------4. Training and validation curves plotted over epochs.------------#

# Plot Training and Validation Accuracy and Loss
plt.figure(figsize=(14, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy - Human LTSM')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss - Human LTSM')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_data[feature_columns], test_data['Activity'])
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Save the model
model.save('human_activity_recognition_model.h5')







# @title Task 2 : Temperature Series Forecasting with GRU


#----1. Python code for loading the dataset----#

#I'm reading line by line because I was getting errors otherwise.
with open('daily-minimum-temperatures-in-me.csv') as f:
    file_content = f.readlines()

file_content = file_content[1:-3]

temperatures = []
dates = []

for line in file_content:
    date_str, temp_str = line.split(",")[:2]
    try:
        # Parse temperature value
        temperature = float(temp_str)
        temperatures.append(temperature)

        # Parse date and normalize components
        date_components = date_str.strip('"').split("-")
        year, month, day = [int(item) for item in date_components]

        # Normalize each component
        normalized_year = (year - 1980) / 10  # Assuming normalization by decade from 1980
        normalized_month = month / 12  # Normalizing by the number of months
        normalized_day = day / 31  # Normalizing by the maximum number of days in a month
        normalized_date = [normalized_year, normalized_month, normalized_day]

        dates.append(normalized_date)

    except ValueError:
        # Handle lines that are giving erros because they don't have a proper float temperature value
        print(f"Skipping line due to parsing error: {line.strip()}")
        continue

dates_array = np.array(dates)
temperatures_array = np.array(temperatures)
print("Dates Array Shape:", dates_array.shape)
print("Temperatures Array Shape:", temperatures_array.shape)



#----1. Python code for preprocessing the dataset----#

# Visualize the temperature data before normalization
plt.figure(figsize=(12, 6))
plt.plot(temperatures)
plt.title('Daily Minimum Temperatures over Time')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.show()

# Normalize the temperature data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_temperatures = scaler.fit_transform(np.array(temperatures).reshape(-1, 1)).flatten()

# Transform the series into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1):
    df = pd.DataFrame(data)
    cols = [df.shift(i) for i in range(n_in, -1, -1)]
    agg = pd.concat(cols, axis=1)
    agg.dropna(inplace=True)
    return agg.values

# Define the number of lagged observations
n_lag = 3
supervised_values = series_to_supervised(scaled_temperatures, n_lag)

# Split into train and test sets
train, test = train_test_split(supervised_values, test_size=0.2, random_state=11)

# Separate into input and output columns
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# Reshape input to be 3D [samples, timesteps, features] as expected by GRU
train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], 1))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], 1))

print(f'Training data shape: {train_X.shape}, Training labels shape: {train_y.shape}')
print(f'Test data shape: {test_X.shape}, Test labels shape: {test_y.shape}')



#----2. Build a GRU model tailored for regression----#

model = Sequential()
model.add(GRU(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))  # One output neuron => regression task
model.compile(optimizer='adam', loss='mean_squared_error')



#------3. Train and validate the model------#

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(train_X, train_y, epochs=50, batch_size=72,
                    validation_split=0.2, verbose=1, callbacks=[early_stopping])

# Plot training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model loss (Task 2)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()



#------4. Evaluate the predictions using appropriate metrics------#

test_X_reshaped = test_X.reshape((test_X.shape[0], test_X.shape[1]))

# Make predictions
predictions = model.predict(test_X_reshaped)

# Invert scaling for actual values
actual_y = scaler.inverse_transform(test_y.reshape(-1, 1)).flatten()
predicted_y = scaler.inverse_transform(predictions).flatten()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_y, predicted_y))
print('Test RMSE: %.3f' % rmse)

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(actual_y, label='Actual')
plt.plot(predicted_y, label='Predicted')
plt.title('Actual vs Predicted Temperatures (Task 2)')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.show()










# @title Task 3 : Unsupervised Feature Learning with Autoencoders

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split



#----1. Python code for loading and preprocessing the dataset----#

# Constants
ENCODING_DIM = 64
BATCH_SIZE = 32
EPOCHS = 50

# Retrieve and preprocess data
data = fetch_olivetti_faces()
faces = data.images

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(faces.ravel(), bins=50)
plt.title('Pixel Value Distribution')

plt.subplot(1, 2, 2)
plt.boxplot(faces.ravel())
plt.title('Pixel Value Boxplot')
plt.show()

train, test = train_test_split(faces, test_size=0.2, random_state=11)

# Reshape images to retain 2D structure
train_reshaped = train.reshape((-1, 64, 64, 1))
test_reshaped = test.reshape((-1, 64, 64, 1))



#----2. Model Architecture: Construct an autoencoder with convolutional layers----#

input_img = layers.Input(shape=(64, 64, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(ENCODING_DIM, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = layers.Conv2D(ENCODING_DIM, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = models.Model(input_img, decoded)
# evaluation using reconstruction errors thanks to binary cross entropy
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")



#----3. Python code for training the model----#
# Train the autoencoder
history = autoencoder.fit(train_reshaped, train_reshaped,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(test_reshaped, test_reshaped))

decoded_imgs = autoencoder.predict(test_reshaped)



#----4. Evaluation using reconstruction error and visualization----#

#-----5. Feature Visualization: Visualize weights of first encoder layer----#
weights = autoencoder.layers[1].get_weights()[0]
fig, axes = plt.subplots(1, 32, figsize=(20, 2))
for i, ax in enumerate(axes):
    ax.imshow(weights[..., i].squeeze(), cmap='gray')
    ax.axis('off')
plt.show()

# Learning Curve
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



# Display the original and reconstructed images
plt.figure(figsize=(20, 4))
for i in range(10):
    # Original image
    plt.subplot(2, 10, i + 1)
    plt.imshow(test[i], cmap='gray')
    plt.axis('off')

    # Reconstructed image
    plt.subplot(2, 10, i + 11)
    plt.imshow(decoded_imgs[i].reshape(64, 64), cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()