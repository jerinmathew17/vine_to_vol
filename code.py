import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import random
import tensorflow as tf

# Interpolated data for grapes from 1 to 50 bottles
bottles = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
                    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                    41, 42, 43, 44, 45, 46, 47, 48, 49, 50])
grape_quantities = np.array([0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 
                             2.5227272727272725, 3.0454545454545454, 
                             3.5681818181818183, 4.090909090909091, 
                             4.613636363636363, 5.136363636363637, 
                             5.659090909090909, 6.181818181818182, 
                             6.704545454545454, 7.227272727272727, 
                             7.75, 8.272727272727273, 
                             8.795454545454545, 9.318181818181818, 
                             9.84090909090909, 10.363636363636363, 
                             10.886363636363637, 11.409090909090908, 
                             11.931818181818182, 12.454545454545453, 
                             12.977272727272727, 13.5, 
                             14.022727272727272, 14.545454545454545, 
                             15.068181818181818, 15.59090909090909, 
                             16.113636363636363, 16.636363636363637, 
                             17.159090909090907, 17.68181818181818, 
                             18.204545454545453, 18.727272727272727, 
                             19.25, 19.772727272727273, 
                             20.295454545454543, 20.818181818181817, 
                             21.34090909090909, 21.863636363636363, 
                             22.386363636363637, 22.909090909090907, 
                             23.43181818181818, 23.954545454545453, 
                             24.477272727272727, 25.0]) 

# Reshape data for model
X = bottles.reshape(-1, 1)  # Features
y = grape_quantities.reshape(-1, 1)  # Target

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

seed_value = 42

random.seed(seed_value)

np.random.seed(seed_value)

tf.random.set_seed(seed_value)

# Build the neural network model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(1,)))# First hidden layer with 5 neurons
model.add(Dense(5, activation='relu', input_shape=(1,))) 
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=300, batch_size=5, verbose=1)

# Predict on the entire range
y_pred_scaled = model.predict(X_scaled)

# Inverse transform to get original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Plot the original data and the fitted model
plt.scatter(bottles, grape_quantities, color='blue', label='Original Data', s=50)
plt.plot(bottles, y_pred, color='red', label='Fitted Model', linewidth=2)
plt.title('Simple Neural Network Fit for Grape Quantity Prediction')
plt.xlabel('Number of Bottles')
plt.ylabel('Grape Quantity')
plt.legend()
plt.grid()
plt.show()

# Predict for 600 bottles
predicted_600_bottles = model.predict(scaler_X.transform(np.array([[600]])))
predicted_grapes_600 = scaler_y.inverse_transform(predicted_600_bottles)
print(f'Predicted grape quantity for 600 bottles: {predicted_grapes_600[0][0]} kg')
