import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

df = pd.read_csv('parkinsons.csv')
df.head()

selected_features = [['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)']]
output_features = ['status']

# Assuming 'selected_features' is a list of lists, where each inner list contains the column names.
# Extract the column names.
selected_columns = selected_features[0]

# Extract the selected features from the DataFrame
X = df[selected_columns]

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the selected features
X_scaled = scaler.fit_transform(X)

# Convert the scaled features back to a DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=selected_columns)

# Print the scaled features
print(X_scaled_df.head())


# Assuming 'output_features' is a list containing the name of the output column
y = df[output_features[0]]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42) # Adjust test_size as needed

# Print the shapes of the resulting sets to verify the split
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)

# Initialize the SVM model
svm_model = SVC(kernel='linear', C=1) # You can experiment with different kernels and C values

# Train the model
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy of SVM model: {accuracy}")
