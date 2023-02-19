import numpy as np
import pandas as pd
# Load the dataset
data = pd.read_csv(r'C:\Users\DELL\Desktop\seeds_data.csv')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1],
data.iloc[:,-1], test_size=0.2, random_state=42)

# Create an instance of SimpleImputer and use it to impute
# the missing values in the training set
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

# Train the KNN algorithm on the original training set and the
# imputed training set
knn_original = KNeighborsClassifier(n_neighbors=3)
knn_imputed = KNeighborsClassifier(n_neighbors=3)

knn_original.fit(X_train, y_train)
knn_imputed.fit(X_train_imputed, y_train)

# Evaluate the performance of the KNN algorithm on the original
# testing set and the imputed testing set
y_pred_original = knn_original.predict(X_test)
y_pred_imputed = knn_imputed.predict(imputer.transform(X_test))

# Compare the performance of the KNN algorithm on the original
# and imputed testing sets
accuracy_original = accuracy_score(y_test, y_pred_original)
accuracy_imputed = accuracy_score(y_test, y_pred_imputed)

print(f'Accuracy on original testing set: {accuracy_original:.2f}')
print(f'Accuracy on imputed testing set: {accuracy_imputed:.2f}')
