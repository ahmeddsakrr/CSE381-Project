import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Print the number of available CPU cores
print("Number of CPU cores:", os.cpu_count())

# Load the dataset
full_train = pd.read_csv("datasets/train_clean.csv")

# Columns to drop
to_drop = ["Name", "Ticket", "Embarked", "PassengerId",
           *[c for c in full_train if "cabin" in c.lower()]]
full_train.drop(columns=to_drop, inplace=True)

# Target column
target_col = "Survived"

X = full_train.drop(target_col, axis=1)
y = full_train[target_col]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the MLPClassifier model
mlp_model = MLPClassifier()

# Define the parameter grid for GridSearchCV
param_grid = {
    'hidden_layer_sizes': [
        (50,),
        (100,),
        (50, 50),
        (100, 50, 25),
        (100, 100, 100),
        (50, 50, 50, 50),
        (300, 300, 200, 100),
        (400, 300, 200, 100),
        (500, 400, 300, 200, 100),
    ],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    # 'learning_rate': ['constant', 'adaptive'],
    'learning_rate': ['constant'],
    'max_iter': [1000],
    'momentum': [1],
    'early_stopping': [True],
}

param_grid = {
    'hidden_layer_sizes': [
        (400, 300, 200, 100),
    ],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.01],
    # 'learning_rate': ['constant', 'adaptive'],
    'learning_rate': ['constant'],
    'max_iter': [1000],
    'momentum': [1],
    'early_stopping': [True],
}

# Create the GridSearchCV object
grid_search = GridSearchCV(mlp_model, param_grid,
                           n_jobs=8, cv=5, scoring='accuracy', verbose=3)

# Fit the model
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test_scaled)

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)

print(f"Best Parameters: {best_params}")
print(f"Best Score: {grid_search.best_score_:.4f}")



joblib.dump(grid_search, 'grid_search_model.joblib')


loaded_grid_search = joblib.load('grid_search_model.joblib')

for k, v in loaded_grid_search.cv_results_.items():
    print(k, v, sep="\t")


real_test = pd.read_csv("datasets/test_clean.csv")

real_test.drop(columns=list(set(to_drop)-{"PassengerId"}), inplace=True)

real_test = real_test.interpolate(method='from_derivatives')

scaler.fit(X_train)

real_test_scaled = scaler.transform(real_test.drop(columns=["PassengerId"]))

pred = pd.DataFrame(best_model.predict(real_test_scaled)).rename(columns={0: "Survived"})

pd.concat([real_test[["PassengerId"]], pred],
          axis=1).to_csv("MLP_v1.csv", index=None)
