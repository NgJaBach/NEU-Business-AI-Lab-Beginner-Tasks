# Importing the necessary libraries
from sklearn.impute import SimpleImputer
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# Load the dataset
dataset = pd.read_csv('pima-indians-diabetes.csv')

# Identify missing data (assumes that missing data is represented as NaN)
missing_data = dataset.isnull().sum()

# Print the number of missing entries in each column
print("Missing data: \n",missing_data)

# Configure an instance of the SimpleImputer class
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer on the DataFrame
imputer.fit(dataset)

# Apply the transform to the DataFrame
dataset_imputed = imputer.transform(dataset)

#Print your updated matrix of features
print("Updated matrix of features: \n", dataset_imputed)
