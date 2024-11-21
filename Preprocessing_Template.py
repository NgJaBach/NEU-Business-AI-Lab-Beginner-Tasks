import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from collections import Counter

# Loading the dataset
df = pd.read_csv('/content/Social_Network_Ads.csv')

# Exploratory Data Analysis (EDA)
pd.set_option('display.max_columns', None)
print(df.head(), '\n')

# Percentage of missing values
def missing_values_table(df):
    # Percentage of missing values for each column
    missing_percent = df.isnull().mean() * 100
    
    # Filter out columns with no missing values
    missing_data = missing_percent[missing_percent > 0].sort_values(ascending=False)
    
    # Create a dataframe of the result
    result = pd.DataFrame({'Missing Values (%)': missing_data, 
                           'Total Missing': df.isnull().sum()[missing_data.index]})
    return result

missing_data = missing_values_table(df)
print(missing_data, '\n')

# Dropping unnecessary columns
# df.drop(['ID', 'Unnamed: 0'], axis=1, inplace=True)

# Separate features and labels
X = df.drop('Purchased', axis=1)
y = df['Purchased']

# Split numerical and categorical data
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Pipeline for numerical data
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()) # NOT needed in tree-based models (RandomForest, XGBoost)
])

# Pipeline for categorical data
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine both pipelines
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Apply pipeline on the dataset
X_preprocessed = preprocessor.fit_transform(X)

# Split the dataset to train set and test set
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}", '\n')

# Function for plotting class distribution
def plot_class_distribution(y, title):
    pd.Series(y).value_counts().plot(kind='bar', title=title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()

# Class distribution before applying SMOTE
plot_class_distribution(y_train, 'Class Distribution Before SMOTE')

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 'Class distribution after applying SMOTE
plot_class_distribution(y_train, 'Class Distribution After SMOTE')
