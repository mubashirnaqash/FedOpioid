import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import numpy as np

def load_data(filepath):
    data = pd.read_csv(filepath)
    print("Shape of the dataset:", data.shape)
    print("Column names:", data.columns.tolist())
    print("First few rows of the dataset:\n", data.head())
    return data

def preprocess_data(data):
    # Vectorize 'context' text data
    vectorizer = TfidfVectorizer(max_features=1000)
    text_features = vectorizer.fit_transform(data['context'].fillna('')).toarray()
    print("Text features shape:", text_features.shape)

    # Drop 'note_id' as it's likely an identifier and separate target variable
    X = data.drop(['Diagnosed opioid dependence', 'note_id'], axis=1)
    y = data['Diagnosed opioid dependence']

    # Numeric features preprocessing pipeline
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    print("Numeric features:", numeric_features)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical features preprocessing pipeline
    categorical_features = X.select_dtypes(include=['object']).columns.drop('context')
    print("Categorical features:", categorical_features)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Pass through other columns (context in this case)
    )

    # Apply preprocessing excluding 'context' which has been vectorized
    X_preprocessed = preprocessor.fit_transform(X.drop('context', axis=1))
    X_preprocessed = np.hstack((X_preprocessed, text_features))
    print("Preprocessed features shape:", X_preprocessed.shape)

    # Balancing the dataset
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_preprocessed, y)
    print("Shape after balancing (features):", X_balanced.shape)
    print("Shape after balancing (target):", y_balanced.shape)

    return X_balanced, y_balanced

def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print("Training set shape:", X_train.shape, y_train.shape)
    print("Test set shape:", X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test

def main():
    data = load_data('ORAB_Annotation_MIMIC.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main()
