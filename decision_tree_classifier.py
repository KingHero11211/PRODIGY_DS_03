# decision_tree_classifier.py
# Author: Aaryan
# Date: [Today's Date]
#
# Prodigy InfoTech Data Science Internship - Task 03
#
# Objective: Build a decision tree classifier to predict customer purchases.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
import os

def build_decision_tree_model():
    """
    Main function to load, preprocess, train, and evaluate the decision tree model.
    """
    print("--- Starting Decision Tree Classifier Task ---")

    # --- Step 0: Setup and Data Loading ---
    
    # Create a directory to save visualizations
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
        
    # Define the local file path
    # This assumes the script is run from the project's root directory (DS_03)
    file_path = 'data/bank.csv'

    try:
        # Read the local CSV file, which uses semicolons as separators
        df = pd.read_csv(file_path, sep=';')
        print(f"Dataset loaded successfully from '{file_path}'.")
    except FileNotFoundError:
        print(f"Error: The file was not found at '{file_path}'.")
        print("Please make sure the 'bank.csv' file is inside the 'data' folder.")
        return
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return

    # --- Step 1: Data Preprocessing ---
    print("\n--- 1. Data Preprocessing ---")

    # Display basic info and check for missing values
    print("Dataset Info:")
    df.info()
    print("\nFirst 5 rows:")
    print(df.head())

    # The target variable is 'y' (subscription to a term deposit)
    # Convert the target variable 'y' from 'yes'/'no' to 1/0
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    print("\nTarget variable 'y' converted to numeric.")

    # Identify categorical columns to be encoded
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"\nCategorical columns to be one-hot encoded: {list(categorical_cols)}")

    # Apply one-hot encoding to categorical features
    # This converts categorical variables into a format that can be provided to ML algorithms
    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print("\nData after one-hot encoding:")
    print(df_processed.head())

    # --- Step 2: Define Features (X) and Target (y) ---
    print("\n--- 2. Defining Features and Target ---")
    
    X = df_processed.drop('y', axis=1)
    y = df_processed['y']
    
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")

    # --- Step 3: Split Data into Training and Testing Sets ---
    print("\n--- 3. Splitting Data into Train and Test Sets ---")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    # --- Step 4: Build and Train the Decision Tree Classifier ---
    print("\n--- 4. Building and Training the Model ---")
    
    # Initialize the Decision Tree Classifier
    # random_state=42 ensures reproducibility
    dt_classifier = DecisionTreeClassifier(random_state=42)
    
    # Train the model on the training data
    dt_classifier.fit(X_train, y_train)
    print("Decision Tree model trained successfully.")

    # --- Step 5: Evaluate the Model ---
    print("\n--- 5. Evaluating the Model ---")
    
    # Make predictions on the test set
    y_pred = dt_classifier.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Display the classification report (precision, recall, f1-score)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Generate and visualize the confusion matrix
    print("\nGenerating Confusion Matrix visualization...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Subscription', 'Subscription'],
                yticklabels=['No Subscription', 'Subscription'])
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.savefig('visualizations/01_confusion_matrix.png')
    plt.show()

    # --- Step 6: Visualize the Decision Tree ---
    print("\n--- 6. Visualizing the Decision Tree ---")
    print("Generating a visualization of the top levels of the Decision Tree...")
    
    plt.figure(figsize=(20, 10))
    # We limit the depth to 3 for readability. A full tree would be too large to interpret.
    plot_tree(dt_classifier, 
              max_depth=3, 
              feature_names=X.columns, 
              class_names=['No', 'Yes'], 
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title("Decision Tree Structure (Top Levels)", fontsize=20)
    plt.savefig('visualizations/02_decision_tree.png')
    plt.show()

    print("\n--- Task Complete ---")
    print("Visualizations have been saved to the 'visualizations' folder.")

if __name__ == "__main__":
    build_decision_tree_model()