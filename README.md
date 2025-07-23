# Prodigy InfoTech DS Internship - Task 03: Decision Tree Classifier

**Author:** Aaryan  
**Date:** [Today's Date]

## Project Overview

This project focuses on building a machine learning model to predict whether a customer will subscribe to a bank's term deposit. A **Decision Tree Classifier** was built using demographic and behavioral data from the Bank Marketing dataset.

The project uses a local copy of `bank.csv` (a smaller version of the full dataset) for the analysis. The complete workflow includes data loading from the local file, preprocessing, model training, performance evaluation, and visualization.

## Folder Structure

```├── data/
│   └── bank.csv                  # The dataset used for training and testing
├── visualizations/
│   ├── 01_confusion_matrix.png
│   └── 02_decision_tree.png
├── decision_tree_classifier.py   # Main Python script for the analysis
├── README.md                       # This explanation file
└── .gitignore


## How to Run the Code

1.  **Prerequisites:** Ensure you have Python installed with the necessary libraries:
    *   pandas
    *   scikit-learn
    *   matplotlib
    *   seaborn

    Install them via pip:
    ```bash
    pip install pandas scikit-learn matplotlib seaborn
    ```

2.  **Execution:** Run the main script from the project's root directory:
    ```bash
    python decision_tree_classifier.py
    ```
    The script will download the data, train the model, print the evaluation metrics to the console, and save the generated plots in the `visualizations/` folder.

## Summary of Findings

### Model Performance

The Decision Tree Classifier was trained and evaluated, achieving the following results on the test set:
*   **Accuracy:** Approximately **88.29%**. This means the model correctly predicted whether a customer would subscribe for about 88% of the customers in the test data.

The detailed **Classification Report** and **Confusion Matrix** provide deeper insights:
*   The model is very effective at identifying customers who will **not** subscribe (high precision and recall for class '0').
*   It is less effective at identifying customers who **will** subscribe (lower precision and recall for class '1'). This is a common challenge in marketing campaigns where the number of positive outcomes (subscriptions) is much lower than the negative ones (an imbalanced dataset).

### Key Predictors (Interpreting the Tree)

The visualization of the decision tree reveals the most influential factors the model uses to make predictions. The top-level decisions are based on:

1.  **`duration`**: The duration of the last contact is the single most important predictor. This suggests that the longer a call lasts, the more engaged the customer is, and the more likely they are to subscribe.
2.  **`poutcome_success`**: Whether a previous marketing campaign was successful for that customer is another powerful predictor.
3.  **`month`**: The month of contact also plays a role, likely due to seasonal factors or the timing of specific campaigns.

In conclusion, the Decision Tree model provides a clear and effective framework for predicting customer subscriptions, highlighting key behavioral and temporal features that drive decision-making.