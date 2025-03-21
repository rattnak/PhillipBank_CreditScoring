# PhillipBank_CreditScoring
Credit Risk Analysis Project
============================

This project performs a credit risk analysis using machine learning techniques. The notebook guides users through data loading, preprocessing, exploratory data analysis (EDA), model training, and evaluation. The goal is to classify loan applications as high or low risk based on historical data.

Project Structure and Code Explanation
--------------------------------------

The following sections provide explanations for each code block in the notebook, describing its function and importance in the overall analysis.

* * * * *

### Import Libraries

The libraries imported in this project provide a foundation for data handling, visualization, and modeling.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
```

-   **pandas**: For data manipulation, loading, and preprocessing, making it easy to work with structured data.
-   **numpy**: For efficient numerical computations, particularly useful when performing mathematical operations on large datasets.
-   **sklearn**: A machine learning library providing algorithms and tools for data splitting, preprocessing, and model evaluation.
    -   `train_test_split`: Splits the dataset into training and testing sets to validate model performance.
    -   `StandardScaler`: Standardizes features by removing the mean and scaling to unit variance, ensuring numerical stability across features.
    -   `LabelEncoder`: Encodes categorical features into numerical values, which is essential for feeding categorical data into machine learning models.
    -   `RandomForestClassifier`: An ensemble learning method that combines multiple decision trees, known for its robustness and high accuracy.
    -   `accuracy_score` and `classification_report`: Evaluate the model's accuracy and provide detailed metrics, such as precision, recall, and F1-score.
-   **matplotlib and seaborn**: Libraries for data visualization, helping to understand data distributions and correlations for better feature engineering and EDA insights.

* * * * *

### Load Dataset

This section loads the data into a pandas DataFrame from a CSV file. By previewing the data with `data.head()`, we get a quick view of the dataset's structure, column names, data types, and potential missing values.

```python
# Load the data
data = pd.read_csv('philip.csv')

# Preview data
print(data.head())
```

* * * * *

### Check for Missing Values

Missing values are checked using `data.isnull().sum()`, which counts null values for each column. Handling missing values is crucial as they can disrupt machine learning models and lead to inaccurate predictions.

```python
# Check for missing values
print(data.isnull().sum())

# Fill missing values only for numeric columns using the mean
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
```

-   **Imputation strategy**: Numeric columns are filled using the mean value, effective if the data distribution is relatively normal. Other imputation methods, like median or mode, could be used for skewed distributions or if outliers are present.

* * * * *

### Encode Categorical Columns

Categorical columns like `EmploymentStatus`, `EducationLevel`, `MaritalStatus`, and `LoanPurpose` are transformed into numerical format using `LabelEncoder`. Label encoding is particularly useful for models like random forests, which do not require a strict ordinal relationship in categorical features.

```python
# Encode categorical columns
categorical_cols = ['EmploymentStatus', 'EducationLevel', 'MaritalStatus', 'LoanPurpose']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
```



-   **Reasoning**: Label encoding is suitable here because we have categorical variables with discrete, non-ordinal values.

* * * * *

### Define Features and Target

The dataset is divided into features and target variables. `features` contain the independent variables used for prediction, while `target` contains the outcome variable (`LoanApproved`).

```python
# Define features (exclude non-numeric fields and the target)
features = data.drop(columns=['ApplicationDate', 'LoanApproved', 'RiskScore'])
target = data['LoanApproved']
```

* * * * *

### Train-Test Split

This block splits the data into training and testing sets to validate model performance. The training set is used to build the model, while the testing set evaluates its generalization to new data.

```python
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Identify categorical columns and apply Label Encoding
categorical_cols = X_train.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Now apply the scaler after encoding is complete
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

-   **Explanation**: `train_test_split` is set to 30% for testing and 70% for training, providing enough data for both training and evaluating model performance. Encoding and scaling are applied to ensure compatibility with the model and avoid feature imbalances.

* * * * *

### Data Visualization

This section generates visual representations of data distribution and variable relationships. Visualization helps to identify patterns, correlations, and outliers.

```python
# Exclude non-numeric columns for correlation calculation
numeric_data = data.select_dtypes(include=[np.number])

# Visualize correlations
plt.figure(figsize=(19, 12))
sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.show()

# Example of distribution plots for key features
sns.histplot(data['CreditScore'], kde=True)
plt.title('Distribution of Credit Scores')
plt.show()

sns.boxplot(data['AnnualIncome'])
plt.title('Annual Income Distribution')
plt.show()
```

-   **Explanation**: A heatmap shows correlations between features, guiding feature selection. Distribution and box plots reveal the distribution shape and presence of outliers for specific features.

* * * * *

### Model Training

A `RandomForestClassifier` is initialized and trained on the training dataset. This model's ensemble approach helps in achieving high accuracy and reduces overfitting.

```python
# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)
```

-   **Reason for Random Forest**: This model works well with imbalanced datasets and can handle a mix of numeric and categorical data without extensive tuning. Its ensemble nature makes it robust and less prone to overfitting compared to single decision trees.

* * * * *

### Model Evaluation

This block evaluates the model's performance using accuracy and classification metrics like precision, recall, and F1-score.

```python
# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Print classification report
print(classification_report(y_test, y_pred))
```

-   **Explanation**: `accuracy_score` provides a general measure of model correctness, while `classification_report` offers a breakdown of precision, recall, and F1-score, giving insight into how well the model distinguishes between classes.

* * * * *

### Feature Importance Analysis

Finally, feature importance is assessed to understand which variables contribute most to the model's decision-making. Important features can be analyzed to inform business insights.

```python
# Feature importance analysis
feature_importances = model.feature_importances_
feature_names = features.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names)
plt.title("Feature Importance")
plt.show()
```

-   **Explanation**: Feature importance highlights the predictive power of each feature, allowing us to understand which factors most influence credit risk. This insight can help refine the model and interpret its predictions.
