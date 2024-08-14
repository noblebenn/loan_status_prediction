# Basic Loan Status Prediction

## Project Overview

This project aims to predict the loan status (approved or not approved) based on a dataset of loan applications. The prediction is made using a Support Vector Classifier (SVC) model. The dataset contains various features such as gender, marital status, education, income, and property area, which are used to make predictions.

## Dataset

The ![dataset](train_u6lujuX_CVtuZ9i (1).csv) used in this project includes information about various loan applicants, including their demographic and financial details. The target variable is `Loan_Status`, which indicates whether the loan was approved (`Y`) or not (`N`).

## Libraries Used

- **NumPy**: For numerical operations.
- **Pandas**: For data manipulation and analysis.
- **Seaborn**: For data visualization.
- **Matplotlib**: For plotting graphs.
- **Scikit-learn**: For machine learning models and accuracy metrics.

## Project Structure

- `Loan_Status_Prediction.ipynb`: The Jupyter notebook containing the code for the project.
- `train_u6lujuX_CVtuZ9i (1).csv`: The dataset used for training and testing the model.
- `README.md`: This file, providing an overview of the project.

## Data Preprocessing

- **Handling Missing Values**: Missing values were handled by removing rows with missing data.
- **Categorical Variables**: Categorical variables were converted to numerical values using label encoding.

## Exploratory Data Analysis (EDA)

Some visualizations were created to understand the distribution of the data and the relationship between features and the target variable.

### Loan Status by Education

![Loan Status by Education](images/loan_status_by_education.png)

### Loan Status by Marital Status

![Loan Status by Marital Status](images/loan_status_by_marital_status.png)

## Model Training

The dataset was split into training and testing sets using an 90/10 split. The SVC model was trained on the training set.

```python
# Fitting the Support Vector Classifier Model
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

