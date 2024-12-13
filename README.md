# **Machine Learning Project - Yeasts' genotype to describe their doubling time**

## **Description**
The primary objective of this project is to apply machine learning algorithms to build and evaluate a model that classifies individuals based on their risk of MICHD. Multiple models, including gradient descent, stochastic gradient descent, least squares, logistic regression, and ridge regression, were tested and optimized using cross-validation to minimize classification error.

## **Project Structure**
The project is organized as follows:  
├── data/dataset/               # Contains data files  (x_test.csv, x_train.csv, y_train.csv, sample_submission.csv)  
├── grading_tests/              # Files allowing to test different functions  
├── implementations.py          # File containing all ML functions  
├── helpers.py                  # Utility functions for the project  
├── README.md                   # Project documentation  
├── run.ipynb                   # To make all process of learning and predicting  
└── (run.py)                    # Not necessary, only if needed but less details than in run.ipynb   

## **Installation and Dependencies**
##### Prerequisites
Ensure that you have the Python 3.7 or more recent versions. You can install the dependencies using the `pip` commands.

##### Install Dependencies
1. Clone this repository:
   ```bash
   git clone https://github.com/CS-433/ml-project-1-heart-breakers.git
   cd ml-project-1-heart-breakers

2.	Install the required libraries:
	```bash
    pip install numpy
    pip install matplotlib



## Data
The dataset used in this project consists of health and lifestyle information from individuals, collected by the Behavioral Risk Factor Surveillance System (BRFSS). It includes features such as age, gender, smoking habits, physical activity, and medical history. The target variable is whether or not an individual has MICHD (binary classification: 1 for disease, -1 for no disease).

##### Example Data Structure:
	•	x_train: Input features (medical data)
	•	y_train: Target labels (-1 for no disease, 1 for disease)
	•	x_test: Input features for the test set

## Main Features
The project includes the following functionalities:

##### Data Preprocessing:
1. Handle Missing Values  
2. One-Hot Encoding for Categorical Features  
3. Outlier Management  
4. Remove Correlated Features  
5. Normalize Data  
6. Balance Classes  

These preprocessing steps collectively improve data quality, ensuring that features are informative, consistently scaled, and suitable for model training. This thorough data preparation process aids in achieving robust and reliable predictions.

##### Hyperparameter Optimization: 
Using cross-validation to select the best hyperparameters (lambda for regularization, gamma for learning rate).  
The cross-validation is made on the regularized logistic regression model.

##### ML Model Implementation:
Various machine learning algorithms were implemented and trained on the dataset.    
•	Linear Regression with gradient descent and stochastic gradient descent.   
•	Ridge Regression (regularized linear regression).  
•	Least Squares.  
•	Logistic Regression with and without regularization.  


##### Model Evaluation: 
Calculating the loss (MSE for linear regression, log-likelihood for logistic regression), generating predictions on the test set and using metrics such as classification accuracy.

##### Prediction Generation : 
Predictions were generated on the test dataset, and the final model was evaluated on AICrowd for ranking purposes.


## Usage
All use of the implemented functions is performed in the file `run.ipynb`, following the different steps listed. 
For the 3rd step : "initialization of the parameters for model training", you can choose the cell to run according to the model that you would like to use for making your predictions.

1. Loading data and preprocessing :   
Data is loaded using the `load_data()` function from `helpers.py`.   
This reads the CSV files and returns x_train, x_test, y_train, and patient IDs.  

Before training a model, data must be preprocessed. This includes:   
•	Handling Missing Values via the `handle_missing_data()` function  
•	Normalization of data using `normalisation()`  
•	Removing Correlated Features with `remove_correlated_features()`  
All the preprocessing is done by calling the function `preprocessed_data()`.

2. Initialisation of parameters and hyperparameters :   
Using predefinite values.  
Using Cross-Validation and Hyperparameter Optimization:   
You can use the `cross_validation_demo()` function to perform cross-validation on hyperparameters lambda and gamma and select the best combination, the one giving the best loss.

Example of utilisation :
```bash
lambdas = [0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1]
best_lambda, best_gamma, best_loss = cross_validation_demo(y_train, x_train, lambdas, gammas)
```


3. Training Models:   
Several models can be trained, including:  
	•	Linear Regression with `mean_squared_error_gd()` for gradient descent or `mean_squared_error_sgd()` for stochastic gradient descent.  
    •   Analytical method for linear regression, solving directly the normal equations using `least_squares()`.  
    •   A type of linear regression that works by applying a regularization term (L2 penalty) in its loss function `ridge_regression()`.  
	•	Logistic Regression with `logistic_regression()` or the regularized version with `reg_logistic_regression()`.  

Example of training with regularized logistic regression:
```bash
w_final, loss_final = reg_logistic_regression(y_train, x_train, lambda_, initial_w, max_iters=1000, gamma=best_gamma)
```


4. Prediction:  
After training, you can generate predictions on the test set and compare them with the expected results.  
`y_test_pred = np.where(x_test @ w_final >= 0, 1, -1)`  # Binary prediction  
The predictions can be saved on a csv file using create_csv_submission().  

5. Submission file to evaluate predictions:  
The predictions can be saved on a csv file using create_csv_submission().  


6. Plot some graphes:  
Make different graphes using `matplotlin.pyplot`.  


## Results and Performance  
The model’s performance is measured using loss functions and metrics like accuracy and F1-score. Final results can be stored and visualized in the results/ directory.

## Possible improvements  
• Implement other classification models such as SVM or Random Forest.  
• Add additional performance metrics to evaluate models more precisely.  
• Optimize execution time using techniques like batch processing or mini-batch gradient descent.


