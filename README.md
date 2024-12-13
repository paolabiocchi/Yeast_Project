# **Machine Learning Project - Yeasts' genotype to describe their doubling time**

## **Description**
The primary objective of this project is to apply machine learning algorithms to find the mutations and the number of proteins' sequences (copy number variations) of each yeast's genotype, that have the most effects on the yeasts' doubling time. To do so, we build and evaluate models that calculate the yeasts' doubling time based on their genotype. Multiple models, including BERT, ElasticNet, GBM, LASSO, LightGBM, Neural Networks, Random Forest, XGBoost, SVM, were tested and optimized using cross-validation to minimize classification error, find the best model that will indicate the most impactful mutations and copy number variations.

## **Project Structure**
The project is organized as follows:  
```
├── data/                       # Contains data files
│ ├── Proteome_1011/            # Contains all the fasta files corresponding to all yeasts' genotypes of each protein
│ ├── copy_number_variation_dataset.csv       # Contains all the copy number variations for all yeasts
│ ├── extended_mutations_[phenotype].csv      # Contains all the specific mutations for all yeasts
│ ├── Finalset_223phenotypes_1011.tab         # Contains all the phenotypes for all yeasts
│ ├── genesMatrix_CopyNumber.csv              # Contains all the copy number variations for all yeasts, as a table
│ ├── mutations_dataset.csv                   # Contains all the mutations for all yeasts, as a list
│ ├── phenotype_dataset.csv                   # Contains all the phenotypes for all yeasts, for easir label??
│ ├── X_matrix_[phenotype].csv                # Contains all the specific mutations and copy number variations for all yeasts
│ ├── y_[phenotype].csv             # Contains the phenotype of interest for all yeasts
├── results/                        # Contains results files
│ ├── plots/                        # Contains all plots (CNV per protein, mutation per protein, mutation per yeast)
│ ├── ...                           # File containing all Shap results  
├── extract_data_[./2].ipynb        # To extract X_matrix and y_[phenotype]
├── preprocessing.py                # To preprocess the data
├── model_[model].ipynb             # All models
├── analyse.txt                     # To save and compare model results
├── save_and_compare_results.py     # To save and compare model and Shap results
├── README.md                       # Project documentation  
├── run.ipynb                       # To make all process of learning and predicting  
```
## **Installation and Dependencies**
##### Prerequisites
Ensure that you have the Python 3.7 or more recent versions. You can install the dependencies using the `pip` commands.

##### Install Dependencies
1. Clone this repository:
   ```bash
   git clone https://github.com/paolabiocchi/Yeast_Project.git
   cd Yeast_Project

2.	Install the required libraries:
	```bash
    pip install dask lightgbm matplotlib numpy pandas scikit-learn shap scipy seaborn skorch torch transformers xgboost

## Data
The dataset used in this project has been extracted from yeasts' genotypes, traducting specific proteins. It includes features that are a mix of each specific mutation which have been found in at least 2 yeasts' genotypes, and all the copy number variations for all yeasts. The target variable is the yeast's doubling time. However, we are more interested in understanding the relationship between the features and the target variable, thus knowning the most impactful mutations and copy number variations.

##### Example Data Structure:
	•	X_matrix  : Input features (mutations + copy number variations)
	•	y_[phenotype] : Phenotype of interest

## Main Features
The project includes the following functionalities:

##### Data Preprocessing:
1. Normalize Data on Copy number variations
2. Remove Correlated Features on Copy number variations
3. PCA on Copy number variations
4. Shuffling  
5. Min-Max on y_[phenotype] (Optional)  

These preprocessing steps collectively decrease dimension, to facilitate model application. This thorough data preparation process aids in achieving robust and reliable predictions.

##### ML Model Implementation:
Various machine learning algorithms were implemented and trained on the dataset.    
•	BERT
•   ElasticNet
•   GBM
•   LASSO
•   LightGBM
•   Neural Networks
•   Random Forest
•   XGBoost
•   SVM  

##### Model Evaluation and Understanding: 
We calculate with every model the loss (MSE), the R2 score (good for understanding goodness of fit), the Shap score (good for understanding feature importance and correlation), and show the 50 most influent features. Then, we compare them to keep the frequent features.


## Usage
You must choose the model you want to try on the dataset, and run every cell until obtaining results that are immediately stored.
For the 3rd step : "initialization of the parameters for model training", you can choose the cell to run according to the model that you would like to use for making your predictions.

1. Extracting data:   
All original data are in the `data/` repository. You must choose the phenotype you are interested in (e.g. `YPD_doublingtime`). To extract the X_matrix_[phenotype] and y_[phenotype], you must run all the cells in the `extract_data_ipynb.ipynb`. You will finally get phenotype_dataset, copy_number_variation_dataset, mutations_dataset, extended_mutations_[phenotype], X_matrix_[phenotype], and y_[phenotype], as well as some plots to describe the datasets, which can be found in `results/plots/` repository.

2. (Optional) Preprocessing data: 
Before training a model, data must be preprocessed. This includes:   
•	Normalization of data using `scale_last_columns()`.
•	Removing Correlated Features with `remove_low_variance_features_last_columns()`.
•	Applying PCA with `apply_pca_last_columns()`.
•	Shuffling with `shuffle_dataset()`.
•	(Optional) Applying Min-Max to y_[phenotype] with `y_preprocessing()`.

All the preprocessing is done by calling the function `preprocessed_data()`, found in the file `preprocessing.py`, called at the begining of many `model_[model].ipynb` files.

3. Training Models:    
All `model_[model].ipynb` models can be run. They all use cross-validation to find the best parameters to find the most accurate model. Some use hyper-parametrization.

4. Results:  
After training, you can save your model and Shap results in the `results/`repository using `save_and_compare_results()` function found in the file of the same name. You can also make different graphes using `matplotlin.pyplot`. You can also write your top 10 features in the `analyse.txt` file.

## Possible improvements  
• Implement other classification models such as SVM or Random Forest.  
• Add additional performance metrics to evaluate models more precisely.  
• Optimize execution time using techniques like batch processing or mini-batch gradient descent.


