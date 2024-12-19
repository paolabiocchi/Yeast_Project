# **Machine Learning Project - Yeasts' genotype to describe their doubling time**

## **Description**
The primary objective of this project is to apply machine learning algorithms to identify the mutations in the protein sequences and copy number variations (CNVs) of each protein sequences, of each yeast genotype that have the most significant effects on their doubling time. To achieve this, we build and evaluate models that predict the yeast doubling time based on their genotype. Multiple models, including ElasticNet, GBM, LASSO, LightGBM, Neural Networks, Random Forest, XGBoost, SVM, were tested and optimized through cross-validation. This process aimed to minimize classification errors and identify the model that best highlights the most impactful mutations and copy number variations, thus providing valuable insights into yeast biology.

## **Project Structure**
The project is organized as follows:  
```
├── data/                                   # Contains data files
│ ├── Proteome_1011/                        # Contains all the fasta files corresponding to all yeasts' genotypes of each protein
│ ├── copy_number_variation_dataset.csv     # Contains all the copy number variations for all yeasts
│ ├── extended_mutations_{phenotype}.csv    # Contains all the specific mutations for all yeasts
│ ├── Finalset_223phenotypes_1011.tab       # Contains all the phenotypes for all yeasts
│ ├── genesMatrix_CopyNumber.csv            # Contains all the copy number variations for all yeasts, as a table
│ ├── mutations_dataset.csv                 # Contains all the mutations for all yeasts, as a list
│ ├── phenotype_dataset.csv                 # Contains all the phenotypes for all yeasts, for easir label??
│ ├── X_matrix_{phenotype}.csv              # Contains all the specific mutations and copy number variations for all yeasts
│ ├── y_{phenotype}.csv                     # Contains the phenotype of interest for all yeasts
│ ├── X_matrix_YPD_doublingtime_sample.csv  # Sample of X_matrix_YPD_doublingtime.csv
│ ├── ...                                   # Contains other data files (pickle, restricted...)
├── extract_data/                                       # To extract data of our interest
│ ├── extract_data_mutations.ipynb                      # To extract the matrix of mutations and y_{phenotype}
│ ├── extract_data_proteins.ipynb                       # To extract the matrix of proteins mutated
│ ├── extract_mutations_from_important_proteins.ipynb   # To extract the matrix of mutations from the important proteins
│ ├── extract_proteins_results.ipynb                    # To extract the results of models run on the matrix of mutated proteins
├── models/                         # All models
├── results/                        # Contains results files
│ ├── plots/                        # Contains all plots (CNV per protein, mutation per protein, mutation per yeast)
│ ├── ...                           # Files containing all model and Shap results  
├── analyse.txt                     # To save and compare model results
├── preprocessing.py                # To preprocess the data
├── README.md                       # Project documentation  
├── requirements.txt                # For all external libraries used
├── run.ipynb                       # To make all process of learning and predicting  
├── save_and_compare_results.py     # To save and compare model and Shap results

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
	•	y_{phenotype} : Phenotype of interest

## Main Features
The project includes the following functionalities:

##### Data Preprocessing:
1. Normalize Data on Copy number variations
2. Remove Correlated Features on Copy number variations
3. PCA on Copy number variations
4. Shuffling  
5. Min-Max on y_{phenotype} (Optional)  

These preprocessing steps collectively decrease dimension, to facilitate model application. This thorough data preparation process aids in achieving robust and reliable predictions.

##### ML Model Implementation:
Various machine learning algorithms were implemented and trained on the dataset.    
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
For the 3rd step: "Initialization of the parameters for model training", you can choose the cell to run according to the model that you would like to use for making your predictions.

1. Extracting data:   
All original data are in the `data/` repository. You must choose the phenotype you are interested in (e.g. `YPD_doublingtime`). To extract the X_matrix_{phenotype} and y_{phenotype}, you must run all the cells in the `extract_data_mutations.ipynb`. You will finally get phenotype_dataset, copy_number_variation_dataset, mutations_dataset, extended_mutations_{phenotype}, X_matrix_{phenotype}, and y_{phenotype}, as well as some plots to describe the datasets, which can be found in `results/plots/` folder. Then, you can extract the X_matrix (proteins+CNVs) in the `extract_data_proteins.ipynb`. After that, you can extract the mutations X_matrix from these important proteins with the `extract_mutations_from_important_proteins.ipynb`.

2. (Optional) Preprocessing data: 
Before training a model, data must be preprocessed. This includes:   
•	Normalization of CNV features using `scale_last_columns()`.
•	Removing correlated CNV features with `remove_low_variance_features_last_columns()`.
•	Applying PCA to CNV features with `apply_pca_last_columns()`.
•	Shuffling with `shuffle_dataset()`.
•	(Optional) Applying Min-Max to y_{phenotype} with `y_preprocessing()`.

All the preprocessing is done by calling the function `preprocessed_data()`, found in the file `preprocessing.py`, called at the begining of many `model_[model].ipynb` files.

3. Training Models:    
All `model_[model].ipynb` models can be run. They all use cross-validation to find the best parameters to find the most accurate model. Some use hyper-parametrization.

4. Results:  
After training, you can save your model and Shap results in the `results/`folder using `save_and_compare_results()` function found in the file of the same name. You can also make different graphs using `matplotlib.pyplot`. You can then write your top 10 features in the `analyse.txt` file.

## Possible improvements  
• Implement other classification models such as BERT.  
• Add additional performance metrics to evaluate models more precisely.  
• Apply models to pairs and triplets.


