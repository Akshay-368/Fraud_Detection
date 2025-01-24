# Fraud_Detection

This project implements various machine learning algorithms to detect fraudulent transactions in credit card data, inspired by the research paper:

**Credit Risk Analysis Using Machine Learning Classifiers**  
Conference Paper · August 2017  
DOI: [10.1109/ICECDS.2017.8389769](https://doi.org/10.1109/ICECDS.2017.8389769)

### Authors:
- *Trilok Pandey* - Vellore Institute of Technology  
- *Satchidananda Dehuri* - Fakir Mohan University

You can explore discussions, stats, and author profiles for the publication on [ResearchGate](https://www.researchgate.net/publication/325983636).

## Description
This project utilizes various machine learning classifiers to predict fraudulent transactions in credit card transaction data. The dataset contains multiple features related to transactions, and the goal is to predict whether a transaction is fraudulent (`Class` = 1) or valid (`Class` = 0).

## Models Implemented
- Naive Bayes
- Decision Trees
- Random Forest
- Gradient Boosting
- Logistic Regression
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)
- Linear Regression (adapted for classification)
- Deep Learning (using Keras)

## Dataset
The dataset used is the "creditcard.csv" file, which has been compressed due to its large size. The compressed file is in `.parquet.gzip` format. To use it, you must unzip the file and load it with `pandas` using `pd.read_parquet()` instead of the usual `pd.read_csv()`.

## Requirements
- Python 3.x
- `pandas`
- `numpy`
- `sklearn`
- `matplotlib`
- `seaborn`
- `tensorflow` (for deep learning)

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fraud_detection_model.git
   ```
2. Unzip the dataset and place the `.parquet.gzip` file in the project folder.
3. Install the necessary libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Modify the code where the dataset is loaded:
   ```python
   data = pd.read_parquet('creditcard.parquet.gzip')
   ```
5. Run the model training:
   ```bash
   python train.py
   ```

This will train the models and evaluate them based on performance metrics such as accuracy, MCC, precision, recall, and F1-score.

## Evaluation Metrics
- **Matthews Correlation Coefficient (MCC)**
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

## Visualizations
The project also provides visualizations such as:
- Heatmap of correlations between features.
- Comparison bar plots of model performance.
- Radar chart comparing various evaluation metrics.
