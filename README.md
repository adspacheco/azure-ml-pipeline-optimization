# Optimizing an ML Pipeline in Azure

## Summary of the problem statement
This project aims to optimize a Machine Learning pipeline in the Azure environment. The central challenge is to construct and optimize a pipeline capable of developing a highly accurate predictive model. 

## How the problem was solved
To achieve this, two main approaches are adopted: the first involves using HyperDrive for hyperparameter optimization of a Scikit-learn-based model, and the second employs Azure's AutoML for automatic model selection and optimization.

### Architecture
The project is structured within an Azure ML pipeline architecture, integrating various stages of data processing, hyperparameter optimization, and modeling. Azure services such as Azure ML Studio are utilized for experiment management, data storage, and model execution for both HyperDrive and AutoML.

### Data
The dataset used consists of specific information (such as demographic features, responses to marketing campaigns, etc.) aimed at predicting a particular outcome (e.g., customer response to a marketing campaign). The data is preprocessed for handling missing values, normalization, and encoding of categorical variables.

### Hyperparameters

For hyperparameter optimization with HyperDrive, the focus is on tuning key parameters of the Scikit-learn model, such as 'C' (regularization strength) and 'max_iter' (maximum number of iterations). We employ a random sampling approach to explore the hyperparameter space, and the Bandit policy as an early stopping strategy to conserve resources.

### Classification Algorithm

The base model is a logistic regression classifier, chosen for its simplicity and effectiveness in binary classification problems. In AutoML, a wider range of classification algorithms is automatically tested and compared, including decision trees, model ensembles, and neural networks, among others.

## Optimization Parameters

In the context of HyperDrive optimization, we focus on fine-tuning specific parameters to achieve superior machine learning performance.


### Regularization Strength (--C):

- Value Range: Between 0.05 and 0.1
- Controls the strength of regularization in logistic regression models. Lower values indicate weaker regularization, while higher values increase regularization, preventing overfitting.

### Maximum Number of Iterations (--max_iter):

- Available Values: 25, 100, 200
- Sets the maximum number of iterations allowed for the convergence of the logistic regression model. Variations in this parameter affect the trade-off between model accuracy and training time.

### Early Stopping Policy (--BanditPolicy):

- Slack Factor: 0.1
- Evaluation Interval: 2
- This policy monitors the performance of models relative to the leading model during the optimization process. If a model is not within 10% of the leader's performance after 2 evaluations, the process is stopped early to conserve resources.

## AutoML 

The AutoML-generated model is an ensemble (PreFittedSoftVotingClassifier) that combines various algorithms to achieve optimal binary classification. 

Key hyperparameters, such as regularization strength (C), learning rate (eta), maximum depth (max_depth), and others, vary for different base learners, including XGBoost and LightGBM models. 

These hyperparameters are automatically tuned to maximize predictive accuracy. 

The ensemble model intelligently assigns weights to each base learner based on their performance, resulting in an accurate and robust binary classification model.

## Comparison
Both models perform exceptionally well, with the AutoML-generated model achieving a slightly higher accuracy of 0.9160 compared to the HyperDrive-optimized model's accuracy of 0.9124.

The HyperDrive-optimized model relies on logistic regression with manually tuned hyperparameters (C=0.0781, max_iter=200).

In contrast, the AutoML-generated model is an ensemble that combines various algorithms and optimizes hyperparameters automatically for each algorithm. It achieves its best performance with a regularization strength (C) of 51.7947 and max iterations of 100 for the ensemble.

Overall, the AutoML approach demonstrates the power of automated model selection and hyperparameter tuning, resulting in a slightly better-performing model in this scenario.

## Improvements
**Class Balancing:** The data has an imbalanced class distribution, which can lead to model bias. We should explore methods like oversampling the minority class or using different evaluation metrics.

**Data Quality:** Cleaning data for missing values, outliers, and inconsistencies is crucial.

**Feature Engineering:** Reviewing and enhancing features may boost predictive accuracy.
