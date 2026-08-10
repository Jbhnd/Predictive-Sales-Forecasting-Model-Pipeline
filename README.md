# Predictive Sales Forecasting Model Pipeline

## Overview
This is an advanced predictive analytics initiative that implements machine learning algorithms to forecast monthly sales revenue. This initiative implements a complete machine learning lifecycle, evaluating multiple model classes (Linear Regression, Random Forest, and Gradient Boosting) to minimize error metrics, and creates a pipeline for future sales predictions.

## Business Challenge
Traditional moving-average forecasting models were failing to account for non-linear feature interactions, leading to supply chain inefficiencies and stockouts. This project builds a robust forecasting pipeline to predict future sales targets based on historical trends in pricing adjustments, location and sku specifics, and marketing spends.

## Live Prediction Using Sample Data
[Businesss Prediction Dashboard](https://predictive-sales-forecasting-model-pipeline.streamlit.app/)

## Tech Stack

* **Data Extraction:** SQL (CTEs, aggregate functions)
* **Modeling & Pipelines:** Python (scikit-learn, pandas, numpy), Sklearn (Preprocessing, Pipeline, Ensemble, Metrics), Joblib
* **Core Algorithms:** Linear Regression, Random Forest Regressor, Gradient Boosting Regressor
* **Feature Attribution:** SHAP (Global, MAE) 
* **Live Prediction:** Streamlit, CSS

## Model Evaluation & Performance Results
Evaluated multiple models using Mean Absolute Error (MAE) and R-squared ($R^2$) to identify the optimal configuration:
| Model | Description | MAE Value | MAE %|
|:-----|:-----|:-----|:-----|
| Regression (OLS)| OLS regression model|161.463 |46.921%|
| Regression (WLS)| Weighted regression model|160.015 |46.500%|
| Random Forest Regressor| Initial random forest|109.645 |31.862%|
| Gradient Boosting Regressor| Initial gradient boosting regressor|115.518 |33.569%|
| Random Forest Regressor| Optimized random forest|106.471 |31.862%|
| Gradient Boosting Regressor| Optimized gradient boosting regressor|99.317 |28.861%|
* Considering the skewness of the sales data, the MEA % is calculated on the median.

## Repository Structure & Project Flow
* **`/data`**: Holds the data for building the model, and sample data for the live app predictions.
* **`/notebooks`**: Holds the model training, hyperparameter tuning (GridSearchCV), feature importance mapping, and validation testing.
* **`/model`**: Holds the tested winning model file.
* **`app.py`**: Live app file.
* **`requirements.txt`**: Requirements for the live app.
