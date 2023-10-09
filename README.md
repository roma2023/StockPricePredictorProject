# Stock Price Prediction Project

## Overview

This project aims to predict stock price movements using machine learning models. It explores various models and techniques to achieve the highest possible precision in predicting stock price increases or decreases.

## Table of Contents

- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Models Explored](#models-explored)
- [Data](#data)
- [Usage](#usage)
- [Lessons Learned](#lessons-learned)
- [Example](#example)

## Models Explored

The project explores the following machine learning models:

1. **k-Nearest Neighbors (k-NN)**:
   - Achieved a precision of 80%.
   - Challenges in high-dimensional spaces were noted.

2. **Support Vector Machine (SVM)**:
   - Achieved a precision of 81.9%.
   - Suitable for hyperdimensional features and mid-sized datasets.

3. **Random Forest (RF)**:
   - Achieved the best precision of 82%.
   - Cross-validation ensured results were not due to chance.
   - Not suitable for boosting due to complexity.

4. **AdaBoost on SVM & Decision Tree**:
   - Boosting the RF model was deemed impractical due to complexity.
   - Decision Tree performed better individually than in combination with SVM.
   - Decision Tree achieved a precision of approximately 82%.

## Data

The dataset used in this project contains stock price details over a period of time. The dataset was preprocessed to extract relevant features for training and testing the machine learning models.

## Usage

To use the best global model for predicting stock price changes, follow these steps:

1. Train the models by running the provided Python scripts for each model.
2. Incorporate the models using the `cross_Validation_ADA()` function.
3. Use the best global model for predictions:

```python
predictions = Best_global_model.predict(live_pred_data)
print(predictions)
```

## Lessons Learned

- **k-Nearest Neighbors (k-NN)**:
  - Achieved an impressive precision of 80% but faced challenges in high-dimensional spaces.

- **Support Vector Machine (SVM)**:
  - Performed slightly better with a precision of 81.9% and is suitable for mid-sized datasets.

- **Random Forest (RF)**:
  - Proved to be the best model, achieving 82% precision. However, it's computationally complex.

- **Combining SVM and Decision Tree through AdaBoost**:
  - Did not significantly improve performance.

- **Decision Tree and SVM**:
  - Were more effective individually than in combination.

## Example

Here is an example of how to use the best global model to predict stock price changes:

```python
predictions = Best_global_model.predict(live_pred_data)
print(predictions)
```
