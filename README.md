# Gaussian Process Regression Practice

A collection of small Python projects and experiments used to learn and develop practical skills with **Gaussian Process Regression (GPR)**.

This repository is a learning and experimentation space rather than a single finished machine-learning application. The projects progress from simple examples to more realistic regression problems, with a focus on understanding how Gaussian processes work in practice and how they can be applied to scientific and engineering data.

## About

The work in this repository forms part of my development toward a larger **AI-driven battery research project**. The notebooks and datasets here are deliberately kept as practice projects so that different aspects of GPR can be tested safely before applying the methods to experimental battery data.

The main goal is to understand not only how to train a GPR model, but also:

- how different kernel functions affect predictions
- how kernel hyperparameters influence model behaviour
- why data scaling is important
- how GPR handles uncertainty
- how to work with one or more input variables
- how to make predictions outside the range of the training data
- how model behaviour changes as the amount and distribution of training data changes

## Projects and Experiments

The repository contains several independent experiments, including:

### 1. Basic GPR regression

Simple examples designed to understand the structure of a Gaussian Process Regressor and the relationship between training data, predictions and uncertainty.

### 2. Synthetic multi-variable regression

Artificial datasets with multiple input variables and a single output variable are used to practise modelling a response that depends on more than one independent variable.

These examples are particularly useful for understanding how GPR can model nonlinear relationships in multidimensional input spaces.

### 3. Weather prediction experiments

GPR is applied to real-world weather data to investigate how a model can learn trends over time and make predictions beyond the available observations.

The experiments also explore the effect of periodic kernels and different kernel choices when the underlying data contains repeating behaviour.

### 4. Kernel experiments

Different kernels are tested to understand how assumptions about smoothness, periodicity and noise influence the resulting Gaussian-process model.

Examples include combinations of kernels such as:

- RBF
- White Kernel
- Periodic / ExpSineSquared
- Constant Kernel

### 5. Prediction and uncertainty

The notebooks investigate both the predicted mean and the uncertainty associated with GPR predictions.

This is one of the main advantages of GPR compared with many conventional regression approaches: the model provides an estimate of uncertainty as well as a predicted value.

## Tools

The main tools used in this repository are:

- **Python**
- **NumPy** – numerical calculations
- **Pandas** – data handling
- **Matplotlib** – plotting and visualisation
- **scikit-learn** – Gaussian Process Regression and preprocessing
- **Jupyter Notebook** – interactive development and experimentation

The main model is:

```python
from sklearn.gaussian_process import GaussianProcessRegressor
```

with kernels imported from:

```python
from sklearn.gaussian_process.kernels import (
    RBF,
    WhiteKernel,
    ExpSineSquared,
    ConstantKernel
)
```

scikit-learn's `GaussianProcessRegressor` implements GPR and supports kernel-based modelling, prediction uncertainty and optimisation of kernel hyperparameters. 

## Typical GPR Workflow

The experiments generally follow this workflow:

```text
Dataset
   ↓
Select input variables (X)
   ↓
Select target variable (y)
   ↓
Scale / preprocess data where appropriate
   ↓
Choose a kernel
   ↓
Create GaussianProcessRegressor
   ↓
Fit model to training data
   ↓
Predict mean and uncertainty
   ↓
Visualise results
   ↓
Evaluate and modify the model
```

A simple example is:

```python
kernel = 1.0 * RBF(1.0) + WhiteKernel(0.1)

gpr = GaussianProcessRegressor(
    kernel=kernel,
    random_state=42
)

gpr.fit(X_train, y_train)

mean_prediction, std_prediction = gpr.predict(
    X_test,
    return_std=True
)
```

## Why GPR?

Gaussian Process Regression is particularly useful for this work because it can model nonlinear relationships while also providing an estimate of predictive uncertainty.

That uncertainty is especially important for scientific and engineering applications, where knowing **how confident a model is** can be as important as the predicted value itself.

The approach is therefore relevant to the eventual battery project, where experimental measurements are limited and intelligently selecting useful experiments may be more valuable than simply maximising the number of measurements.

## Learning Objectives

The main objectives of this repository are to develop practical understanding of:

1. Gaussian processes and probabilistic regression
2. Kernel functions and kernel composition
3. Hyperparameter optimisation
4. Feature scaling
5. Multivariable regression
6. Extrapolation and interpolation
7. Predictive uncertainty
8. Model validation and visualisation
9. Applying GPR to scientific datasets

## Relationship to the Battery AI Project

This repository should be considered **preparatory work** for the wider AI battery project.

The experiments here are not intended to claim a battery-specific result. Instead, they provide a controlled environment in which GPR techniques can be developed and tested before being applied to experimental battery data.

The eventual objective is to use machine-learning techniques to help understand and optimise relationships between experimental battery parameters and measured performance.

## Status

**Status: Active learning / experimentation**

The notebooks are expected to change as my understanding of Gaussian Processes develops. Some experiments are intentionally simple or exploratory and are retained because they document the learning process.

## References

The implementation is based primarily on the Gaussian Process tools provided by scikit-learn:

- scikit-learn Gaussian Process Regression documentation:
  https://scikit-learn.org/stable/modules/gaussian_process.html

A useful conceptual reference for Gaussian processes is:

- C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine Learning*, MIT Press, 2006:
  https://gaussianprocess.org/gpml/

## Author

Emile Chhowalla
