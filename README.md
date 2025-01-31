# deep-learning-challenge

# Neural Network Model Analysis: Predicting Alphabet Soup Charity Success

## Introduction

The objective of this analysis was to build and optimize a neural network to predict whether a charity funded by Alphabet Soup would be successful. Despite multiple attempts at optimization, I was unable to reach the target accuracy of 75%. Below is a summary of the steps taken, challenges faced, and future recommendations.

---

## Data Preprocessing

### Target and Features

- **Target Variable**: `IS_SUCCESSFUL` (binary classification: 1 for success, 0 for failure).
- **Feature Variables**: Columns such as `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, and `ASK_AMT`.
- **Dropped Columns**: `EIN` and `NAME` were removed as they were not relevant for the prediction task.

### Preprocessing Steps

- Categorical features were encoded using `pd.get_dummies()`.
- Features were scaled using `StandardScaler()`.
- Data was split into training and testing sets using `train_test_split`.

---

## Compiling, Training, and Evaluating the Model

### Model Architecture

I created a neural network with:
- **First hidden layer**: 128 neurons (based on common practice for the number of features).
- **Second hidden layer**: 64 neurons (smaller, to distill features).
- **Output layer**: 1 neuron with a sigmoid activation function for binary classification.

I used **ReLU** activation for the hidden layers and **sigmoid** for the output layer. The model was trained for 50 epochs, but the accuracy plateaued at **73%**—below the target of 75%.

---

## Optimization Attempts

### 1. **Model Architecture Adjustments**:
I increased the number of neurons (e.g., to 256) and added more hidden layers, but this led to **overfitting** without improving accuracy (as discussed in past questions).

### 2. **Early Stopping and Epoch Adjustments**:
I added **early stopping** (as suggested), but it didn’t help in achieving better performance. The model's accuracy remained stagnant.

### 3. **Feature Engineering**:
I combined low-frequency categories into an "Other" category, but it didn’t noticeably impact the accuracy (as per my previous question regarding rare categories).

---

## Summary of Results

Despite multiple optimization attempts, including changing architecture, tuning hyperparameters, and adjusting preprocessing, the highest accuracy reached was **73%**. This indicates that further optimizations did not significantly improve the model.

---

## Alternative Approaches

Given the limitations of the neural network model, I would recommend using **Random Forest** or **Gradient Boosting (e.g., XGBoost)** models. These models are effective for classification tasks with tabular data and can handle non-linear relationships without extensive tuning. They may also provide better insights into feature importance and prevent overfitting.

---

## Conclusion

In conclusion, although I tried various optimization strategies (such as changing the network architecture and tweaking hyperparameters), I was unable to surpass a 73% accuracy. Using models like **Random Forest** or **XGBoost** may offer better performance for this task.
