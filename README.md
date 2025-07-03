🧠 Task 7: Support Vector Machines (SVM)
📌 Objective
This task aims to implement Support Vector Machine (SVM) models for classification tasks using both linear and RBF kernels. The goal is to understand how SVMs work for linearly and non-linearly separable data.

🛠 Tools & Libraries Used

Python

Scikit-learn

NumPy

Pandas

Matplotlib

Seaborn

📁 Dataset
We used the Breast Cancer Wisconsin Dataset, available from sklearn.datasets.load_breast_cancer.

This dataset contains features computed from a digitized image of a breast mass. It includes 30 numeric features and a target label (0 = malignant, 1 = benign).

📋 Steps Performed
Data Loading & Inspection

Loaded the dataset using load_breast_cancer() from sklearn.

Checked the shape, data types, and class distribution.

Data Preprocessing

Standardized the features using StandardScaler for optimal SVM performance.

Performed train-test split using an 80-20 ratio.

Model Training

Trained a Linear SVM using SVC(kernel='linear').

Trained a Non-linear (RBF Kernel) SVM using SVC(kernel='rbf').

Evaluation

Evaluated both models using:

Confusion Matrix

Classification Report (Accuracy, Precision, Recall, F1-score)

Compared the results of both kernels.

Cross-Validation

Used cross_val_score() with 5-fold validation to assess model generalization.

Visualization

Optional: Reduced features with PCA to 2D and visualized decision boundaries.

📈 Results
Linear Kernel SVM Accuracy: ~96-97%

RBF Kernel SVM Accuracy: ~97-98%

RBF generally performed slightly better due to capturing non-linear patterns.

🧠 Key Concepts Learned
Difference between linear and non-linear SVM

Role of hyperparameters like C (regularization) and gamma (RBF spread)

Importance of feature scaling in distance-based models

Kernel trick and margin maximization

📂 Files Included
svm_task7.ipynb: Jupyter notebook containing the code and outputs.

README.md: This file.

(Optional) Visualizations or screenshots.

✅ Conclusion
Support Vector Machines are powerful for both linear and non-linear classification problems. With proper preprocessing and parameter tuning, they offer high accuracy and robustness, especially in binary classification settings like this one.



