Credit Card Fraud Detection
Overview
This project builds a machine learning pipeline to detect fraudulent credit card transactions using anonymized transaction data. The focus is on handling class imbalance, trying multiple models, and comparing their performance for fraud detection.

Dataset Source: Public credit card fraud dataset (e.g., Kaggle – anonymized PCA features, Time, Amount, Class).

Target variable:
Class = 0 → Legitimate transactionClass = 1 → Fraudulent transactionHighly imbalanced dataset with very few fraud cases compared to normal transactions.

Key Steps->

Data Loading & Exploration.
Loaded the CSV using pandas.
Checked for missing values and basic statistics using df.isnull().sum() and df.describe().

Examined class distribution with value_counts() to verify severe class imbalance.

Feature Analysis: Computed a Spearman correlation matrix to detect highly correlated features.Visualized correlations using a seaborn heatmap to understand relationships between Time, V1–V28, Amount, and Class.

Preprocessing: Selected relevant features: Time, V1–V28, and Amount.Standardized numerical features using StandardScaler to normalize the input space.

Handled class imbalance using Random Under-Sampling (RUS) from imblearn to create a balanced dataset for training.

Train–Test Split: Split the resampled data into training and testing sets using train_test_split with a 75/25 split.

Models:

1. Implemented Decision Tree Classifier: Performed manual pruning search by iterating over max_depth values (1–19) and comparing train vs test scores.Selected a shallow tree (max_depth=2) to reduce overfitting and improve generalization.Visualized the final tree using sklearn.tree.plot_tree.

2. Support Vector Machine (SVC)Trained an SVM with a sigmoid kernel on the resampled dataset.
Bagging Classifier (with KNN): Used BaggingClassifier with KNeighborsClassifier as the base estimator to reduce variance.

3. Random Forest Classifier: Trained a default RandomForestClassifier for ensemble-based classification.

Model Comparison->
1. Computed train and test accuracy for each model.
2. Compared overfitting gaps (train accuracy − test accuracy) across Decision Tree, SVM, Bagging, and Random Forest.

3. Identified the Decision Tree (with pruning) as the best-behaved model under the chosen metric in this pipeline.

Tech Stack->
Language: Python
Libraries: pandas, numpy, scikit-learn, imbalanced-learn, seaborn, matplotlib

Possible Improvements->
1. Use precision, recall, F1-score, ROC-AUC, and PR-AUC for imbalanced evaluation.
2. Apply cross-validation and hyperparameter tuning (GridSearchCV/RandomizedSearchCV).
3. Explore class weighting or SMOTE instead of only random under-sampling.Add model explainability (feature importance, SHAP, etc.).
