# DonorsChoose Truncated SVD & XGBoost 

This project implements dimensionality reduction using **Truncated SVD** and applies **XGBoost** to predict project approval in the DonorsChoose dataset.

## Dataset
- **DonorsChoose.org** project proposals dataset.
- Contains project details, teacher information, resource requirements, and approval status.

## Key Steps
1. **Preprocessing**:
   - Cleaned and vectorized categorical features such as project categories, school states, and teacher prefixes.
   - Processed essays and project titles using TF-IDF and sentiment analysis.

2. **Dimensionality Reduction**:
   - Constructed a co-occurrence matrix from top 2,000 words in essays and titles.
   - Applied **Truncated SVD** to reduce dimensionality of the co-occurrence matrix.

3. **Modeling**:
   - Stacked all processed features (text vectors, numerical data, and sentiment scores).
   - Applied **XGBoost** classifier to predict whether a project will be approved.
   - Performed hyperparameter tuning using grid search and cross-validation.

## Results
- Achieved strong predictive performance with optimized hyperparameters.
- Generated 3D scatter plots to visualize AUC across different model configurations.

## Conclusion
Truncated SVD effectively reduced the feature space, improving model training time while maintaining accuracy in predicting project approvals.
