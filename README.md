## Task 1: Data Understanding

Q1. Why is this dataset not linearly separable?

This dataset is not linearly separable because heart disease does not depend on a single
medical factor. Multiple features such as age, cholesterol, chest pain type, and heart
rate interact with each other in complex ways. Patients with similar values for one feature
can still have different outcomes based on other features, which creates overlapping
patterns that cannot be separated using a straight line or simple linear boundary.

Q2. Which characteristics of this dataset make model selection difficult?

Model selection is difficult because the dataset contains real-world medical data with
non-linear relationships and overlapping classes. Although the dataset is not very large,
it has enough complexity that simple models may underfit, while more complex models may
overfit. Different models handle this trade-off differently, making it challenging to
choose the best model without careful evaluation.

## Task 5:  THINKING TASK


Q1. If two models give similar validation accuracy, how would you decide which one to deploy in a real production system?

When two models have similar validation accuracy, accuracy alone is not enough to choose a model for deployment. In a real system, especially for medical data like heart disease prediction, interpretability and reliability become critical. A model that is easier to understand, such as a constrained decision tree, is often preferred because its decisions can be explained to doctors and stakeholders.Another important factor is stability. A model with a smaller train–validation gap is more likely to generalize well to new data. Computational efficiency also matters: for example, KNN can be slow during prediction because it compares each new sample to the entire training dataset. Most importantly, the cost of errors must be considered. In heart disease prediction, false negatives are more dangerous than false positives. Therefore, metrics like recall and sensitivity, along with domain requirements, play a key role in deciding which model to deploy, not accuracy alone.

Q2. Why is accuracy often a misleading metric? Explain using this dataset as context.

Accuracy can be misleading because it does not account for the type of errors a model makes. In the heart disease dataset, predicting a patient as healthy when they actually have heart disease is far more serious than predicting disease when none exists. Accuracy treats both mistakes equally, which hides this risk.A model can achieve high accuracy while still performing poorly on critical cases, such as patients with early or borderline symptoms. This makes accuracy insufficient for evaluating real-world usefulness. Accuracy also ignores model stability. In earlier tasks, some models achieved perfect training accuracy but showed signs of overfitting. Without comparing training and validation performance, accuracy alone can give a false sense of confidence. For this dataset, metrics like recall, precision, and validation trends are essential to properly evaluate model performance beyond accuracy.
