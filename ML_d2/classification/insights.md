# Classification Insights – Student Performance

## Problem Framing
The original target variable, **Performance Index**, is continuous.
To apply classification techniques, it was transformed into three categories:

- **Low**: Performance Index < 40  
- **Medium**: 40 ≤ Performance Index < 70  
- **High**: Performance Index ≥ 70  

This categorization reflects real-world academic performance grouping.

---

## Models Implemented

### Logistic Regression
- Linear, multi-class classification model
- Feature scaling applied using `StandardScaler`
- Achieved high overall accuracy (~95%)
- Balanced precision and recall across all classes

**Observation:**  
Logistic Regression benefits from feature scaling due to its gradient-based optimization.

---

### Decision Tree
- Non-linear, rule-based classifier
- No feature scaling required
- More interpretable due to decision rules
- Controlled model complexity using `max_depth`

**Observation:**  
Decision Trees can capture non-linear patterns but may overfit if not constrained.

---

### Random Forest
- Ensemble of multiple decision trees
- Reduces overfitting compared to a single tree
- Strong and stable performance across classes

**Observation:**  
Random Forest balances interpretability and performance, making it suitable for real-world classification tasks.

---

### Support Vector Machine (SVM)
- Margin-based classifier
- Feature scaling required
- Effective in high-dimensional spaces

**Observation:**  
SVM provides strong decision boundaries but is computationally more expensive than tree-based models.

---

## Model Comparison Summary

| Aspect | Logistic Regression | Decision Tree |
|------|---------------------|---------------|
| Model Type | Linear | Non-linear |
| Feature Scaling | Required | Not Required |
| Interpretability | Medium | High |
| Overfitting Risk | Low | Moderate |
| Performance | Strong & Stable | Competitive |

---

## Key Learnings
- Problem framing influences model selection
- Preprocessing steps depend on the algorithm, not just the dataset
- Using shared preprocessing improves code maintainability
- Comparing models gives better insight than relying on a single algorithm

---

## Conclusion
Both models performed well for student performance classification.
Logistic Regression offers stability and generalization, while Decision Trees provide interpretability.
The final model choice depends on use-case priorities.
