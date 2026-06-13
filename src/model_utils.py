
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix

def evaluate_and_log(model, X_test, y_test, model_name="Model"):
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    auc_pr = auc(recall, precision)
    f1 = f1_score(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Reds')
    plt.title(f'Confusion Matrix: {model_name}\nAUC-PR: {auc_pr:.2f}')
    plt.show()
    return {"Model": model_name, "AUC-PR": auc_pr, "F1-Score": f1}

def get_shap_values(model, X_data):
    if hasattr(model, 'feature_names_in_'):
        X_data = X_data[model.feature_names_in_]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    if isinstance(shap_values, list):
        return explainer, shap_values[1], X_data
    return explainer, shap_values, X_data
