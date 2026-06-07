from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix
import logging

logger = logging.getLogger(__name__)

def evaluate_and_log(model, X_test, y_test, model_name="Model"):
    try:
        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        auc_pr = auc(recall, precision)
        f1 = f1_score(y_test, y_pred)

        logger.info(f"Evaluation for {model_name}: AUC-PR={auc_pr:.4f}, F1={f1:.4f}")
        return {"AUC-PR": auc_pr, "F1": f1}
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return None
