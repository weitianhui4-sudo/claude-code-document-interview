import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


class Evaluator:
    """
    Evaluates classifier predictions against ground-truth labels.
    Reports per-class precision, recall, F1, and macro averages.
    """

    def __init__(self, label_names: list[str]):
        self.label_names = label_names

    def evaluate(self, y_true: list[str], y_pred: np.ndarray) -> dict:
        report = classification_report(
            y_true, y_pred,
            target_names=self.label_names,
            output_dict=True,
            zero_division=0,
        )
        metrics = {
            'macro_f1':       f1_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_recall':    recall_score(y_true, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=self.label_names),
            'report': report,
        }
        self._print(y_true, y_pred, metrics)
        return metrics

    def _print(self, y_true, y_pred, metrics: dict) -> None:
        print(classification_report(
            y_true, y_pred,
            target_names=self.label_names,
            zero_division=0,
        ))

        cm = metrics['confusion_matrix']
        print("Confusion Matrix:")
        header = f"{'':20s}" + "".join(f"{n:>18s}" for n in self.label_names)
        print(header)
        for i, row_label in enumerate(self.label_names):
            row = f"{row_label:20s}" + "".join(f"{v:18d}" for v in cm[i])
            print(row)
        print()

    def compare(self, results: dict[str, dict]) -> None:
        col_w = 28
        print("=== Model Comparison ===")
        header = f"{'Model':{col_w}}  {'Precision':>10}  {'Recall':>10}  {'F1 (macro)':>10}"
        print(header)
        print("-" * len(header))
        for name, res in results.items():
            print(
                f"{name:{col_w}}"
                f"  {res['macro_precision']:>10.4f}"
                f"  {res['macro_recall']:>10.4f}"
                f"  {res['macro_f1']:>10.4f}"
            )
        print()
