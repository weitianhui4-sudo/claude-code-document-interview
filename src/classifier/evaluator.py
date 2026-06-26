import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix


class Evaluator:
    """
    Evaluates classifier predictions against ground-truth labels.
    Reports per-class precision/recall/F1, macro-F1, and confusion matrix.
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
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=self.label_names)

        self._print(y_true, y_pred, cm)
        return {'macro_f1': macro_f1, 'report': report, 'confusion_matrix': cm}

    def _print(self, y_true, y_pred, cm) -> None:
        print(classification_report(
            y_true, y_pred,
            target_names=self.label_names,
            zero_division=0,
        ))
        print("Confusion Matrix:")
        header = f"{'':20s}" + "".join(f"{n:>18s}" for n in self.label_names)
        print(header)
        for i, row_label in enumerate(self.label_names):
            row = f"{row_label:20s}" + "".join(f"{v:18d}" for v in cm[i])
            print(row)
        print()

    def compare(self, results: dict[str, dict]) -> None:
        print("=== Model Comparison ===")
        for name, res in results.items():
            print(f"  {name:25s}  macro-F1: {res['macro_f1']:.4f}")
