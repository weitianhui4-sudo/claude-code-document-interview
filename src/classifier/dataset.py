import numpy as np
import pandas as pd
from pathlib import Path

LABEL_NAMES = ['Compliance Risk', 'Contract Breach', 'Neutral']
DEFAULT_PATH = Path('data/synthetic_dataset.csv')


class SyntheticDataset:
    """
    Generates a synthetic imbalanced legal-document dataset that mirrors
    the real task structure (HTML noise, system artifacts, class skew).

    Class distribution: Neutral ~80%, Compliance Risk ~13%, Contract Breach ~7%
    """

    _TEMPLATES = {
        'Compliance Risk': [
            "<p>The entity failed to adhere to &amp; regulatory requirements under Section {s}.</p> ||NULL||",
            "ALERT: Compliance breach detected in fiscal period {s}. ===",
            "<div>Non-compliance with AML directives noted. Record ID: {s} �</div>",
        ],
        'Contract Breach': [
            "<b>Party A has materially breached clause {s} of the agreement.</b>",
            "CONTRACT VIOLATION\x00: delivery obligation unmet as of period {s}. ***",
            "Breach of warranty under &lt;Section {s}&gt; confirmed by legal review.",
        ],
        'Neutral': [
            "The quarterly report for period {s} has been filed with the registrar.",
            "<p>Standard operating procedure {s} reviewed and approved. N/A</p>",
            "Document {s} archived. No further action required. []",
        ],
    }

    def __init__(
        self,
        n: int = 10_000,
        class_probs: tuple = (0.13, 0.07, 0.80),
        seed: int = 42,
    ):
        self.n = n
        self.class_probs = class_probs
        self.seed = seed

    def save(self, path: Path = DEFAULT_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.generate().to_csv(path, index=False)
        print(f"Dataset saved to {path}")
        return path

    @staticmethod
    def load(path: Path = DEFAULT_PATH) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No dataset found at {path}. Run SyntheticDataset().save() first."
            )
        df = pd.read_csv(path)
        print(f"Dataset loaded from {path}  ({len(df):,} rows)")
        return df

    def generate(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        labels = rng.choice(LABEL_NAMES, size=self.n, p=self.class_probs)

        records = []
        for i, label in enumerate(labels):
            templates = self._TEMPLATES[label]
            tmpl = templates[i % len(templates)]
            records.append({
                'text': tmpl.format(s=rng.integers(100, 999)),
                'label': label,
            })

        return pd.DataFrame(records)
