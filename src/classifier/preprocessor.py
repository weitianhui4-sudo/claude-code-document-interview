import re
import html
import numpy as np
import pandas as pd
from typing import Optional


class TextPreprocessor:
    """
    Cleans raw legal/news text:
    - Unescapes HTML entities
    - Strips HTML/XML tags
    - Removes system-generated noise (control chars, NULL tokens, BOM, etc.)
    - Normalizes whitespace
    Never drops records — non-string inputs are coerced to empty string.
    """

    _NOISE_PATTERNS = [
        r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]',  # control characters
        r'[�﻿]',                        # unicode replacement char / BOM
        r'\b(?:NULL|NULLVALUE|N/A|undefined)\b', # placeholder tokens
        r'\|\|+',                                # pipe metadata separators
        r'={3,}|-{3,}|\*{3,}',                  # decorative dividers
        r'\[\s*\]|\(\s*\)',                      # empty brackets
    ]

    def __init__(self, extra_noise_patterns: Optional[list[str]] = None):
        patterns = self._NOISE_PATTERNS + (extra_noise_patterns or [])
        self._noise_re = re.compile(
            '|'.join(patterns), re.IGNORECASE | re.UNICODE
        )
        self._whitespace_re = re.compile(r'\s+')

    def clean(self, text) -> str:
        if not isinstance(text, str):
            if text is None or (isinstance(text, float) and np.isnan(text)):
                return ""
            text = str(text)

        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = self._noise_re.sub(' ', text)
        text = self._whitespace_re.sub(' ', text).strip()
        return text

    def transform(self, df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
        """Return copy of df with a new 'clean_text' column."""
        df = df.copy()
        df['clean_text'] = df[text_col].apply(self.clean)
        return df
