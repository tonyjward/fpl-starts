"""Turns the six raw predictors into the model matrix.

- continuous: z = (x - mean) / sd, with mean and sd learned from the training
  rows only; a missing value is replaced by the training mean, so z = 0 and it
  contributes nothing to the logit;
- binary flags: 0/1, unscaled;
- categorical: one 0/1 column per non-reference level, in the fixed order
  from `spec.CATEGORICAL`; the reference level is all zeros.

`fit` is the only place any statistic is learned. `transform` is a pure
function of those statistics and its input, so transforming validation,
test or prediction rows can never change what was learned.
"""

import numpy as np
import pandas as pd

from . import spec


class Preprocessor:
    def __init__(self, means=None, sds=None):
        self.means = dict(means) if means else None
        self.sds = dict(sds) if sds else None

    @property
    def fitted(self):
        return self.means is not None

    def fit(self, df):
        self.means, self.sds = {}, {}
        for c in spec.CONTINUOUS:
            values = pd.to_numeric(df[c], errors="coerce")
            mean = values.mean()
            if pd.isna(mean):
                raise ValueError("cannot fit {0}: no non-missing training values".format(c))
            filled = values.fillna(mean)
            sd = float(filled.std(ddof=0))
            self.means[c] = float(mean)
            self.sds[c] = sd if sd > 0 else 1.0
        return self

    def transform(self, df):
        if not self.fitted:
            raise RuntimeError("Preprocessor.transform called before fit")
        cols = {}
        for feature, (reference, levels) in spec.CATEGORICAL.items():
            values = df[feature]
            unknown = set(values.unique()) - set(levels) - {reference}
            if unknown:
                raise ValueError("unexpected {0} level(s): {1}".format(feature, sorted(unknown)))
            for level in levels:
                cols["{0}__{1}".format(feature, level)] = (values == level).astype(float).values
        for c in spec.CONTINUOUS:
            values = pd.to_numeric(df[c], errors="coerce").fillna(self.means[c])
            cols[c] = ((values - self.means[c]) / self.sds[c]).values
        for c in spec.BINARY:
            values = pd.to_numeric(df[c], errors="coerce")
            if values.isna().any() or not values.isin([0, 1]).all():
                raise ValueError("{0} must be 0/1 with no missing values".format(c))
            cols[c] = values.astype(float).values
        out = pd.DataFrame(cols, index=df.index)
        return out[spec.transformed_feature_names()]

    def to_dict(self):
        return {
            "continuous": {c: {"training_mean": self.means[c], "training_sd": self.sds[c],
                               "missing_value": "training_mean"} for c in spec.CONTINUOUS},
            "binary": {c: {"encoding": "0/1, unscaled"} for c in spec.BINARY},
            "categorical": {f: {"reference": ref, "levels": list(levels),
                                "encoding": "one 0/1 column per non-reference level"}
                            for f, (ref, levels) in spec.CATEGORICAL.items()},
        }

    @classmethod
    def from_dict(cls, d):
        cont = d["continuous"]
        return cls(means={c: v["training_mean"] for c, v in cont.items()},
                   sds={c: v["training_sd"] for c, v in cont.items()})


def model_frame(panel):
    """The raw predictor columns, in spec order, for rows of a panel."""
    missing = [c for c in spec.RAW_FEATURES if c not in panel.columns]
    if missing:
        raise KeyError("panel lacks raw features: {0}".format(missing))
    return panel[spec.RAW_FEATURES]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))
