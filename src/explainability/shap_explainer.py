"""SHAP explainability for the LightGBM ranking model."""
import logging
import numpy as np
import shap

logger = logging.getLogger(__name__)


class ShapExplainer:
    def __init__(self, model) -> None:
        # FIX: shap.Explainer() fails for LightGBM — must use TreeExplainer
        self._explainer = shap.TreeExplainer(model)

    def explain(self, features: np.ndarray) -> np.ndarray:
        """Return SHAP values for the given feature matrix.

        Args:
            features: 2D numpy array of shape (n_samples, n_features).

        Returns:
            SHAP values array of same shape.
        """
        shap_values = self._explainer.shap_values(features)
        logger.debug("SHAP values computed for %d samples", features.shape[0])
        return shap_values
