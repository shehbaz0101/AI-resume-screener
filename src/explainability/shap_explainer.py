import shap


class ShapExplainer:

    def __init__(self, model):

        self.explainer = shap.Explainer(model)

    def explain(self, features):

        shap_values = self.explainer(features)

        return shap_values