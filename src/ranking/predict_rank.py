class RankPredictor:

    def __init__(self, model):

        self.model = model


    def rank(self, features):

        scores = self.model.predict(features)

        return scores