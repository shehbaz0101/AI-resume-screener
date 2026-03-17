import lightgbm as lgb


class RankModel:

    def __init__(self):

        self.model = lgb.LGBMRegressor()


    def train(self, X, y):

        self.model.fit(X, y)


    def predict(self, X):

        return self.model.predict(X)