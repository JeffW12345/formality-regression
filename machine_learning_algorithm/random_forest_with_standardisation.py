import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from machine_learning_algorithm.algorithm_superclass import MachineLearningAlgorithm


class RandomForestRegressionWithStandardization(MachineLearningAlgorithm):
    def __init__(self, n_estimators=100, max_depth=None):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def train_test_and_publish(self) -> None:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        rmse_scores = []
        r2_scores = []

        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("random_forest", RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=42))
            ])

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            root_mean_squared_error = np.sqrt(mean_squared_error(y_test, y_pred))
            rmse_scores.append(root_mean_squared_error)

            r2 = r2_score(y_test, y_pred)
            r2_scores.append(r2)

        self.publish_results(r2_scores, rmse_scores)

    def publish_results(self, r2_scores, rmse_scores):
        self.results.root_mean_squared_error = np.mean(rmse_scores)
        self.results.r_squared = np.mean(r2_scores)
        self.results.has_standardisation = True
        self.results.algorithm_name = "Random Forest Regression"
        self.results.print_to_spreadsheet()
