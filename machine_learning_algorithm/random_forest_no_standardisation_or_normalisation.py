import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from machine_learning_algorithm.algorithm_superclass import MachineLearningAlgorithm


class RandomForestRegressionNoStandardisationOrNormalisation(MachineLearningAlgorithm):
    def __init__(self, n_estimators=100, max_depth=None):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def train_test_and_publish(self) -> None:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        mean_squared_error_scores = []
        rmse_scores = []
        r2_scores = []

        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            model = RandomForestRegressor(n_estimators=self.n_estimators,
                                          max_depth=self.max_depth,
                                          random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mean_squared_error_scores.append(mse)
            rmse_scores.append(np.sqrt(mse))

            r2_scores.append(r2_score(y_test, y_pred))

        self.publish_results(mean_squared_error_scores, r2_scores, rmse_scores)

    def publish_results(self, mean_squared_error_scores, r2_scores, rmse_scores):
        self.results.mean_absolute_error = np.mean(mean_squared_error_scores)  # Note: change to MSE if needed
        self.results.root_mean_squared_error = np.mean(rmse_scores)
        self.results.r_squared = np.mean(r2_scores)
        self.results.algorithm_name = "Random Forest Regression"
        self.results.print_to_spreadsheet()
