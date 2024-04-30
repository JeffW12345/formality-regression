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

    def train_and_test(self) -> None:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

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
            self.rmse_scores.append(root_mean_squared_error)

            r2 = r2_score(y_test, y_pred)
            self.r2_scores.append(r2)

    def publish_results(self) -> None:
        self.update_mean_squared_error_and_r_squared_in_results_object()
        self.results.has_standardisation = True
        self.results.algorithm_name = "Random Forest Regression"
        self.results.print_to_spreadsheet()
