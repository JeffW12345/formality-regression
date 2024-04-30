import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from machine_learning_algorithm.algorithm_superclass import MachineLearningAlgorithm


class SVMRegressionWithNormalization(MachineLearningAlgorithm):
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1):
        super().__init__()
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon

    def train_test_and_publish(self) -> None:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        rmse_scores = []
        r2_scores = []

        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)

            mse = mean_squared_error(y_test, y_pred)
            rmse_scores.append(np.sqrt(mse))
            r2_scores.append(r2_score(y_test, y_pred))

        self.publish_results(r2_scores, rmse_scores)

    def publish_results(self, r2_scores: list, rmse_scores: list) -> None:
        self.results.root_mean_squared_error = np.mean(rmse_scores)
        self.results.r_squared = np.mean(r2_scores)
        self.results.has_normalisation = True
        self.results.algorithm_name = "Support Vector Machine Regression"
        self.results.print_to_spreadsheet()
