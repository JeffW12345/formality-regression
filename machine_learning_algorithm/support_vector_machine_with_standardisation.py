import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from machine_learning_algorithm.algorithm_superclass import MachineLearningAlgorithm


class SVMRegressionWithStandardization(MachineLearningAlgorithm):
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1):
        super().__init__()
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon

    def train_test_and_publish(self) -> None:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)

            self.rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            self.r2_scores.append(r2_score(y_test, y_pred))

        self.publish_results()

    def publish_results(self) -> None:
        self.update_mean_squared_error_and_r_squared_in_results_object()
        self.results.has_standardisation = True
        self.results.algorithm_name = "Support Vector Machine Regression"
        self.results.print_to_spreadsheet()
