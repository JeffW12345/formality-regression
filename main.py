from machine_learning_algorithm.linear_regression_no_standardisation_or_normalisation import \
    LinearRegressionNoStandardisationOrNormalisation
from machine_learning_algorithm.linear_regression_with_normalisation import LinearRegressionWithNormalization
from machine_learning_algorithm.linear_regresssion_with_standardisation import LinearRegressionWithStandardization
from machine_learning_algorithm.polynomial_regression_no_standardisation_or_normalisation import \
    PolynomialRegressionNoStandardisationOrNormalisation
from machine_learning_algorithm.polynomial_regression_with_normalisation import PolynomialRegressionWithNormalization
from machine_learning_algorithm.polynomial_regression_with_standardisation import \
    PolynomialRegressionWithStandardization
from machine_learning_algorithm.random_forest_no_standardisation_or_normalisation import \
    RandomForestRegressionNoStandardisationOrNormalisation
from machine_learning_algorithm.random_forest_with_normalisation import RandomForestRegressionWithNormalization
from machine_learning_algorithm.random_forest_with_standardisation import RandomForestRegressionWithStandardization


def perform_and_publish_tests() -> None:
    algorithms = [
        LinearRegressionNoStandardisationOrNormalisation(),
        LinearRegressionWithNormalization(),
        LinearRegressionWithStandardization(),

        PolynomialRegressionNoStandardisationOrNormalisation(),
        PolynomialRegressionWithNormalization(),
        PolynomialRegressionWithStandardization(),

        RandomForestRegressionNoStandardisationOrNormalisation(),
        RandomForestRegressionWithNormalization(),
        RandomForestRegressionWithStandardization()
    ]

    for algorithm in algorithms:
        algorithm.train_test_and_publish()


if __name__ == '__main__':
    perform_and_publish_tests()
