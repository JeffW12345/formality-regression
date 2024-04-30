from machine_learning_algorithm.linear_regression_no_standardisation_or_normalisation import \
    LinearRegressionNoStandardisationOrNormalisation

from machine_learning_algorithm.polynomial_regression_no_standardisation_or_normalisation import \
    PolynomialRegressionNoStandardisationOrNormalisation

from machine_learning_algorithm.random_forest_no_standardisation_or_normalisation import \
    RandomForestRegressionNoStandardisationOrNormalisation

from machine_learning_algorithm.support_vector_machine_no_standardisation_or_normalisation import \
    SVMRegressionNoStandardisationOrNormalisation

from machine_learning_algorithm.support_vector_machine_with_standardisation import SVMRegressionWithStandardization
from machine_learning_algorithm.support_vector_with_machine_normalisation import SVMRegressionWithNormalization


def perform_and_publish_tests() -> None:
    algorithms = [
        LinearRegressionNoStandardisationOrNormalisation(),

        PolynomialRegressionNoStandardisationOrNormalisation(),

        RandomForestRegressionNoStandardisationOrNormalisation(),

        SVMRegressionNoStandardisationOrNormalisation(),
        SVMRegressionWithNormalization(),
        SVMRegressionWithStandardization()
    ]

    for algorithm in algorithms:
        algorithm.train_test_and_publish()


if __name__ == '__main__':
    perform_and_publish_tests()
