import numpy as np
from src.config import QuantileConfig


def WIS_3_and_coverage(
    y_true: float,
    lower: float,
    middle: float,
    upper: float,
    quantile_levels: list[float] = QuantileConfig.quantile_levels,
) -> tuple[float, int]:
    """Weighted interval score (WIS) for K = 3, that is 3 quantile intervals, e.g. [0.05, 0.5, 0.95]
    REFs:
    (1) https://arxiv.org/pdf/2005.12881 - formula (4)
    (2) https://cran.r-project.org/web/packages/scoringutils/vignettes/metric-details.html
    (3) https://catboost.ai/en/docs/concepts/loss-functions-regression#MultiQuantile
    """
    assert (
        len(quantile_levels) == 3
    ), f"The number of quantile levels is expected = 3, but {len(quantile_levels)} were given"

    wis_list = []
    for i, y_quant in enumerate([lower, middle, upper]):
        q = quantile_levels[i]
        wis = 0  # Initialize wis for each quantile level
        if y_true <= y_quant:
            wis += (
                2 * (1 - q) * (y_quant - y_true)
            )  # one can also multiply one side (e.g. by 2) for asymmetry
        else:
            wis += 2 * q * (y_true - y_quant)

        wis_list.append(wis)

    # here we normalize by 1/(2K + 1)
    wis = 2 * sum(wis_list) / len(wis_list)

    # coverage - how many validation data fall within the confidence interval
    coverage = 1  # assume is within coverage
    if (y_true < np.minimum(upper, lower)) or (y_true > np.maximum(upper, lower)):
        coverage = 0

    return wis, coverage


def mae_score(observations, point_forecasts):
    return np.abs(observations - point_forecasts).mean()


# vectorize the function
v_WIS_3_and_coverage = np.vectorize(WIS_3_and_coverage)


def mean_WIS_3_and_coverage(
    y_true: np.array, lower: np.array, middle: np.array, upper: np.array
) -> tuple[float, int]:
    WIS_scores, coverage = v_WIS_3_and_coverage(y_true, lower, middle, upper)

    MWIS = np.mean(WIS_scores)

    coverage = coverage.sum() / coverage.shape[0]

    return MWIS, coverage


if __name__ == "__main__":
    print("plot WIS")
    import matplotlib.pyplot as plt

    # Define quantile_levels and chosen [lower, middle, upper]
    quantile_levels = QuantileConfig.quantile_levels
    # this is what we get from the model
    lower, middle, upper = 5, 10, 15

    # Generate y_true values
    y_true_values = np.linspace(0, 20, 1000)

    # Calculate WIS for each y_true value
    WIS_values_coverage = [
        WIS_3_and_coverage(y, lower, middle, upper) for y in y_true_values
    ]
    WIS_values = [val[0] for val in WIS_values_coverage]
    coverage = [val[1] for val in WIS_values_coverage]

    # Plot the function
    plt.plot(y_true_values, WIS_values, label="WIS", color="blue")
    plt.plot(y_true_values, coverage, label="Coverage", color="green", linestyle="--")

    plt.xlabel("Y_true")
    plt.ylabel("WIS and Coverage")
    plt.title("Weighted Interval Score (WIS) and Coverage")
    plt.legend()
    plt.grid(True)
    plt.show()
