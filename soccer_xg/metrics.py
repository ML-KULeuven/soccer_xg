"""A collection of metrics for evaluation xG models."""
import numpy as np
from scipy import integrate
from sklearn.neighbors import KernelDensity


def expected_calibration_error(y_true, y_prob, n_bins=5, strategy='uniform'):
    """Compute the Expected Calibration Error (ECE).

    This method implements equation (3) in [1], as well as the ACE variant in [2].
    In this equation the probability of the decided label being correct is
    used to estimate the calibration property of the predictor.

    Note: a trade-off exist between using a small number of `n_bins` and the
    estimation reliability of the ECE.  In particular, this method may produce
    unreliable ECE estimates in case there are few samples available in some bins.

    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.
    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.
    n_bins : int, default=5
        Number of bins to discretize the [0, 1] interval. A bigger number
        requires more data. Bins with no samples (i.e. without
        corresponding values in `y_prob`) will not be returned, thus the
        returned arrays may have less than `n_bins` values.
    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy used to define the widths of the bins.
        uniform
            The bins have identical widths. This corresponds to the ECE formula.
        quantile
            The bins have the same number of samples and depend on `y_prob`. This
            corresponds to the ACE formula.

    Returns
    -------
    ece : float
       The expected calibration error.

    References
    ----------
    [1]: Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger,
       On Calibration of Modern Neural Networks.
       Proceedings of the 34th International Conference on Machine Learning
       (ICML 2017).
       arXiv:1706.04599
       https://arxiv.org/pdf/1706.04599.pdf
    [2]: Nixon, Jeremy, et al.,
       Measuring calibration in deep learning.
       arXiv:1904.01685
       https://arxiv.org/abs/1904.01685

    """

    if y_prob.shape != y_true.shape:
        raise ValueError(f'Shapes must match')
    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError('y_prob has values outside [0, 1].')
    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError('Only binary classification is supported.')

    if strategy == 'quantile':  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
    elif strategy == 'uniform':
        bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    n = y_prob.shape[0]
    accs, confs, counts = _reliability(y_true, y_prob, bins)
    return np.sum(counts * np.abs(accs - confs) / n)


def _reliability(y_true, y_prob, bins):
    n_bins = len(bins) - 1
    accs = np.zeros(n_bins)
    confs = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for m in range(n_bins):
        low = bins[m]
        high = bins[m + 1]

        where_in_bin = (low <= y_prob) & (y_prob < high)
        if where_in_bin.sum() > 0:
            accs[m] = (
                np.sum((y_prob[where_in_bin] >= 0.5) == y_true[where_in_bin])
                / where_in_bin.sum()
            )
            confs[m] = np.mean(
                np.maximum(y_prob[where_in_bin], 1 - y_prob[where_in_bin])
            )
            counts[m] = where_in_bin.sum()

    return accs, confs, counts


def bayesian_calibration_curve(y_true, y_pred, n_bins=100):
    """Compute true and predicted probabilities for a calibration curve using
    kernel density estimation instead of bins with a fixed width.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True targets.
    y_prob : array-like of shape (n_samples,)
        Probabilities of the positive class.
    n_bins : float, default=100
        Number of bins to discretize the [0, 1] interval. A bigger number
        requires more data.

    Returns
    -------
    prob_true : ndarray of shape (n_bins,)
        The proportion of samples whose class is the positive class, in each
        bin (fraction of positives).
    prob_pred : ndarray of shape (n_bins,)
        The mean predicted probability in each bin.
    number_total : ndarray of shape (n_bins,)
        The number of examples in each bin.
    """
    y_true = np.array(y_true, dtype=bool)
    bandwidth = 1 / n_bins
    kde_pos = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(
        (y_pred[y_true])[:, np.newaxis]
    )
    kde_total = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(
        y_pred[:, np.newaxis]
    )
    sample_probabilities = np.linspace(0.01, 0.99, 99)
    number_density_offense_won = np.exp(
        kde_pos.score_samples(sample_probabilities[:, np.newaxis])
    ) * np.sum((y_true))
    number_density_total = np.exp(
        kde_total.score_samples(sample_probabilities[:, np.newaxis])
    ) * len(y_true)
    number_pos = (
        number_density_offense_won
        * np.sum(y_true)
        / np.sum(number_density_offense_won)
    )
    number_total = (
        number_density_total * len(y_true) / np.sum(number_density_total)
    )
    predicted_pos_percents = np.nan_to_num(number_pos / number_total, 1)

    return (
        100.0 * sample_probabilities,
        100.0 * predicted_pos_percents,
        number_total,
    )


def max_deviation(sample_probabilities, predicted_pos_percents):
    """Compute the largest discrepancy between the model and expectation.
    """
    abs_deviations = np.abs(predicted_pos_percents - sample_probabilities)
    return np.max(abs_deviations)


def residual_area(sample_probabilities, predicted_pos_percents):
    """Compute the total area under the curve of |predicted prob - expected prob|
    """
    abs_deviations = np.abs(predicted_pos_percents - sample_probabilities)
    return integrate.trapz(abs_deviations, sample_probabilities)
