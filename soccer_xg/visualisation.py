import matplotlib.pyplot as plt
import matplotsoccer as mps
import numpy as np
import numpy.ma as ma
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import auc, roc_curve
from soccer_xg import metrics


def plot_calibration_curve(
    y_true,
    y_pred,
    name='Calibration curve',
    min_samples=None,
    axis=None,
    **kwargs,
):
    """Plot the validation data.

    Parameters
    ----------
    axis : matplotlib.pyplot.axis object or ``None`` (default=``None``)
        If provided, the validation line will be overlaid on ``axis``.
        Otherwise, a new figure and axis will be generated and plotted on.
    **kwargs
        Arguments to ``axis.plot``.

    Returns
    -------
    matplotlib.pylot.axis
        The axis the plot was made on.

    Raises
    ------
    NotFittedError
        If the model hasn't been fit **and** validated.
    """

    if axis is None:
        axis = plt.figure(figsize=(5, 5)).add_subplot(111)

    axis.set_title(name)
    axis.plot([0, 100], [0, 100], ls='--', lw=1, color='grey')
    axis.set_xlabel('Predicted probability')
    axis.set_ylabel('True probability in each bin')
    axis.set_xlim((0, 100))
    axis.xaxis.set_major_locator(MultipleLocator(20))
    axis.xaxis.set_minor_locator(MultipleLocator(10))
    axis.set_ylim((0, 100))
    axis.yaxis.set_major_locator(MultipleLocator(20))
    axis.yaxis.set_minor_locator(MultipleLocator(10))
    # axis.set_aspect(1)
    axis.grid(which='both')

    (
        sample_probabilities,
        predicted_pos_percents,
        num_plays_used,
    ) = metrics.bayesian_calibration_curve(y_true, y_pred)

    if min_samples is not None:
        axis.plot(
            sample_probabilities,
            predicted_pos_percents,
            c='c',
            alpha=0.3,
            **kwargs,
        )
        sample_probabilities = ma.array(sample_probabilities)
        sample_probabilities[num_plays_used < min_samples] = ma.masked
        predicted_pos_percents = ma.array(predicted_pos_percents)
        predicted_pos_percents[num_plays_used < min_samples] = ma.masked

    max_deviation = metrics.max_deviation(
        sample_probabilities, predicted_pos_percents
    )
    residual_area = metrics.residual_area(
        sample_probabilities, predicted_pos_percents
    )

    axis.plot(
        sample_probabilities,
        predicted_pos_percents,
        c='c',
        label='Calibration curve\n(area = %0.2f, max dev = %0.2f)'
        % (residual_area, max_deviation),
        **kwargs,
    )

    axis.legend(loc='lower right')

    ax2 = axis.twinx()
    ax2.hist(
        y_pred * 100,
        bins=np.arange(0, 101, 1),
        density=True,
        alpha=0.4,
        facecolor='grey',
    )
    ax2.set_ylim([0, 0.2])
    ax2.set_yticks([0, 0.1, 0.2])

    plt.tight_layout()
    return axis


def plot_roc_curve(y_true, y_prob, name='Calibration curve', axis=None):

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    if axis is None:
        axis = plt.figure(figsize=(5, 5)).add_subplot(111)

    axis.plot(
        fpr, tpr, linewidth=1, label='ROC curve (area = %0.2f)' % roc_auc
    )

    # reference line, legends, and axis labels
    axis.plot([0, 1], [0, 1], linestyle='--', color='gray')
    axis.set_title('ROC curve')
    axis.set_xlabel('False Positive Rate')
    axis.set_ylabel('True Positive Rate')
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_locator(MultipleLocator(0.20))
    axis.xaxis.set_minor_locator(MultipleLocator(0.10))
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_locator(MultipleLocator(0.20))
    axis.yaxis.set_minor_locator(MultipleLocator(0.10))
    axis.grid(which='both')

    # sns.despine()
    # plt.gca().xaxis.set_ticks_position('none')
    # plt.gca().yaxis.set_ticks_position('none')
    plt.gca().legend()

    axis.legend(loc='lower right')
    plt.tight_layout()


def plot_heatmap(model, data, axis=None):

    if axis is None:
        axis = plt.figure(figsize=(8, 10)).add_subplot(111)

    z = model.estimate(data)['xG'].values
    axis = mps.field(ax=axis, show=False)
    axis = mps.heatmap(
        z.reshape((106, 69)).T, show=False, ax=axis, cmap='viridis_r'
    )
    axis.set_xlim((70, 108))
    axis.set_axis_off()
    return axis
