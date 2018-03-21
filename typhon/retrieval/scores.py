import numpy as np

r"""
This module contains several functions for the computations of scores that
are commonly used to assess the quality of a prediction :math:`y` with respect to
a reference value :math:`y_{true}`.
"""
def mape(y_pred, y_test):
    r"""
    The Mean Absolute Percentage Error (MAPE)

    The MAPE is computed as the mean of the absolute value of the relative
    error in percent, i.e.:

    .. math::

        MAPE(\mathbf{y}, \mathbf{y}_{true}) =
        \sum_{i = 0}^n \frac{|(\mathbf{y})_i - (\mathbf{y}_{true})_i|}
             {|(\mathbf{y}_{true})_i|}

    Arguments:

        y_pred(numpy.array): The predicted/estimated values.

        y_test(numpy.array): The true values.

    Returns:

        The MAPE for the given predictions.

    """
    return np.nanmean(100.0 * np.abs(y_test - y_pred.ravel()) / np.abs(y_test).ravel())

def bias(y_pred, y_test):
    r"""
    The mean bias in percent.

    Arguments:

        y_pred(numpy.array): The predicted/estimated values.

        y_test(numpy.array): The true values.

    Returns:

        The mean of the signed bias in percent.

    """
    return np.mean(100.0 * y_test - y_pred / y_test)

def quantile_score(y_tau, y_test, taus):
    r"""
    The quantile score for a given quantile :math:`\tau`, which assesses how
    well the predicted quantiles :math:`y_\tau` in `y_tau` estimate the
    quantiles of the conditional distribution :math:`P(y | x)` of which
    the values :math:`y_{true}` in `y_test` are materializations:

        .. math::

            L_\tau(y_\tau, y_{true}) = (y_\tau - y_{true})
            \cdot (\tau - \mathrm{1}_{y < y_{true}})

    Arguments:

        y_tau(numpy.array): Numpy array with shape (n, m) containing the m
                            predicted quantiles with the n different predictions along
                            the first axis.

        y_test(numpy.array): Numpy array with the n scalar materializations of the
                             conditional distributions whose quantiles are
                             estimated by the elements in `y_tau`

        taus(numpy.array): Numpy array containing the quantile values
                           :math:`\tau` that are estimated by the columns in
                           `y_tau`.

    Returns:

        The n-times-m array containing the quantiles scores for each of the estimates
        and estimated quantiles.

    Raises:

        ValueError
            If the shapes of `y_tau`, `y_test` and `taus` are inconsistent.
    """
    taus = np.asarray(taus)
    m = taus.size

    y_tau = y_tau.reshape(-1, m)
    n = y_tau.shape[0]

    try:
        y_test = y_test.reshape(n, 1)
    except:
        raise ValueError("Shape of y_test is incompatible with y_tau and taus.")

    abs_1 = taus * np.abs(y_tau - y_test)
    abs_2 = (1.0 - taus) * np.abs(y_tau - y_test)

    return np.where(y_tau < y_test, abs_1, abs_2)

def mean_quantile_score(y_tau, y_test, taus):
    r"""
    This is just a wrapper around `quantile_score` function, which
    computes the mean along the first dimension.
    """
    return np.nanmean(quantile_score(y_tau, y_test, taus), axis = 0)
