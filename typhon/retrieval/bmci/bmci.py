"""
Bayesian Montecarlo Integration.

Contains the `BMCI` class which implements the Bayesian Monte Carlo
Integration (BMCI) method as proposed by Evans et al. in [Evans]_.

.. [Evans] Evans, F. K. et al. Submillimeter-Wave Cloud Ice Radiometer: Simulations
   of retrieval algorithm performance. Journal of Geophysical Research 107, 2002
"""
import numpy as np

class BMCI:
    r"""
    Bayesian Monte Carlo Integration

    This class implements methods for solving Bayesian inverse problems using
    Monte Carlo integration. The method uses a data base of atmospheric states
    :math:`x_i` and corresponding  measurements :math:`\mathbf{y}_i`, which are used
    to compute integrals of the posterior distribution by means of importance sampling.

    The measurements in the database are assumed to be given by a
    :math:`n \\times m` matrix :math:`\mathbf{y}`, where n is the number of
    cases in the database. Currently only scalar retrievals are supported,
    which means that :math:`\mathbf{x}` is assumed to be :math:`n`-element vector
    containing the retrieval quantities corresponding to the observations in
    :math:`\mathbf{y}`.

    The method assumes that the measurement uncertainty can be described by
    a zero-mean Gaussian distribution with covariance matrix :math:`\mathbf{S}_o`
    so that given an ideal forward model :math:`F: \mathrm{R}^n \rightarrow \mathrm{R}^m`
    the probability of a measurement :math:`\mathbf{y}` conditional on an
    atmospheric state :math:`\mathbf{x}` is proportional to

    .. math::
        P(\mathbf{y} | \mathbf{x}) \sim \exp
        \{ -\frac{1}{2} (\mathbf{y} - F(\mathbf{x}))^T
        \mathbf{S}_o^{-1} (\mathbf{y} - F(\mathbf{x})) \}

    Attributes

        x: 1D array
           The retrieval quantity corresponding to the atmospheric states represented
           in the data base.

        y: 2D array
           The measured or simulated brightness temperatures corresponding to the
           atmospheric states represented in the data base.

        s_o: 2D array
           The covariance matrix describing the measurement uncertainty.

        n: int
           The number of entries in the retrieval database.

        m: int
           The number of channels in a single measurement :math:`\mathbf{y}_i`

        pc1: 1D array
            The eigenvector corresponding to the smallest eigenvalue of the
            observation uncertainty covariance matrix. Along this vector the
            entries in the database will be ordered, which can be used to
            accelerate the retrieval.

        pc1_e: float64
            The smallest eigenvalue of the observation uncertainty covariance
            matrix.

        pc1_proj: 1D array with n elements
            The projections of the measurements in `y` onto `pc1` along which
            the database entries are ordered.

    """
    def __init__(self, y, x, s_o):
        """
        Create a QRNN instance from a given training data base
        `y, x` and measurement uncertainty given by the covariance
        matrix `s_o`.

        Args

            y: 2D array
            The measured or simulated brightness temperatures corresponding
            to the atmospheric states represented in the data base. These
            are sorted along the eigenvector of the observation uncertainty
            covariance matrix with the smallest eigentvector.

            x: 1D array
            The retrieval quantity corresponding to the atmospheric states
            represented in the data base. These will be sorted along the
            eigenvector of the observation uncertainty covariance matrix
            with the smallest eigentvector.

            s_o: 2D array
            The covariance matrix describing the measurement uncertainty.
        """
        self.n = y.shape[0]
        self.m = y.shape[1]

        if not s_o.ndim == 2:
            raise Exception("Covariance matrix must be a 2D array.")
        if not self.m == s_o.shape[0] or not self.m == s_o.shape[1]:
            raise Exception("Covariance matrix must be square and agree "
                            + "with the dimensions of measurement vector.")

        # Covariance Matrix
        self.s_o = s_o
        self.s_o_inv = np.linalg.inv(self.s_o)


        # Eigenvalues of s
        self.y_mean  = np.mean(y, axis = 0)

        w, v = np.linalg.eig(self.s_o)

        # Columns of u are eigenvectors y^T * y (Covariance Matrix)
        u, s, _ = np.linalg.svd(np.transpose(y - self.y_mean) / np.sqrt(self.n),
                                full_matrices = False)

        inds = np.argsort(w)
        self.pc1_e = w[inds[0]] #s[0] ** 2.0
        self.pc1 = v[:, inds[0]] #u[:, 0]
        self.pc1_proj = np.dot((y - self.y_mean), self.pc1)

        indices = np.argsort(self.pc1_proj)
        self.pc1_proj = self.pc1_proj[indices]
        self.x = x[indices]
        self.y = y[indices, :]

    def __find_hits(self, y_obs, ds = 10.0):
        """
        Finds the range of indices in the database guaranteed to have a greater
        :math:`\chi^2` value than `ds`.

        Args:

            y_obs: 1D array
                   The measurement for which to compute the 
        """
        y_proj = np.dot(self.pc1, (y_obs - self.y_mean))
        s_l = y_proj - np.sqrt(ds / self.pc1_e)
        s_u = y_proj + np.sqrt(ds / self.pc1_e)
        inds = np.searchsorted(self.pc1_proj, np.array([s_l, s_u]))

        return inds[0], inds[1], inds[1] - inds[0]

    def weights(self, y, y_train = None):
        r"""
        Compute the importance sampling weights for a given observation `y`.
        If `y_train` is given it will be used as the database observations
        for which to compute the weights. Thie can be used to reduce the lookup
        scope in order to improve computational performance.

        The weight :math:`w_i` for database entry :math:`i` and given
        observation :math:`y` is computed as:

        .. math::
            w_i = \exp \{ -\frac{1}{2} (\mathbf{y} - \mathbf{y}_i)^T
            \mathbf{S}_o^{-1}(\mathbf{y} - \mathbf{y}_i) \}

        Args:

            y(numpy.array): The observations for which to compute the weights.

            y_train(numpy.array): 2D array containing the observations from the
                                  database for which to compute the weights.
                                  Channels are assumed to be along axis 1.

        Returns:

            ws: 1D array
                Array containing the importance sampling weights.

        Return
        """
        if not y_train:
            y_train = self.y

        y = y.reshape((1,-1))
        dy = y_train - y
        ws = dy * np.dot(dy, self.s_o_inv)
        ws = np.exp(-0.5 * ws.sum(axis=1, keepdims=True))
        return ws

    def predict(self, y_obs, x2_max = -1.0):
        """
        This performs the BMCI integration to approximate the mean and variance
        of the posterior distribution:

        .. math::
            \int x p(x | \mathbf{y}) \: dx \approx
            \sum_i \frac{w_i(\mathbf{y}) x}{\sum_j w_j(\mathbf{y})}


            \int x^2 p(x | \mathbf{y}) \: dx \approx
            \sum_i \frac{w_i(\mathbf{y}) x^2}{\sum_j w_j(\mathbf{y})}

        If the keyword arg `x2_max` is provided, then the weights will be
        computed exluding database entries that are guaranteed to have a
        :math:`\Chi^2` value higher than `x2_max`.

        Args

            y_obs: 2D array
                   The observations for which to perform the retrieval.

        Returns

            `xs`: 1D array
                  The retrieved posterior mean of the retrieval quantity
                  x.
            `vs': 1D array
                  The retrieved posterior standard deviations.

        """
        if x2_max < 0.0:
            xs = np.zeros(y_obs.shape[0])
            vs = np.zeros(y_obs.shape[0])
            for i in range(y_obs.shape[0]):
                ws = self.weights(self.y, y_obs[i,:])
                c  = ws.sum()
                print(c.shape)
                print(ws.shape)
                print(self.x.shape)
                if c > 0.0:
                    xs[i] = np.sum(self.x * ws.ravel() / c)
                    vs[i] = np.sqrt(np.sum(self.x ** 2.0 * ws.ravel() / c))
                else:
                    xs[i] = np.mean(self.x)
                    vs[i] = np.std(self.x)
                if i % 10 == 0:
                    print("progress: " + str(i))
            return xs, vs
        else:
            xs = np.zeros(y_obs.shape[0])
            for i in range(y_obs.shape[0]):
                i_l, i_u, n_hits = self.__find_hits(y_obs[i, :], x2_max)
                if n_hits > 0:
                    ws = self.weights(self.y[i_l : i_u, :], y_obs[i,:])
                    c  = ws.sum()
                    xs[i] = np.sum(self.x[i_l : i_u] * ws.ravel() / c)
                    vs[i] = np.sqrt(np.sum(self.x[i_l : i_u]**2 * ws.ravel() / c))
                else:
                    xs[i] = np.float('nan')
                    vs[i] = np.float('nan')
            return xs, vs

    def predict_quantiles(self, y_obs, taus):
        r"""
        This estimates the quantiles given in `taus` by approximating
        the CDF of the posterior as

        .. math::

            F(x) & = \int_{-\infty}^{x} p(x') \: dx'
            \sim \sum_{x_i < x} w_i

        and then interpolating :math:`F^{-1}` to the requested quantiles.

        Args:

            y_obs(numpy.array): `m`-times-`n`` matrix containing the `m`
                                 observations with `n`` channels for which to
                                 compute the percentiles.

            taus(numpy.array): 1D array containing the `k` quantiles
                               :math:`\tau \in [0,1]` to compute.

        Returns:

            A 2D numpy.array with shape `(m, k)` array containing the estimated
            quantiles.

        """
        xs = np.zeros((y_obs.shape[0], taus.size))
        for i in range(y_obs.shape[0]):
            ws = self.weights(y_obs[i,:])
            ws_sum = np.cumsum(ws)
            ws_sum /= ws_sum[-1]
            xs[i, :] = np.interp(taus, ws_sum, self.x.ravel())
        return xs
