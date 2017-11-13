import numpy as np

class BMCI:
    def __init__(self, y, x, s_o, s_s = None):
        self.n = y.shape[0]
        self.m = y.shape[1]

        if not s_o.ndim == 2:
            raise Exception("Covariance matrix must be a 2D array.")
        if not self.m == s_o.shape[0] or not self.m == s_o.shape[1]:
            raise Exception("Covariance matrix must be square and agree "
                            + "with the dimensions of measurement vector.")

        # Covariance Matrix
        self.s = s_o
        if s_s:
            try:
                self.s += s_s
            except:
                raise Exception("Covariance matrix s_s is inconsistent with s_o.")
        self.s_inv = np.linalg.inv(self.s)


        # SV Decomposution

        self.y_mean  = np.mean(y, axis = 0)

        # Columns of u are eigenvectors y^T * y (Covariance Matrix)
        u, s, _ = np.linalg.svd(np.transpose(y - self.y_mean) / np.sqrt(self.n),
                                full_matrices = False)

        self.pc1_e = s[0] ** 2.0
        self.pc1 = u[:, 0]
        self.pc1_proj = np.dot((y - self.y_mean), self.pc1)

        indices = np.argsort(self.pc1_proj)
        self.pc1_proj = self.pc1_proj[indices]
        self.x = x[indices]
        self.y = y[indices, :]

    def weights(self, y_train, y):
        y = y.reshape((1,-1))
        dy = y_train - y
        ws = dy * np.dot(dy, self.s_inv)
        print(-0.5 * ws.sum(axis=1, keepdims=True))
        ws = np.exp(-0.5 * ws.sum(axis=1, keepdims=True))
        return ws

    def _find_hits(self, y, ds = 10.0):
        y_proj = np.dot(self.pc1, (y - self.y_mean))
        s_l = y_proj - np.sqrt(ds / self.pc1_e)
        s_u = y_proj + np.sqrt(ds / self.pc1_e)
        inds = np.searchsorted(self.pc1_proj, np.array([s_l, s_u]))
        print(inds[1] - inds[0])

        return inds[0], inds[1], inds[1] - inds[0]

    def predict(self, y_test, ds = -1.0):

        if ds < 0.0:
            xs = np.zeros(y_test.shape[0])
            for i in range(y_test.shape[0]):
                ws = self.weights(self.y, y_test[i,:])
                c  = ws.sum()
                if c > 0.0:
                    xs[i] = np.sum( self.x * ws / c)
                else:
                    xs[i] = np.mean(self.x)
                if i % 10 == 0:
                    print("progress: " + str(i))
            return xs
        else:
            xs = np.zeros(y_test.shape[0])
            for i in range(y_test.shape[0]):
                i_l, i_u, n_hits = self._find_hits(y_test[i, :], ds)
                if n_hits > 0:
                    ws = self.weights(self.y[i_l : i_u, :], y_test[i,:])
                    c  = ws.sum()
                    xs[i] = np.sum(self.x[i_l : i_u] * ws / c)
                else:
                    xs[i] = np.float('nan')
            return xs

    def predict_intervals(self, y_test, taus):
        xs = np.zeros((y_test.shape[0], taus.size))
        for i in range(y_test.shape[0]):
            ws = self.weights(y_test[i,:])
            ws_sum = np.cumsum(ws)
            ws_sum /= ws_sum[-1]
            xs[i, :] = np.interp(taus, ws_sum, self.x.ravel())
        return xs
