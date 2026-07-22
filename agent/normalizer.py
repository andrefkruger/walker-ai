import numpy as np


class RunningMeanStd:
    """Normaliza observações online com média/variância acumuladas."""

    def __init__(self, shape):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = 1e-4  # evita divisão por zero no início

    def update(self, x: np.ndarray):
        x = np.atleast_2d(x).astype(np.float64)
        batch_mean  = x.mean(axis=0)
        batch_var   = x.var(axis=0)
        batch_count = x.shape[0]

        delta     = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a      = self.var   * self.count
        m_b      = batch_var  * batch_count
        M2       = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var  = M2 / tot_count

        self.mean  = new_mean
        self.var   = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / (np.sqrt(self.var) + 1e-8)).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean  = d["mean"]
        self.var   = d["var"]
        self.count = d["count"]
