from __future__ import annotations

import numpy as np

from biocompute.device import SpikeBlock


class SpikeFeatures:
    def __init__(self, channels: np.ndarray, bins: int = 2, taus_ms: tuple[float, ...] = (40.0, 160.0), step_ms=20.0):
        self.channels = np.asarray(channels, int)
        self.bins, self.taus_ms, self.step_ms = bins, tuple(taus_ms), step_ms
        self.decay = np.exp(-step_ms / np.asarray(self.taus_ms))
        self.reset()

    @property
    def dim(self) -> int:
        return len(self.channels) * (self.bins + len(self.taus_ms))

    def reset(self) -> None:
        self.traces = np.zeros((len(self.taus_ms), len(self.channels)), dtype=np.float32)

    def __call__(self, block: SpikeBlock) -> np.ndarray:
        counts = block.counts(self.bins)[:, self.channels]
        return self.from_counts(counts)

    def from_counts(self, counts: np.ndarray) -> np.ndarray:
        total = counts.sum(0)
        self.traces = self.traces * self.decay[:, None] + total[None]
        return np.sqrt(np.concatenate([counts.reshape(-1), self.traces.reshape(-1)])).astype(np.float32)


def _backend(d: int):
    try:
        import torch

        if d >= 1500 and torch.cuda.is_available():
            return torch, torch.device("cuda")
    except ImportError:
        pass
    return None, None


class LinearReadout:
    def __init__(self, in_dim: int, out_dim: int = 23):
        self.W = np.zeros((in_dim + 1, out_dim), dtype=np.float32)
        self.mean = np.zeros(in_dim, dtype=np.float32)
        self.std = np.ones(in_dim, dtype=np.float32)

    y_mean: np.ndarray | None = None
    y_std: np.ndarray | None = None

    def _y_in(self, Y: np.ndarray) -> np.ndarray:
        return Y if self.y_mean is None else (Y - self.y_mean) / self.y_std

    def _y_out(self, P: np.ndarray) -> np.ndarray:
        return P if self.y_mean is None else P * self.y_std + self.y_mean

    def fit(self, X: np.ndarray, Y: np.ndarray, alpha: float = 1.0, weights: np.ndarray | None = None,
            standardize_y: bool | None = None) -> None:
        if standardize_y is not None:
            self.y_mean, self.y_std = (Y.mean(0), np.maximum(Y.std(0), 1e-2)) if standardize_y else (None, None)
        Y = self._y_in(Y)
        self.mean = X.mean(0)
        self.std = np.maximum(X.std(0), 1e-3)
        torch, dev = _backend(X.shape[1])
        if torch is not None:
            f = lambda a: torch.as_tensor(np.ascontiguousarray(a, np.float32), device=dev)
            d = X.shape[1] + 1
            ZtZ = torch.zeros(d, d, device=dev)
            ZtY = torch.zeros(d, Y.shape[1], device=dev)
            mean, std = f(self.mean), f(self.std)
            for i in range(0, len(X), 20000):
                Z = torch.cat([(f(X[i : i + 20000]) - mean) / std, torch.ones(len(X[i : i + 20000]), 1, device=dev)], 1)
                Yt = f(Y[i : i + 20000])
                if weights is not None:
                    sw = f(np.sqrt(weights[i : i + 20000]))[:, None]
                    Z, Yt = Z * sw, Yt * sw
                ZtZ += Z.T @ Z
                ZtY += Z.T @ Yt
            reg = alpha * len(X) / 1000.0 * torch.eye(d, device=dev)
            reg[-1, -1] = 0.0
            self.W = torch.linalg.solve(ZtZ + reg, ZtY).cpu().numpy().astype(np.float32)
            del ZtZ, ZtY
            torch.cuda.empty_cache()
            return
        Z = np.c_[(X - self.mean) / self.std, np.ones(len(X))]
        if weights is not None:
            sw = np.sqrt(weights)[:, None]
            Z, Y = Z * sw, Y * sw
        reg = alpha * len(X) / 1000.0 * np.eye(Z.shape[1])
        reg[-1, -1] = 0.0
        self.W = np.linalg.solve(Z.T @ Z + reg, Z.T @ Y).astype(np.float32)

    def fit_cv(self, X: np.ndarray, Y: np.ndarray, alphas=None, folds: int = 5, standardize_y: bool = False,
               weights: np.ndarray | None = None) -> dict:
        alphas = np.logspace(-2, 5, 15) if alphas is None else np.asarray(alphas)
        Y_raw = Y
        if standardize_y:
            Y = (Y - Y.mean(0)) / np.maximum(Y.std(0), 1e-2)
        n = len(X)
        edges = np.linspace(0, n, folds + 1).astype(int)
        scores = np.zeros((len(alphas), folds))
        torch, dev = _backend(X.shape[1])
        if torch is not None:
            return self._fit_cv_torch(X, Y, Y_raw, alphas, edges, standardize_y, weights, torch, dev)
        X64, Y64 = np.asarray(X, np.float64), np.asarray(Y, np.float64)
        blocks = []
        for f in range(folds):
            Xb, Yb = X64[edges[f] : edges[f + 1]], Y64[edges[f] : edges[f + 1]]
            blocks.append((len(Xb), Xb.sum(0), Xb.T @ Xb, Xb.T @ Yb, Yb.sum(0)))
        tot = [sum(b[i] for b in blocks) for i in range(5)]
        for f in range(folds):
            nt = tot[0] - blocks[f][0]
            sx, G, C, sy = (tot[i] - blocks[f][i] for i in (1, 2, 3, 4))
            mean, ym = sx / nt, sy / nt
            std = np.sqrt(np.maximum(np.diag(G) / nt - mean**2, 0.0))
            std = np.maximum(std, 1e-3)
            ZtZ = (G - nt * np.outer(mean, mean)) / np.outer(std, std)
            ZtY = (C - nt * np.outer(mean, ym)) / std[:, None]
            evals, evecs = np.linalg.eigh(ZtZ)
            proj = evecs.T @ ZtY
            Zv = (X64[edges[f] : edges[f + 1]] - mean) / std
            Yv = Y64[edges[f] : edges[f + 1]]
            for i, a in enumerate(alphas):
                W = evecs @ (proj / (evals + a * nt / 1000.0)[:, None])
                scores[i, f] = r2_pooled(Yv, Zv @ W + ym)
        mean_scores = scores.mean(1)
        best = int(np.argmax(mean_scores))
        self.fit(X, Y_raw, alpha=float(alphas[best]), standardize_y=standardize_y, weights=weights)
        self.alpha = float(alphas[best])
        return {"alpha": self.alpha, "val_r2": float(mean_scores[best]),
                "val_r2_by_alpha": dict(zip(map(float, alphas), map(float, mean_scores)))}

    def _fit_cv_torch(self, X, Y, Y_raw, alphas, edges, standardize_y, weights, torch, dev) -> dict:
        folds = len(edges) - 1
        g = lambda a: torch.as_tensor(np.ascontiguousarray(a, np.float32), device=dev)
        blocks = []
        for f in range(folds):
            Xb, Yb = g(X[edges[f] : edges[f + 1]]), g(Y[edges[f] : edges[f + 1]])
            h = lambda t: t.cpu().numpy().astype(np.float64)
            blocks.append((len(Xb), h(Xb.sum(0)), h(Xb.T @ Xb), h(Xb.T @ Yb), h(Yb.sum(0))))
            del Xb, Yb
        tot = [sum(b[i] for b in blocks) for i in range(5)]
        scores = np.zeros((len(alphas), folds))
        for f in range(folds):
            nt = tot[0] - blocks[f][0]
            sx, G, C, sy = (tot[i] - blocks[f][i] for i in (1, 2, 3, 4))
            mean, ym = sx / nt, sy / nt
            std = np.maximum(np.sqrt(np.maximum(np.diag(G) / nt - mean**2, 0.0)), 1e-3)
            ZtZ = (G - nt * np.outer(mean, mean)) / np.outer(std, std)
            del G
            ZtY = (C - nt * np.outer(mean, ym)) / std[:, None]
            evals, evecs = torch.linalg.eigh(g(ZtZ))
            del ZtZ
            proj = evecs.T @ g(ZtY)
            Zv = (g(X[edges[f] : edges[f + 1]]) - g(mean)) / g(std)
            Yv = g(Y[edges[f] : edges[f + 1]])
            ymt = g(ym)
            ss_tot = ((Yv - Yv.mean(0)) ** 2).sum().clamp(min=1e-12)
            for i, a in enumerate(alphas):
                W = evecs @ (proj / (evals.clamp(min=0) + a * nt / 1000.0)[:, None])
                scores[i, f] = float(1.0 - ((Yv - (Zv @ W + ymt)) ** 2).sum() / ss_tot)
            del evecs, Zv, proj
            torch.cuda.empty_cache()
        del blocks, tot
        torch.cuda.empty_cache()
        mean_scores = scores.mean(1)
        best = int(np.argmax(mean_scores))
        self.fit(X, Y_raw, alpha=float(alphas[best]), standardize_y=standardize_y, weights=weights)
        self.alpha = float(alphas[best])
        return {"alpha": self.alpha, "val_r2": float(mean_scores[best]),
                "val_r2_by_alpha": dict(zip(map(float, alphas), map(float, mean_scores)))}

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._y_out(np.r_[(x - self.mean) / self.std, 1.0] @ self.W).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._y_out(np.c_[(X - self.mean) / self.std, np.ones(len(X))] @ self.W).astype(np.float32)


def r2_pooled(Y: np.ndarray, P: np.ndarray) -> float:
    return float(1.0 - ((Y - P) ** 2).sum() / max(((Y - Y.mean(0)) ** 2).sum(), 1e-12))


def r2_per_output(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    ss_res = ((Y - P) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


class MotorMapDecoder:
    def __init__(self, channels: np.ndarray, n_out: int = 23, gain: float = 1.0, rate_scale: float = 2.0,
                 tau_ms: float = 40.0, step_ms: float = 20.0, seed: int = 0):
        channels = np.asarray(channels, int)
        rng = np.random.default_rng(seed)
        channels = rng.permutation(channels)
        groups = np.array_split(channels, 2 * n_out)
        self.plus, self.minus = groups[0::2], groups[1::2]
        self.gain, self.rate_scale = gain, rate_scale
        self.decay = float(np.exp(-step_ms / tau_ms))
        self.n_out = n_out
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros(self.n_out, dtype=np.float32)

    def __call__(self, block: SpikeBlock) -> np.ndarray:
        counts = block.counts(1)[0]
        diff = np.array([counts[p].sum() - counts[m].sum() for p, m in zip(self.plus, self.minus)], np.float32)
        self.state = self.decay * self.state + (1 - self.decay) * diff
        return np.tanh(self.gain * self.state / self.rate_scale).astype(np.float32)


class MLPReadout:
    def __init__(self, in_dim: int, out_dim: int = 23, hidden: int = 256, skip_dim: int = 0, dropout: float = 0.2):
        self.in_dim, self.out_dim, self.hidden, self.skip_dim, self.dropout = in_dim, out_dim, hidden, skip_dim, dropout
        self.params = None

    def _net(self, torch):
        nn = torch.nn

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.drop = nn.Dropout(self.dropout)
                s.l1 = nn.Linear(self.in_dim, self.hidden)
                s.l2 = nn.Linear(self.hidden, self.hidden)
                s.l3 = nn.Linear(self.hidden, self.out_dim)
                s.skip = nn.Linear(max(self.skip_dim, 1), self.out_dim, bias=False)

            def forward(s, x):
                h = torch.nn.functional.elu(s.l1(s.drop(x)))
                h = torch.nn.functional.elu(s.l2(h))
                y = s.l3(h)
                if self.skip_dim:
                    y = y + s.skip(x[:, -self.skip_dim :])
                return y

        return Net()

    def fit(self, X: np.ndarray, Y: np.ndarray, weights: np.ndarray | None = None, epochs: int = 30,
            batch: int = 512, lr: float = 1e-3, weight_decay: float = 1e-4, val_frac: float = 0.1, seed: int = 0) -> dict:
        import torch

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.mean, self.std = X.mean(0), np.maximum(X.std(0), 1e-3)
        self.y_mean, self.y_std = Y.mean(0), np.maximum(Y.std(0), 1e-2)
        n = len(X)
        cut = int(n * (1 - val_frac))
        Xt = torch.as_tensor(((X - self.mean) / self.std).astype(np.float32))
        Yt = torch.as_tensor(((Y - self.y_mean) / self.y_std).astype(np.float32))
        Wt = torch.as_tensor((np.ones(n) if weights is None else weights).astype(np.float32))
        net = self._net(torch).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=weight_decay)
        idx_train = torch.arange(cut)
        for ep in range(epochs):
            net.train()
            perm = idx_train[torch.randperm(cut)]
            for i in range(0, cut, batch):
                b = perm[i : i + batch]
                xb, yb, wb = Xt[b].to(dev), Yt[b].to(dev), Wt[b].to(dev)
                loss = ((((net(xb) - yb) ** 2).mean(1)) * wb).sum() / wb.sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            P = torch.cat([net(Xt[i : i + 4096].to(dev)).cpu() for i in range(cut, n, 4096)]).numpy()
        V = Yt[cut:].numpy()
        val_r2 = float(1 - ((V - P) ** 2).sum() / max(((V - V.mean(0)) ** 2).sum(), 1e-12))
        self.params = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        del net, Xt, Yt
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return {"val_r2": val_r2}

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.params
        z = ((X - self.mean) / self.std).astype(np.float32)
        elu = lambda a: np.where(a > 0, a, np.expm1(np.minimum(a, 0)))
        h = elu(z @ p["l1.weight"].T + p["l1.bias"])
        h = elu(h @ p["l2.weight"].T + p["l2.bias"])
        y = h @ p["l3.weight"].T + p["l3.bias"]
        if self.skip_dim:
            y = y + z[:, -self.skip_dim :] @ p["skip.weight"].T
        return (y * self.y_std + self.y_mean).astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x[None])[0]


class EnsembleReadout:
    def __init__(self, members: list):
        self.members = list(members)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean([m.predict(X) for m in self.members], axis=0).astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x[None])[0]
