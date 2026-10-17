import numpy as np


def gradient_descent(X, y, alpha, n_iters, tol=1e-8):

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    n_samples = X.shape[0]


    X_design = np.hstack([np.ones((n_samples, 1)), X])
    n_params = X_design.shape[1]

    # Weights
    theta = np.zeros(n_params)

    loss_history = []
    converged_iter = None
    prev_loss = None

    for i in range(1, n_iters + 1):
        y_pred = X_design @ theta
        error = y - y_pred
        # MSE
        loss = np.mean(error ** 2)
        loss_history.append(loss)

        # gradient: -(2/n) * X^T @ (y - y_pred)
        grad = -(2.0 / n_samples) * (X_design.T @ error)

        theta = theta - alpha * grad
  
        if not np.isfinite(loss):
            break

        if prev_loss is not None and abs(prev_loss - loss) < tol:
            converged_iter = i
            break

        prev_loss = loss

    return {
        "theta": theta,
        "loss_history": loss_history,
        "converged_iter": converged_iter,
    }