import numpy as np
from types import SimpleNamespace


def adam_minimize(
    fun,
    x0,
    jac=None,
    args=(),
    lr=0.1,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    maxiter=100,
    tol=1e-6,
    callback=None,
):
    """
    Simple Adam minimization function with an interface similar to scipy.optimize.minimize.

    Parameters
    ----------
    fun : callable
        fun(x, *args) -> float
    x0 : array_like
        Initial value.
    jac : callable or None
        jac(x, *args) -> grad (same shape as x). If None, use numerical differentiation.
    args : tuple
        Additional arguments for fun/jac.
    lr : float
        Learning rate (alpha).
    beta1, beta2, eps : float
        Adam hyperparameters.
    maxiter : int
        Maximum number of iterations.
    tol : float
        Stop when ||grad|| < tol.
    callback : callable or None
        callback(xk) is called after each step.
    """
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0

    def numerical_grad(f, x, *args):
        grad = np.zeros_like(x)
        h = 1e-8
        fx = f(x, *args)
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += h
            grad[i] = (f(xp, *args) - fx) / h
        return grad

    if jac is None:
        jac_fun = numerical_grad
    else:
        jac_fun = jac

    fun_vals = []
    success = False
    message = ""

    for it in range(1, maxiter + 1):
        g = jac_fun(x, *args)
        g_norm = np.linalg.norm(g)
        fun_vals.append(fun(x, *args))

        if g_norm < tol:
            success = True
            message = "Gradient norm below tolerance."
            break

        t += 1
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)

        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)

        if callback is not None:
            callback(x)

    if not success:
        message = "Maximum iterations reached."
        fun_vals.append(fun(x, *args))        

    result = SimpleNamespace(
        x=x,
        fun=fun_vals[-1],
        jac=jac_fun(x, *args),
        nfev=len(fun_vals),
        nit=it,
        success=success,
        message=message,
        fun_vals=np.array(fun_vals),
    )
    return result
