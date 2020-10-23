import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ssi
import scipy.optimize as scop
import scipy.integrate as si
import time

from uit_scripts.shotnoise import gen_shot_noise as gsn

def timestamp(txt=None):
    t0 = time.localtime()
    if txt is None:
        print(t0[3], t0[4], t0[5], t0[6])
    else:
        print(txt, t0[3], t0[4], t0[5], t0[6])


def one_s_two_exp(t, c, d, amp, sc):
    """Create a one sided exponentially decaying function
    using two exponentials.

    Args:
        t (np.ndarray): x axis
        c (float): amplitude diff. bw. the exponentials
        d (float): diff. in temporal scale bw. the exp.
        amp (float): common amplitude
        sc (float): common temporal scale

    Returns:
        np.ndarray: exponential function of t
    """
    term1 = c * np.exp(-2 * np.abs(t) * sc / (1 - d))
    term2 = (1 - c) * np.exp(- 2 * np.abs(t) * sc / (1 + d))
    return amp * (term1 + term2) / (1 + d - 2 * c * d)


def FuncPen(x, a0, a1, a2, a3):
    # modified function definition with Penalization
    sig_fit = one_s_two_exp(x, a0, a1, a2, a3)
    integral = si.simps(sig_fit, x)
    # integral = si.quad(one_s_two_exp, - np.inf, np.inf, args=(a0, a1, a2, a3))[0]
    penalization = abs(1. - integral) * 10000
    term1 = a0 * np.exp(-2 * np.abs(x) * a3 / (1 - a1))
    term2 = (1 - a0) * np.exp(- 2 * np.abs(x) * a3 / (1 + a1))
    # amp = 1
    return a2 * (term1 + term2) / (1 + a1 - 2 * a0 * a1) + penalization


def optimize(x, y, pen=False):
    if pen:
        func = FuncPen
    else:
        func = one_s_two_exp
    # assert x.ndim == y.ndim == 1, 'Only dim 1 arrays accepted'
    half = int(len(y) / 2 - 10)
    y[:half] = 0
    # Find the largest value
    p = int(np.argwhere(y == np.max(y)))
    # Find the first non-zero value
    # p_s = int((y != 0).argmax(axis=0))
    # Find midpoint b.w. top and start
    # p_m = int((p_t + p_s) / 2)
    zeros = np.zeros_like(y[:p])
    x = x[p:] - x[p]
    y = y[p:]
    sig_cov, _ = scop.curve_fit(func, x, y, p0=[1, 0.5, 1, 1], bounds=([0, 0, 0, 0], [1, 1, 100, 100]))
    sig_fit = one_s_two_exp(x, *sig_cov)
    print('Integral of pulse function = %.3e' % si.simps(sig_fit, x))
    return np.r_[zeros, sig_fit]


def find_nearest(array, value):
    idx = []
    array = np.asarray(array)
    for i in value:
        idx.append((np.abs(array - i)).argmin())
    return idx


def make_signal_zeropad(
        gamma, K, dt, Kdist=False, mA=1., kappa=0.5, TWkappa=0, ampta=False,
        TWdist='exp', Adist='exp', seedTW=None, seedA=None, convolve=True,
        dynamic=False, additive=False, eps=0.1, noise_seed=None,
        kernsize=2**11, kerntype=0, lam=0.5, dkern=False, tol=1e-5):
    """
    Use:
        make_signal(
            gamma, K, dt, Kdist=False, mA=1.,kappa=0.5, TWkappa=0, ampta=False,
            TWdist='exp', Adist='exp', seedTW=None, seedA=None, convolve=True,
            dynamic=False, additive=False, eps=0.1, noise_seed=None,
            kernsize=2**11, kerntype=0, lam=0.5, dkern=False)

    Meta-function with all options. Calls all the above functions.
    Input:
        See the other functions for explanation.
        amptd: If True, returns amplitudes and duration times as well.
    Output:
        The output is given in the following order:
        T, S, S+dynamic noise, S+additive noise, A, ta
        Only outputs noise or amplitudes and duration times if prompted.

    All time is normalized by duration time.
    """
    A, ta, Tend = gsn.amp_ta(
        gamma, K, Kdist=Kdist, mA=mA, kappa=kappa, TWkappa=TWkappa,
        TWdist=TWdist, Adist=Adist, seedTW=seedTW, seedA=seedA)

    # Lets make the first and last 10 percent zeros.
    mask10p = (Tend * .1 < ta) & (ta < Tend * .9)
    ta = ta[mask10p]
    A = A[mask10p]

    if convolve:
        T, S = gsn.signal_convolve(
            A, ta, Tend, dt,
            kernsize=kernsize, kerntype=kerntype,
            lam=lam, dkern=dkern, tol=tol)
    else:
        T, S = gsn.signal_superposition(
            A, ta, Tend, dt,
            kerntype=kerntype, lam=lam, dkern=dkern)

    if (dynamic or additive):
        X = gsn.gen_noise(
            gamma, eps, T, mA=mA,
            kernsize=kernsize, kerntype=kerntype,
            lam=lam, dkern=dkern, tol=tol,
            noise_seed=noise_seed)

    res = (T, S)
    if dynamic:
        res += (S+X[0],)
    if additive:
        res += (S+X[1],)
    if ampta:
        res += (A, ta)
    return res


def est_corr(signal):
    # N = int(1000)
    # phi = 0.6
    # X0 = 0.

    # signal = AR1(N, phi, X0)

    Xn = (signal-signal.mean())/signal.std()

    R = ssi.correlate(Xn, Xn, mode='full')
    n = np.arange(- (signal.size-1), signal.size)

    # Biased correlation function (introduces systematic errors)
    Rb = R / signal.size
    # Unbiased correlation function (introduces no systematic errors, but diverges)
    Rub = R / (signal.size - np.abs(n))

    # Rtrue = phi**np.abs(n)

    plt.figure('AC')
    plt.plot(n, Rub, label='unbiased')
    plt.plot(n, Rb, label='biased')
    # plt.plot(n, Rtrue, 'k:', label='true R')
    plt.xlabel('n')
    plt.ylabel('R_X(n)')
    plt.legend()


def est_psd(signal, *args, new_fig=True):
    """Estimate the power spectral density of `signal`.

    Args:
        signal (np.ndarray): the signal that is analyzed
        *args (str): optional arguments that create `n` different
            plots from the `matplotlib.pyplot` library,
            e.g. `plot`, `loglog`, etc. Defaults to `loglog`.
        new_fig (bool, optional): a new figure is created, set to False
            if you want to just call the plot commands. Defaults to True.
    """
    Xn = (signal - signal.mean()) / signal.std()
    # Periodogram - simple calculation
    fp, P_Xn = ssi.periodogram(Xn, fs=1.)

    if len(args) == 0:
        args = ['loglog']
    for kind in args:
        if new_fig:
            plt.figure()
            plt.title(f'psd - {kind}')
        plot = getattr(plt, kind)
        plot(fp[1:], P_Xn[1:], label='periodogram')

        # Using the Welch method - window size (nperseg) is most important for us.
        for nperseg in 2**np.array([13, 10, 7]):
            f, S_Xn = ssi.welch(Xn, fs=1., nperseg=nperseg)

            plot(f[1:], S_Xn[1:], label='welch 2^{}'.format(int(np.log2(nperseg))))

        # plt.loglog(fp, (1-phi**2)/(1+phi**2-2*phi *
        #                            np.cos(2*np.pi*fp)), 'k:', label='true')
        f_pow = fp[1:]**(- 1 / 2) * 2
        plot(fp[1:], f_pow, label='f^(-1/2)')
        f_pow = fp[1:]**(- 2) * 1e-3
        plot(fp[1:], f_pow, label='f^(-2)')
        f_pow = fp[1:]**0 * 1e3
        plot(fp[1:], f_pow, label='f^0')

        plt.xlabel('f')
        plt.ylabel('PSD')
        plt.legend()
