import itertools
import time

import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as si
import scipy.optimize as scop
import scipy.signal as ssi
from scipy import stats
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


def psd(signal, *args, fs=1e-2, new_fig=True):
    """Estimate the power spectral density of `signal`.

    Args:
        signal (np.ndarray): the signal that is analyzed
        *args (str): optional arguments that create `n` different
            plots from the `matplotlib.pyplot` library,
            e.g. `plot`, `loglog`, etc. Defaults to `loglog`.
        new_fig (bool, optional): a new figure is created, set to False
            if you want to just call the plot commands. Defaults to True.
    """
    c = ['b', 'r', 'g', 'magenta', 'yellow', 'royalblue',
         'chartreuse', 'firebrick', 'darkorange']
    c = itertools.cycle(c)
    Xn = (signal - signal.mean()) / signal.std()
    # Periodogram - simple calculation
    fp, P_Xn = ssi.periodogram(Xn, fs=fs)
    if len(args) == 0:
        args = ['loglog']
    for kind in args:
        if new_fig or len(args) > 1:
            plt.figure()
            plt.title(f'psd - {kind}')
        plot = getattr(plt, kind)
        al = .5
        plot(fp[1:], P_Xn[1:], f'{next(c)}', label='periodogram', alpha=al)

        # Using the Welch method - window size (nperseg) is most important for us.
        for nperseg in 2**np.array([13, 10]):
            if al < 1.:
                al += .2
            f, S_Xn = ssi.welch(Xn, fs=fs, nperseg=nperseg)

            plot(f[1:], S_Xn[1:], f'{next(c)}', label='welch $2^{' +
                 str(int(np.log2(nperseg))) + '}$', alpha=al)

        # plt.loglog(fp, (1-phi**2)/(1+phi**2-2*phi *
        #                            np.cos(2*np.pi*fp)), 'k:', label='true')
        w = 2 * np.pi * fp[1:]
        t_d = (1 / fs)**2
        lor = 2 * t_d / (1 + (t_d * w)**2)
        plot(fp[1:], lor, 'k--', label='Lorentz spectrum', alpha=.8)
        # f_pow = fp[1:]**(- 1 / 2) * 2
        # plot(fp[1:], f_pow, label='$f^{-1/2}$')
        # f_pow = fp[1:]**(- 2) * 1e-3
        # plot(fp[1:], f_pow, label='$f^{-2}$')
        # f_pow = fp[1:]**0 * 1e3
        # plot(fp[1:], f_pow, label='$f^0$')

        plt.xlabel('$f$')
        plt.ylabel('PSD')
        # plt.legend()


def pdf(signal, *args, new_fig=True):
    """Estimate the PDF of `signal`.

    Args:
        signal (np.ndarray): the signal that is analyzed
        *args (str): optional arguments that create `n` different
            plots from the `matplotlib.pyplot` library,
            e.g. `plot`, `loglog`, etc. Defaults to `loglog`.
        new_fig (bool, optional): a new figure is created, set to False
            if you want to just call the plot commands. Defaults to True.
    """
    # bins = np.linspace(-5, 5, 30)
    histogram, bins = np.histogram(signal, bins=10, density=True)
    bin_centers = 0.5*(bins[1:] + bins[:-1])
    # Compute the PDF on the bin centers from scipy distribution object
    # norm_pdf = stats.norm.pdf(bin_centers)

    if len(args) == 0:
        args = ['semilogy']
    for kind in args:
        if new_fig or len(args) > 1:
            plt.figure()
            plt.title(f'pdf - {kind}')
        plot = getattr(plt, kind)
        plot(bin_centers, histogram, label="Histogram of samples")
        # plot(bin_centers, norm_pdf, label="PDF")
        plt.xlabel('$f$')
        plt.ylabel('PDF')
        plt.legend()


def est_pdf(signal):
    """Estimate the PDF of `signal`.

    Args:
        signal (np.ndarray): the signal that is analysed
    """
    # bins = np.linspace(-5, 5, 30)
    histogram, bins = np.histogram(signal, bins=10, density=True)
    bin_centers = 0.5*(bins[1:] + bins[:-1])
    # Compute the PDF on the bin centers from scipy distribution object
    # norm_pdf = stats.norm.pdf(bin_centers)
    return bin_centers, histogram


def ridge_plot_psd(data, fs, *args, xlabel=False, ylabel=False, labels=False, figname=None):
    """Plot data in a ridge plot with fixed width and fixed height per ridge.

    Args:
        data (list): a list of n tuples/lists of length 2: (x, y)-pairs
        xlabel (bool or str, optional): x-label placed at the bottom, nothing if False. Defaults to False.
        ylabel (bool or str, optional): y-label placed at all ridge y-axis, nothing if False. Defaults to False.
        labels (bool or list, optional): list of str with the labels of the ridges; must be the same length as `data`. Defaults to False.
        figname (None or str, optional): first arg in plt.figure(); useful for tracking figure-object. Defaults to None.

    *args:
        'squeeze': set spacing between subplots to -.5.
    """
    fsize = (4, 1.5 * len(data))
    gs = grid_spec.GridSpec(len(data), 1)
    if figname is not None:
        fig = plt.figure(figname, figsize=fsize)
    else:
        fig = plt.figure(figsize=fsize)
    ax_objs = []
    l2 = []
    c = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    clr = np.linspace(1, 0, 4)
    c = itertools.cycle(c)
    clr = itertools.cycle(clr)
    for i, s in enumerate(data):
        col = next(c)
        col1 = np.copy(col)
        # === Calculate PSD
        signal = s[1]
        Xn = (signal - signal.mean()) / signal.std()
        ax_objs.append(fig.add_subplot(gs[i:i + 1, 0:]))
        # === Do actual plotting
        fp, P_Xn = ssi.periodogram(Xn, fs=fs)
        col[i] = next(clr)
        l = ax_objs[-1].loglog(fp[1:], P_Xn[1:], color=(col[0], col[1], col[2]), label='periodogram', alpha=.5)[0]
        l2.append(l)
        for nperseg in 2**np.array([13, 10]):
            f, S_Xn = ssi.welch(Xn, fs=fs, nperseg=nperseg)
            col[i] = next(clr)
            ax_objs[-1].loglog(f[1:], S_Xn[1:], color=(col[0], col[1], col[2]), label='welch $2^{' + str(int(np.log2(nperseg))) + '}$', alpha=.6)
        w = 2 * np.pi * fp[1:]
        t_d = (1 / fs)**2
        lor = 2 * t_d / (1 + (t_d * w)**2)
        col[i] = next(clr)
        ax_objs[-1].loglog(fp[1:], lor, '--', color=(col[0], col[1], col[2]),
                           label='Lorentz spectrum', alpha=.8)
        # === Do actual plotting
        if 'squeeze' in args:
            if i % 2:
                ax_objs[-1].tick_params(axis='y', which='both', left=False,
                                        labelleft=False, labelright=True)
                ax_objs[-1].spines['left'].set_color('k')
            else:
                ax_objs[-1].tick_params(axis='y', which='both', right=False,
                                        labelleft=True, labelright=False)
                ax_objs[-1].spines['right'].set_color('k')
        if i == 0:
            spines = ["bottom"]
        elif i == len(data) - 1:
            spines = ["top"]
        else:
            spines = ["top", "bottom"]
        ax_objs[-1].patch.set_alpha(0)
        for sp in spines:
            ax_objs[-1].spines[sp].set_visible(False)
            # ax_objs[-1].spines['left'].set_color((col1[0], col1[1], col1[2]))
            # ax_objs[-1].spines['right'].set_color((col1[0], col1[1], col1[2]))
            ax_objs[-1].yaxis.label.set_color((col1[0], col1[1], col1[2]))
            ax_objs[-1].tick_params(axis='y', which='both', colors=(col1[0], col1[1], col1[2]))
        if not 'squeeze' in args:
            ax_objs[-1].spines['left'].set_color((col1[0], col1[1], col1[2]))
            ax_objs[-1].spines['right'].set_color((col1[0], col1[1], col1[2]))
        # if ylabel:
        #     plt.ylabel(ylabel)
        if i == len(data) - 1:
            if xlabel:
                plt.xlabel(xlabel)
            plt.tick_params(axis='x', which='both', top=False)
        elif i == 0:
            plt.tick_params(axis='x', which='both',
                            bottom=False, labelbottom=False)
        else:
            plt.tick_params(axis='x', which='both', bottom=False,
                            top=False, labelbottom=False)

    if ylabel:
        fig.text(0.01, 0.5, ylabel, ha='left',
                 va='center', rotation='vertical')

    ax_objs[-1].legend(prop={'size': 6})
    if labels:
        if len(labels) == len(data):
            fig.legend(l2, labels, loc='lower center',  bbox_to_anchor=(.5, 1.),
                    bbox_transform=ax_objs[0].transAxes, ncol=len(data))
        else:
            print('Length of labels and data was not equal.')
    if 'squeeze' in args:
        gs.update(hspace=-0.4)
    else:
        gs.update(hspace=0.)


def ridge_plot(data, *args, xlabel=False, ylabel=False, labels=False, figname=None, y_scale=1., **kwargs):
    """Plot data in a ridge plot with fixed width and fixed height per ridge.

    Args:
        data (list): a list of n tuples/lists of length 2: (x, y)-pairs; list of n np.ndarrays: (y)
        xlabel (bool or str, optional): x-label placed at the bottom, nothing if False. Defaults to False.
        ylabel (bool or str, optional): y-label placed at all ridge y-axis, nothing if False. Defaults to False.
        labels (bool or list, optional): list of str with the labels of the ridges; must be the same length as `data`. Defaults to False.
        figname (None or str, optional): first arg in plt.figure(); useful for tracking figure-object. Defaults to None.
        y_scale (float, optional): scale of y axis relative to the default. Defaults to 1.

    *args:
        'dots': add the 'o' str to the plt.<plt_type>
        'slalomaxis': numbers on the y axis change between left and right to prevent overlap
        'x_lim_S': limit the x axis based on the smallest x ticks insted of the largest (default)
        'grid': turn on grid
        'squeeze': set spacing between subplots to -.5. This turns ON 'slalomaxis' and OFF 'grid'.

    **kwargs:
        plt_type (str, optional): plt class (loglog, plot, semilogx etc.) Defaults to plot.
        xlim (list or tuple, optional): min and max value along x axis (len: 2)
    """
    if 'plt_type' in kwargs.keys():
        plt_type = kwargs['plt_type']
    else:
        plt_type = 'plot'
    fsize = (4, y_scale * len(data))
    gs = grid_spec.GridSpec(len(data), 1)
    if figname is not None:
        fig = plt.figure(figname, figsize=fsize)
    else:
        fig = plt.figure(figsize=fsize)
    ax_objs = []
    l2 = []
    ls = ['-', '--']
    ls = itertools.cycle(ls)
    c = ['r', 'g', 'b', 'magenta', 'darkorange',
         'chartreuse', 'firebrick', 'yellow', 'royalblue']
    c = itertools.cycle(c)
    if 'xlim' in kwargs.keys():
        x_min, x_max = kwargs['xlim']
    # TODO: only given T_max and dt (optional), calculate time / x axis
    # elif len([a for a in args if not isinstance(args, str)]) > 0:
    #     if len([a for a in args if not isinstance(args, str)]) == 1:
    #         T = [a for a in args if not isinstance(args, str)][0]
    #         dt = 1
    #     elif len([a for a in args if not isinstance(args, str)]) == 2:
    #         T = [a for a in args if not isinstance(args, str)][0]
    #         dt = [a for a in args if not isinstance(args, str)][1]
    elif len(data[0]) != 2:
        x_min, x_max = 0, len(data[0])
    elif 'x_lim_S' in args:
        x_min, x_max = x_limit([d[0] for d in data], plt_type, False)
    else:
        x_min, x_max = x_limit([d[0] for d in data], plt_type)

    # Loop through data
    for i, s in enumerate(data):
        col = next(c)
        lnst = next(ls)
        ax_objs.append(fig.add_subplot(gs[i:i + 1, 0:]))
        if i == 0:
            spines = ["bottom"]
        elif i == len(data) - 1:
            spines = ["top"]
        else:
            spines = ["top", "bottom"]

        # Plot data
        p_func = getattr(ax_objs[-1], plt_type)
        line_type = '-o' if 'dots' in args else '-'
        if len(s) == 2:
            l = p_func(s[0], s[1], line_type, color=col, markersize=1.5)[0]
        else:
            l = p_func(s, line_type, color=col, markersize=1.5)[0]

        l2.append(l)
        ax_objs[-1].patch.set_alpha(0)
        plt.xlim([x_min, x_max])
        if 'squeeze' in args:
            if i % 2:
                ax_objs[-1].tick_params(axis='y', which='both', left=False,
                                        labelleft=False, labelright=True)
                ax_objs[-1].spines['left'].set_color('k')
            else:
                ax_objs[-1].tick_params(axis='y', which='both', right=False,
                                        labelleft=True, labelright=False)
                ax_objs[-1].spines['right'].set_color('k')
        elif 'slalomaxis' in args:
            if i % 2:
                ax_objs[-1].tick_params(axis='y', which='both',
                                labelleft=False, labelright=True)
        for sp in spines:
            ax_objs[-1].spines[sp].set_visible(False)
        if 'squeeze' not in args:
            ax_objs[-1].spines['left'].set_color(col)
            ax_objs[-1].spines['right'].set_color(col)
        ax_objs[-1].tick_params(axis='y', which='both', colors=col)
        ax_objs[-1].yaxis.label.set_color(col)
        if 'grid' in args and 'squeeze' not in args:
            plt.grid(True, which="major", ls="-", alpha=0.2)
        if 'grid' in args and 'squeeze' in args:
            plt.minorticks_off()
            plt.grid(True, axis='y', which="major", ls=lnst, alpha=0.2)
            plt.grid(True, axis='x', which="major", ls='-', alpha=0.2)
        if i == len(data) - 1:
            if xlabel:
                plt.xlabel(xlabel)
            plt.tick_params(axis='x', which='both', top=False)
        elif i == 0:
            plt.tick_params(axis='x', which='both',
                            bottom=False, labelbottom=False)  # , labeltop=True
        else:
            plt.tick_params(axis='x', which='both', bottom=False,
                            top=False, labelbottom=False)

    if ylabel:
        # fig.add_subplot(111, frame_on=False)
        # plt.ylabel(ylabel)
        fig.text(0.01, 0.5, ylabel, ha='left', va='center', rotation='vertical')

    if labels:
        if len(labels) == len(data):
            l_d = len(data)
            c_max = 4
            n_row = int(np.ceil(l_d / c_max))
            n_col = 1
            while l_d > n_col * n_row:
                n_col += 1
            fig.legend(l2, labels, loc='lower center',  bbox_to_anchor=(.5, 1.),
                    bbox_transform=ax_objs[0].transAxes, ncol=n_col)
        else:
            print('Length of labels and data was not equal.')
    if 'squeeze' in args:
        gs.update(hspace=-0.5)
    else:
        gs.update(hspace=0.)


def x_limit(data, plt_type, maxx=True):
    t_min = data[0]
    x_max = data[0][-1]
    for t in data[1:]:
        t_0, t_max = np.min(t), np.max(t)
        if maxx:
            t_min = t if t_0 < t_min[0] else t_min
            # t_max = t if t_1 > t_max[-1] else t_max
            x_max = t_max if t_max > x_max else x_max
        else:
            t_min = t if t[0] > t_min[0] else t_min
            # t_max = t if t[-1] < t_max[-1] else t_max
            x_max = t_max if t_max < x_max else x_max
    diff = .05 * (x_max - t_min[0])
    # x_max = t_max[-1] + diff
    x_max += diff
    if plt_type in ['loglog', 'semilogx']:
        x_min = .8 * t_min[t_min > 0][0] if t_min[0] < diff else t_min[0] - diff
        # if x_min < 0:
        #     x_min = 1e-10
    else:
        x_min = t_min[0] - diff
    return x_min, x_max
