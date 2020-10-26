import sys

import matplotlib
import matplotlib.pyplot as plt

import slf
import tools

sys.path.append('../uit_scripts')
# plt.style.use('ggplot')
plt.rcParams['axes.grid'] = True
# Customize matplotlib
matplotlib.rcParams.update({
    'text.usetex': True,
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'pgf.texsystem': 'pdflatex'
})


def compare_variations():
    # f^(-1/2) candidates:
    # kern, tw, gamma = 1exp, var_rate, 1.
    p = slf.FPPProcess()
    ps = slf.SDEProcess()
    kern = ['1exp', 'power', '1exp']  # 'power', '1exp'
    tw = ['exp', 'exp', 'var_rate']  # 'exp', 'var_rate', 'ray', 'cluster'
    g = .01
    K = 100
    dt = 1e-1
    for k, TW in zip(kern, tw):
        print(k, TW)
        p.set_params(gamma=g, K=K, kern=k, snr=.0, tw=TW, dt=dt)
        param = p.create_realisation(fit=False)
        p.plot_realisation('plot_psd', parameter=param[-1])  #psd=True)
    ps.set_params(gamma=g, K=K, dt=dt)
    ps.plot_realisation('plot_psd')  #psd=True)

    plt.show()


def slf_amplitude():
    r_sde = slf.Realisation(process='SDE')
    r_sde.set_params(gamma=.1, K=1000, dt=.1)
    r_sde.plotter()
    r_sde.plot_psd()
    plt.show()


def sde_change_gamma():
    """Plot the psd of the stochastic logistic function
    for different gamma values to see the 1/f^(1/2) scaling.
    """
    N = int(1e5)
    dt = 1e-1
    gamma = [.01, .1, 1., 10.]
    p = slf.SDEProcess()
    plt.figure()
    for i, g in enumerate(gamma):
        plt.subplot(2, 2, i + 1)
        p.set_params(gamma=g, K=int(N * g * dt), dt=dt)
        s = p.create_realisation(fit=False)
        # print(s)
        plt.title(f'gamma = {g:.0e}')
        tools.psd(s[-1], new_fig=False)
        # mask = int(len(s[-1]) * .1)
        # plt.semilogy(abs(s[-1]))
    plt.show()


if __name__ == '__main__':
    # sde_change_gamma()
    compare_variations()
