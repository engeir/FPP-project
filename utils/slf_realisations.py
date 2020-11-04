import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import slf
import tools
import plot_utils as pu

sys.path.append('../uit_scripts')
plt.style.use('ggplot')
pu.figure_setup()
# plt.rcParams['axes.grid'] = True
# # Customize matplotlib
# matplotlib.rcParams.update({
#     'text.usetex': True,
#     'font.family': 'DejaVu Sans',
#     'axes.unicode_minus': False,
#     'pgf.texsystem': 'pdflatex'
# })

save_path = '/home/een023/Documents/FPP_SOC_Chaos/report/figures/'


def fpp_example(save=False):
    """Example of FPP process, exponential everything.
    """
    p = slf.FPPProcess()
    psde = slf.SDEProcess()
    N = int(1e4)
    dt = 1e-2
    gamma = .1
    snr = 0.
    figs = 'fpp_example'
    p.set_params(gamma=gamma, K=int(N * gamma * dt), dt=dt, snr=snr)
    s = p.create_realisation(fit=False)
    pulse = p.fit_pulse(s[0], s[1], s[2])
    p.plot_realisation('plot_real', parameter=s, fit=False)
    plt.subplot(2, 1, 1)
    plt.plot(s[0], pulse[0], 'r', label='Pulse')
    plt.legend()

    if save:
        plt.tight_layout()
        plt.savefig(f'{save_path}{figs}.pdf', bbox_inches='tight', format='pdf', dpi=600)
        plt.savefig(f'{save_path}{figs}.pgf', bbox_inches='tight')
    else:
        plt.show()


def fpp_sde_realisations(save=False):
    """Example of FPP and SDE realisations with varying gamma.

    Args:
        save (bool, optional): save if True, show if False. Defaults to False.
    """
    pf = slf.FPPProcess()
    ps = slf.SDEProcess()
    N = int(1e4)
    dt = 1e-2
    snr = .01
    gamma = [.1, 1., 10.]
    figg = ['fpp_gamma', 'sde_gamma']
    figs = []
    for g in gamma:
        for f in figg:
            figs.append(f'{f}_{g}')
    for i, g in enumerate(gamma):
        pf.set_params(gamma=g, K=int(N * g * dt), dt=dt, snr=snr)
        ps.set_params(gamma=g, K=int(N * g * dt), dt=dt)

        s1, _, s2 = pf.create_realisation(fit=False)
        if g == 10.:
            s2[:100] = np.nan
        s = (s1, s2)
        plt.figure(figs[2 * i], figsize=(5, 3.5))
        ps.plotter(*s, new_fig=False)

        s = ps.create_realisation(fit=False)
        plt.figure(figs[2 * i + 1], figsize=(5, 3.5))
        ps.plotter(*s, new_fig=False)

    if save:
        for f in figs:
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}{f}.pdf', bbox_inches='tight', format='pdf', dpi=200)
            plt.savefig(f'{save_path}{f}.pgf', bbox_inches='tight')
    else:
        plt.show()


def power_law_pulse(save=False):
    """Power law kernel function.

    Observe:
        For tw = [exp,
                  var_rate]
        Any gamma give f^(-1/2)
    """
    p = slf.FPPProcess()
    kern = 'power'
    N = int(1e5)
    dt = 1e-2
    gamma = [.01, .1, 1., 10.]
    figs = ['psd', 'pdf']
    for i, g in enumerate(gamma):
        p.set_params(gamma=g, K=int(N * g * dt), dt=dt, kern=kern, tw='exp')
        s = p.create_realisation(fit=False)

        plt.figure(figs[0])
        plt.subplot(2, 2, i + 1)
        plt.title(f'$\gamma = {g}$')
        p.plot_realisation('plot_psd', parameter=s[-1], fs=dt, new_fig=False)
        # tools.psd(s[-1], new_fig=False)

        plt.figure(figs[1])
        plt.subplot(2, 2, i + 1)
        plt.title(f'$\gamma = {g}$')
        p.plot_realisation('plot_pdf', parameter=s[-1], new_fig=False)
        # tools.pdf(s[-1], new_fig=False)

    if save:
        for f in figs:
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}power_{f}.pdf', bbox_inches='tight', format='pdf', dpi=600)
            plt.savefig(f'{save_path}power_{f}.pgf', bbox_inches='tight')
    else:
        plt.show()


def waiting_times(save=False):
    """Different tw and gamma.

    Observe:
        tw=var_rate: large gamma give f^(-1/2)
        tw=cluster: large gamma give... two Lorentzian?
    """
    p = slf.FPPProcess()
    N = int(1e5)
    dt = 1e-2
    gamma = [.1, 1., 10.]
    TW = ['exp', 'var_rate', 'cluster', 'cox']  # , 'gam', 'deg', 'unif'
    figs = [f'gamma={g}' for g in gamma]
    for j, g in enumerate(gamma):
        for i, tw in enumerate(TW):
            p.set_params(gamma=g, K=int(N * g * dt), dt=dt, tw=tw)
            s = p.create_realisation(fit=False)

            plt.figure(figs[j])
            plt.subplot(2, 2, i + 1)
            plt.title('$\\tau_{\mathrm{w}}=\mathrm{' + tw.replace('_', '\_') + '}$')
            p.plot_realisation('plot_psd', parameter=s[-1], fs=dt, new_fig=False)
            # tools.psd(s[-1], new_fig=False)

            # plt.figure('pdf')
            # plt.subplot(3, 2, i + 1)
            # plt.title('$\\tau_{\mathrm{w}}=\mathrm{' + tw.replace('_', '\_') + '}$')
            # tools.pdf(s[-1], new_fig=False)

    if save:
        for f in figs:
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}tw_{f}.pdf',
                        bbox_inches='tight', format='pdf', dpi=600)
            plt.savefig(f'{save_path}tw_{f}.pgf', bbox_inches='tight')
    else:
        plt.show()


def amplitude_dist(save=False):
    """Plot psd and pdf for different amp distributions.

    Observe: (as expected)
        psd do not change
        pdf do change
    """
    amp = ['exp', 'pareto', 'gam', 'unif', 'ray', 'deg']  # 'alap'
    p = slf.FPPProcess()
    N = int(1e5)
    dt = 1e-1
    g = 1.
    figs = ['psd', 'pdf']
    for i, a in enumerate(amp):
        p.set_params(gamma=g, K=int(N * g * dt), dt=dt, amp=a)
        s = p.create_realisation(fit=False)
        # p.plot_realisation('plot_real', parameter=s)

        plt.figure(figs[0])
        plt.subplot(3, 2, i + 1)
        plt.title(f'Adist = {a}')
        p.plot_realisation('plot_psd', parameter=s[-1], fs=dt, new_fig=False)
        # tools.psd(s[-1], new_fig=False)

        plt.figure(figs[1])
        plt.subplot(3, 2, i + 1)
        plt.title(f'Adist = {a}')
        p.plot_realisation('plot_pdf', parameter=s[-1], new_fig=False)
        # tools.pdf(s[-1], new_fig=False)

    if save:
        for f in figs:
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}adist_{f}.pdf',
                        bbox_inches='tight', format='pdf', dpi=600)
            plt.savefig(f'{save_path}adist_{f}.pgf', bbox_inches='tight')
    else:
        plt.show()


def compare_variations():
    """Test function: plot using e.g. different kernel and tw.
    """
    # f^(-1/2) candidates:
    # kern, tw, gamma = 1exp, var_rate, 1.
    p = slf.FPPProcess()
    ps = slf.SDEProcess()
    kern = ['1exp', 'power', '1exp']  # 'power', '1exp'
    tw = ['exp', 'exp', 'var_rate']  # 'exp', 'var_rate', 'ray', 'cluster'
    N = int(1e5)
    dt = 1e-2
    g = .01
    for k, TW in zip(kern, tw):
        print(k, TW)
        p.set_params(gamma=g, K=int(N * g * dt), kern=k, snr=.0, tw=TW, dt=dt)
        param = p.create_realisation(fit=False)
        p.plot_realisation('plot_psd', parameter=param[-1])  #psd=True)
    ps.set_params(gamma=g, K=int(N * g * dt), dt=dt)
    ps.plot_realisation('plot_psd')  #psd=True)

    plt.show()


def fpp_change_gamma(save=False):
    """Plot the psd of the FPP (tw=var_rate, cox) for different
    gamma values.

    Observe:
        var_rate: Large gamma give f^(-1/2)
        cox:
    """
    N = int(1e5)
    dt = 1e-2
    gamma = [.01, .1, 1., 10.]
    p = slf.FPPProcess()
    tw = 'cox'
    figs = ['psd', 'tw']
    for i, g in enumerate(gamma):
        p.set_params(gamma=g, K=int(N * g * dt), dt=dt, tw=tw)
        s = p.create_realisation(fit=False)
        # p.plot_realisation('plot_real', parameter=s)

        plt.figure(figs[0])
        plt.subplot(2, 2, i + 1)
        plt.title(f'$\gamma = {g}$')
        p.plot_realisation('plot_psd', parameter=s[-1], fs=dt, new_fig=False)
        plt.figure(figs[1])
        plt.subplot(2, 2, i + 1)
        plt.title(f'$\gamma = {g}$')
        p.plot_realisation('plot_tw', parameter=s, new_fig=False)
        # tools.psd(s[-1], new_fig=False, fs=dt)

    if save:
        for f in figs:
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}{tw}_{f}.pdf',
                        bbox_inches='tight', format='pdf', dpi=600)
            plt.savefig(f'{save_path}{tw}_{f}.pgf', bbox_inches='tight')
    else:
        plt.show()


def sde_change_gamma(save=False):
    """Plot the psd of the stochastic logistic function
    for different gamma values.

    Observe:
        Small gamma give f^(-1/2) (as expected)
    """
    N = int(1e5)
    dt = 1e-2
    gamma = [.01, .1, 1., 10.]
    p = slf.SDEProcess()
    figs = ['gamma']
    plt.figure(figs[0])
    for i, g in enumerate(gamma):
        plt.subplot(2, 2, i + 1)
        p.set_params(gamma=g, K=int(N * g * dt), dt=dt)
        s = p.create_realisation(fit=False)
        plt.title(f'$\gamma = {g}$')
        p.plot_realisation('plot_psd', parameter=s[-1], fs=dt, new_fig=False)
        # tools.psd(s[-1], new_fig=False)
        # mask = int(len(s[-1]) * .1)
        # plt.semilogy(abs(s[-1]))

    if save:
        for f in figs:
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}sde_{f}.pdf',
                        bbox_inches='tight', format='pdf', dpi=600)
            plt.savefig(f'{save_path}sde_{f}.pgf', bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    # fpp_example()
    fpp_sde_realisations()
    # power_law_pulse()
    # waiting_times()
    # amplitude_dist()
    # compare_variations()
    # fpp_change_gamma()
    # sde_change_gamma()
