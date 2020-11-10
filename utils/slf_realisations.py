import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec

import slf
import tools
import plot_utils as pu

sys.path.append('/home/een023/uit_scripts')
from uit_scripts.plotting import figure_defs as fd
fd.set_rcparams_article_thickline(plt.rcParams)
plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.style.use('ggplot')
# pu.figure_setup()

# plt.rcParams['axes.grid'] = True
# # Customize matplotlib
# matplotlib.rcParams.update({
#     'text.usetex': True,
#     'font.family': 'DejaVu Sans',
#     'axes.unicode_minus': False,
#     'pgf.texsystem': 'pdflatex'
# })

data_path = '/home/een023/Documents/FPP_SOC_Chaos/report/data/'
save_path = '/home/een023/Documents/FPP_SOC_Chaos/report/figures/'


def fpp_example(data=True, save=False):
    """Example of FPP process, exponential everything.
    """
    figs = 'fpp_example'
    file = f'{data_path}{figs}.npz'
    p = slf.FPPProcess()
    if not data:
        psde = slf.SDEProcess()
        N = int(1e4)
        dt = 1e-2
        gamma = .1
        snr = 0.
        p.set_params(gamma=gamma, K=int(N * gamma * dt), dt=dt, snr=snr)
        s = p.create_realisation(fit=False)
        pulse = p.fit_pulse(s[0], s[1], s[2])[0]

        np.savez(file, s=s, pulse=pulse)
    else:
        print(f'Loading data from {file}')
        f = np.load(file, allow_pickle=True)
        s = f['s']
        pulse = f['pulse']

    p.plot_realisation('plot_real', parameter=s, fit=False)
    plt.subplot(2, 1, 1)
    plt.plot(s[0], pulse, 'r', label='Pulse')
    plt.legend()

    if save:
        plt.tight_layout()
        plt.savefig(f'{save_path}{figs}.pdf', bbox_inches='tight', format='pdf', dpi=600)
        plt.savefig(f'{save_path}{figs}.pgf', bbox_inches='tight')
    else:
        plt.show()


def fpp_sde_realisations(data=True, save=False):
    """Example of FPP and SDE realisations with varying gamma.

    Args:
        save (bool, optional): save if True, show if False. Defaults to False.
    """
    file = f'{data_path}fpp_sde.npz'
    gamma = [.1, 1., 10.]
    figs = ['fpp_gamma', 'sde_gamma']
    if not data:
        pf = slf.FPPProcess()
        ps = slf.SDEProcess()
        N = int(1e4)
        dt = 1e-2
        snr = .01
        fpp = []
        sde = []
        for g in gamma:
            pf.set_params(gamma=g, K=int(N * g * dt), dt=dt, snr=snr)
            ps.set_params(gamma=g, K=int(N * g * dt), dt=dt)

            s1, _, s2 = pf.create_realisation(fit=False)
            if g == 10.:
                s2[:100] = np.nan
            s = (s1, s2)
            fpp.append(s)

            s = ps.create_realisation(fit=False)
            sde.append(s)

        np.savez(file, fpp=fpp, sde=sde)
    else:
        print(f'Loading data from {file}')
        f = np.load(file, allow_pickle=True)
        fpp = f['fpp']
        sde = f['sde']

    lab = [f'$\gamma = {g}$' for g in gamma]
    tools.ridge_plot(fpp, xlabel='$ t $', ylabel='$ \Phi $', labels=lab, figname=figs[0])
    tools.ridge_plot(sde, xlabel='$ t $', ylabel='$ \Phi $', labels=lab, figname=figs[1])

    if save:
        for f in figs:
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}{f}.pdf',
                        bbox_inches='tight', format='pdf', dpi=200)
            plt.savefig(f'{save_path}{f}.pgf', bbox_inches='tight')
    else:
        plt.show()


def fpp_sde_real_L(data=True, save=False):
    """Example of FPP and SDE realisations with varying gamma.

    Args:
        save (bool, optional): save if True, show if False. Defaults to False.
    """
    file = f'{data_path}fpp_sde_L.npz'
    gamma = [.01, .1, 1.]
    figs = ['fpp_gamma_L', 'sde_gamma_L']
    if not data:
        pf = slf.FPPProcess()
        ps = slf.SDEProcess()
        N = int(1e6)
        dt = 1e-2
        snr = .01
        fpp = []
        sde = []
        for g in gamma:
            pf.set_params(gamma=g, K=int(N * g * dt), dt=dt, snr=snr)
            ps.set_params(gamma=g, K=int(N * g * dt), dt=dt)

            s1, _, s2 = pf.create_realisation(fit=False)
            if g == 10.:
                s2[:100] = np.nan
            s = (s1, s2)
            fpp.append(s)

            s = ps.create_realisation(fit=False)
            sde.append(s)

        np.savez(file, fpp=fpp, sde=sde)
    else:
        print(f'Loading data from {file}')
        f = np.load(file, allow_pickle=True)
        fpp = f['fpp']
        sde = f['sde']

    plt.rcParams['lines.linewidth'] = .4
    lab = [f'$\gamma = {g}$' for g in gamma]
    tools.ridge_plot(fpp, xlabel='$ t $', ylabel='$\Phi$', labels=lab, figname=figs[0])
    tools.ridge_plot(sde, xlabel='$ t $', ylabel='$\Phi$', labels=lab, figname=figs[1])

    if save:
        for f in figs:
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}{f}.pdf',
                        bbox_inches='tight', format='pdf', dpi=200)
            plt.savefig(f'{save_path}{f}.pgf', bbox_inches='tight', dpi=200)
    else:
        plt.show()


def fpp_sde_psdpdf(data=True, save=False):
    # file = f'{data_path}fpp_sde_psdpdf.npz'
    file = f'{data_path}fpp_sde_L.npz'
    gamma = [.1, 1., 10.]
    figs = ['fpp_psd', 'sde_psd']
    dt = 1e-2
    if not data:
        pf = slf.FPPProcess()
        ps = slf.SDEProcess()
        N = int(5e5)
        snr = 0.01
        fpp = []
        sde = []
        for g in gamma:
            pf.set_params(gamma=g, K=int(N * g * dt), dt=dt, snr=snr)
            ps.set_params(gamma=g, K=int(N * g * dt), dt=dt)

            s1, _, s2 = pf.create_realisation(fit=False)
            s = (s1, s2)
            fpp.append(s)

            s = ps.create_realisation(fit=False)
            sde.append(s)

        np.savez(file, fpp=fpp, sde=sde)
    else:
        print(f'Loading data from {file}')
        f = np.load(file, allow_pickle=True)
        fpp = f['fpp']
        sde = f['sde']

    lab = [f'$\gamma = {g}$' for g in gamma]
    tools.ridge_plot_psd(fpp, dt, xlabel='$ f $', ylabel='$ S $', labels=lab, figname=figs[0])
    tools.ridge_plot_psd(sde, dt, xlabel='$ f $', ylabel='$ S $', labels=lab, figname=figs[1])

    if save:
        for f in figs:
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}{f}.pdf',
                        bbox_inches='tight', format='pdf', dpi=200)
            plt.savefig(f'{save_path}{f}.pgf', bbox_inches='tight', dpi=200)
    else:
        plt.show()


def fpp_tw_real(data=True, save=False):
    file = f'{data_path}fpp_tw_real.npz'
    rate = ['ou', 'ou']
    figs = ['cox', 'tick']
    gamma = [.01, .1, 1.]
    if not data:
        p = slf.FPPProcess()
        dt = 1e-2
        N = int(1e6)
        snr = .01
        fpps = [[], []]
        for i, (r, tw) in enumerate(zip(rate, figs)):
            for g in gamma:
                p.set_params(gamma=g, K=int(N * g * dt), dt=dt,
                             tw=tw, snr=snr, rate=r, amp='ray')
                s1, _, s2 = p.create_realisation(fit=False)
                s = (s1, s2)
                fpps[i].append(s)
                print(p.K)

        # fpp_vr = fpps[0]
        fpp_c = fpps[0]
        fpp_t = fpps[1]
        del fpps
        np.savez(file, fpp_c=fpp_c, fpp_t=fpp_t)
    else:
        print(f'Loading data from {file}')
        f = np.load(file, allow_pickle=True)
        # fpp_vr = f['fpp_vr']
        fpp_c = f['fpp_c']
        fpp_t = f['fpp_t']

    plt.rcParams['lines.linewidth'] = .4
    lab = [f'$\gamma = {g}$' for g in gamma]
    # tools.ridge_plot(fpp_vr, xlabel='$ t $', ylabel='$\Phi$', labels=lab, figname=figs[0])
    tools.ridge_plot(fpp_c, xlabel='$ t $', ylabel='$\Phi$', labels=lab, figname=figs[0])
    tools.ridge_plot(fpp_t, xlabel='$ t $', ylabel='$\Phi$', labels=lab, figname=figs[1])

    if save:
        for f in figs:
            print(f'Saving to {save_path}{f}.*')
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}{f}.pdf',
                        bbox_inches='tight', format='pdf', dpi=200)
            plt.savefig(f'{save_path}{f}.pgf', bbox_inches='tight', dpi=200)
    else:
        plt.show()


def fpp_tw_psd(save=False):
    file = f'{data_path}fpp_tw_real.npz'
    figs = ['cox_psd', 'tick_psd']
    gamma = [.01, .1, 1.]
    dt = 1e-2

    print(f'Loading data from {file}')
    f = np.load(file, allow_pickle=True)
    # fpp_vr = f['fpp_vr']
    fpp_c = f['fpp_c']
    fpp_t = f['fpp_t']

    lab = [f'$\gamma = {g}$' for g in gamma]
    # tools.ridge_plot_psd(fpp_vr, dt, xlabel='$ f $', ylabel='$ S $', labels=lab, figname=figs[0])
    tools.ridge_plot_psd(fpp_c, dt, xlabel='$ f $', ylabel='$ S $', labels=lab, figname=figs[0])
    tools.ridge_plot_psd(fpp_t, dt, xlabel='$ f $', ylabel='$ S $', labels=lab, figname=figs[1])

    if save:
        for f in figs:
            print(f'Saving to {save_path}{f}.*')
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}{f}.pdf',
                        bbox_inches='tight', format='pdf', dpi=200)
            plt.savefig(f'{save_path}{f}.pgf', bbox_inches='tight', dpi=200)
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

        plt.figure(figs[0], figsize=(7, 5))
        plt.subplot(2, 2, i + 1)
        plt.title(f'$\gamma = {g}$')
        p.plot_realisation('plot_psd', parameter=s[-1], fs=dt, new_fig=False)
        # tools.psd(s[-1], new_fig=False)

        plt.figure(figs[1], figsize=(7, 5))
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
    gamma = [.1]
    TW = ['exp', 'var_rate', 'tick', 'cox']  # , 'gam', 'deg', 'unif'
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


if __name__ == '__main__':
    # fpp_example()
    # fpp_sde_realisations()
    # fpp_sde_real_L()
    # fpp_sde_psdpdf()
    # fpp_tw_real(data=False)
    fpp_tw_psd()
    # power_law_pulse()
    # waiting_times()
    # amplitude_dist()
