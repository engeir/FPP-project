import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec

import slf
import tools
import plot_utils as pu

sys.path.append('/home/een023/uit_scripts')
from uit_scripts.plotting import figure_defs as fd
import uit_scripts.stat_analysis as sa
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
        snr = .0
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
        snr = .0
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
            print(f'Saving to {save_path}{f}.*')
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}{f}.pdf',
                        bbox_inches='tight', format='pdf', dpi=200)
            plt.savefig(f'{save_path}{f}.pgf', bbox_inches='tight', dpi=200)
    else:
        plt.show()


def fpp_sde_psdpdf(data=True, save=False):
    # file = f'{data_path}fpp_sde_psdpdf.npz'
    # file = f'{data_path}fpp_sde.npz'
    file = f'{data_path}fpp_sde_L.npz'
    gamma = [.01, .1, 1.]
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
            print(f'Saving to {save_path}{f}.*')
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}{f}.pdf',
                        bbox_inches='tight', format='pdf', dpi=200)
            plt.savefig(f'{save_path}{f}.pgf', bbox_inches='tight', dpi=200)
    else:
        plt.show()


def fpp_tw_real(data=True, save=False):
    file = f'{data_path}fpp_tw_real.npz'
    rate = ['n-random', 'n-random']
    figs = ['cox', 'tick']
    gamma = [.01, .1, 1.]
    if not data:
        p = slf.FPPProcess()
        dt = 1e-2
        N = int(1e6)
        snr = .0
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
    del f

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


def fpp_tw_dist(data: bool = True, save=False):
    file = f'{data_path}fpp_tw_dist.npz'
    rate = ['n-random', 'n-random', 'n-random']
    figs = ['pareto', 'cox', 'tick']
    gamma = [.001, .01, .1, 1., 10.]
    if not data:
        p = slf.FPPProcess()
        dt = 1e-2
        N = int(1e6)
        snr = .0
        fpps = [[], [], []]
        for i, (r, tw) in enumerate(zip(rate, figs)):
            for g in gamma:
                p.set_params(gamma=g, K=int(N * g * dt), dt=dt,
                             tw=tw, snr=snr, rate=r)
                s = p.get_tw()
                # print(len(s[0]), len(s[1]))
                fpps[i].append(s)

        fpp_e = fpps[0]
        fpp_c = fpps[1]
        fpp_t = fpps[2]
        del fpps
        # np.savez(file, fpp_e=fpp_e, fpp_c=fpp_c, fpp_t=fpp_t)
    else:
        print(f'Loading data from {file}')
        f = np.load(file, allow_pickle=True)
        fpp_e = f['fpp_e']
        fpp_c = f['fpp_c']
        fpp_t = f['fpp_t']
        del f

    # # Create waiting times
    # TW_c = []
    # TW_t = []
    # while len(fpp_c) > 0:
    #     c = fpp_c.pop()
    #     tw_c = p.get_tw(parameter=c)
    #     TW_c.insert(0, tw_c)
    #     t = fpp_t.pop()
    #     tw_t = p.get_tw(parameter=t)
    #     TW_t.insert(0, tw_t)

    lab = [f'$\gamma = {g}$' for g in gamma]
    # tools.ridge_plot_psd(fpp_vr, dt, xlabel='$ f $', ylabel='$ S $', labels=lab, figname=figs[0])
    tools.ridge_plot(fpp_e, xlabel='$ \\tau_\mathrm{w} $', plt_type='loglog',
                         ylabel='$ P_{\\tau_{\mathrm{w}}} $', labels=lab, figname=figs[0])
    tools.ridge_plot(fpp_c, xlabel='$ \\tau_\mathrm{w} $', plt_type='loglog',
                         ylabel='$ P_{\\tau_{\mathrm{w}}} $', labels=lab, figname=figs[1])
    tools.ridge_plot(fpp_t, xlabel='$ \\tau_\mathrm{w} $', plt_type='loglog',
                         ylabel='$ P_{\\tau_{\mathrm{w}}} $', labels=lab, figname=figs[2])

    if save:
        for f in figs:
            print(f'Saving to {save_path}fpp_tw_dist_{f}.*')
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}fpp_tw_dist_{f}.pdf',
                        bbox_inches='tight', format='pdf', dpi=200)
            plt.savefig(f'{save_path}fpp_tw_dist_{f}.pgf', bbox_inches='tight', dpi=200)
    else:
        plt.show()


def fpp_tw_cox(data=True, save=False):
    file = f'{data_path}fpp_tw_cox.npz'
    rate = 'n-random'
    tw = 'pareto'
    figs = ['fpp', 'sde']
    gamma = 1.
    dt = 1e-2
    if not data:
        p = slf.FPPProcess()
        ps = slf.SDEProcess()
        N = int(1e6)
        snr = .0
        fpp = []
        sde = []
        p.set_params(gamma=gamma, K=int(N * gamma * dt), dt=dt,
                     tw=tw, snr=snr, rate=rate)
        ps.set_params(gamma=gamma, K=int(N * gamma * dt), dt=dt)
        s1, _, s2 = p.create_realisation(fit=False)
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
        del f

    # # Create waiting times
    # TW_c = []
    # TW_t = []
    # while len(fpp_c) > 0:
    #     c = fpp_c.pop()
    #     tw_c = p.get_tw(parameter=c)
    #     TW_c.insert(0, tw_c)
    #     t = fpp_t.pop()
    #     tw_t = p.get_tw(parameter=t)
    #     TW_t.insert(0, tw_t)

    lab = [f'$\gamma = {gamma}$']
    # tools.ridge_plot(fpp, xlabel='$ t $', ylabel='$ \Phi $', labels=lab, figname=figs)
    tools.ridge_plot_psd(fpp, dt, xlabel='$ f $',
                     ylabel='$ S $', labels=lab, figname=figs[0])
    tools.ridge_plot_psd(sde, dt, xlabel='$ f $',
                     ylabel='$ S $', labels=lab, figname=figs[1])

    if save:
        for f in figs:
            print(f'Saving to {save_path}fpp_tw_cox_{f}.*')
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}fpp_tw_cox_{f}.pdf',
                        bbox_inches='tight', format='pdf', dpi=200)
            plt.savefig(f'{save_path}fpp_tw_cox_{f}.pgf',
                        bbox_inches='tight', dpi=200)
    else:
        plt.show()


def sde_tw(data=True, save=False):
    # file_grab = f'{data_path}fpp_sde.npz'
    # gamma = [.1, 1., 10.]
    # N = int(1e4)
    file_grab = f'{data_path}fpp_sde_L.npz'
    gamma = [.01, .1, 1.]
    N = int(1e6)
    file = f'{data_path}sde.npz'
    figs = ['sde_tw']
    if not data:
        p = slf.SDEProcess()
        dt = 1e-2
        print(f'Loading data from {file}')
        f = np.load(file_grab, allow_pickle=True)
        sde = list(f['sde'])
        del f

        tw = []
        print(gamma[::-1])
        for g in gamma[::-1]:
            p.set_params(gamma=g, K=int(N * g * dt), dt=dt)
            # s1, s2 = sde.pop()
            # s2[s2 < 1e-1] = 0
            s = p.get_tw(sde.pop())
            print(len(s[1]), p.K, p.gamma)
            tw.insert(0, s)

        # np.savez(file, tw=tw)
    else:
        f = np.load(file, allow_pickle=True)
        tw = f['tw']
        for TW in tw:
            print(len(TW[1]))
    lab = [f'$\gamma = {g}$' for g in gamma]
    tools.ridge_plot(tw, xlabel='$ \\tau_\mathrm{w} $', ylabel='$ P_{\\tau_\mathrm{w}} $',
                     labels=lab, figname=figs[0], plt_type='loglog')

    if save:
        for f in figs:
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}{f}.pdf',
                        bbox_inches='tight', format='pdf', dpi=200)
            plt.savefig(f'{save_path}{f}.pgf', bbox_inches='tight')
    else:
        plt.show()


def fpptw_sde_real(data=True, save=False):
    """Example of FPP with different waiting times and SDE realisations with varying gamma.

    Args:
        data (bool, optional): use stored data from .npz if True, create new if False. Defaults to True.
        save (bool, optional): save if True, show if False. Defaults to False.
    """
    file = f'{data_path}fpptw_sde.npz'
    figs = ['fpp', 'sde']
    gamma = [.01, .1, 1.]
    rate = 'ou'
    tw = 'cox'
    if not data:
        pf = slf.FPPProcess()
        ps = slf.SDEProcess()
        N = int(1e6)
        dt = 1e-2
        snr = .0
        fpp = []
        sde = []
        for g in gamma:
            pf.set_params(gamma=g, K=int(N * g * dt), dt=dt,
                          snr=snr, rate=rate, tw=tw)
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
    tools.ridge_plot(fpp, xlabel='$ t $', ylabel='$\Phi$',
                     labels=lab, figname=figs[0])
    tools.ridge_plot(sde, xlabel='$ t $', ylabel='$\Phi$',
                     labels=lab, figname=figs[1])

    if save:
        for f in figs:
            print(f'Saving to {save_path}comp_{f}.*')
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}comp_{f}.pdf',
                        bbox_inches='tight', format='pdf', dpi=200)
            plt.savefig(f'{save_path}comp_{f}.pgf', bbox_inches='tight', dpi=200)
    else:
        plt.show()


def fpptw_sde_psd(data=True, save=False):
    file = f'{data_path}fpptw_sde.npz'
    gamma = [.01, .1, 1.]
    figs = ['fpp_psd', 'sde_psd']
    dt = 1e-2

    print(f'Loading data from {file}')
    f = np.load(file, allow_pickle=True)
    fpp = f['fpp']
    sde = f['sde']

    lab = [f'$\gamma = {g}$' for g in gamma]
    tools.ridge_plot_psd(fpp, dt, xlabel='$ f $',
                         ylabel='$ S $', labels=lab, figname=figs[0])
    tools.ridge_plot_psd(sde, dt, xlabel='$ f $',
                         ylabel='$ S $', labels=lab, figname=figs[1])

    if save:
        for f in figs:
            print(f'Saving to {save_path}comp_{f}.*')
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}comp_{f}.pdf',
                        bbox_inches='tight', format='pdf', dpi=200)
            plt.savefig(f'{save_path}comp_{f}.pgf', bbox_inches='tight', dpi=200)
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


def amplitude_dist(data=True, save=False):
    """Plot psd and pdf for different amp distributions.

    Observe: (as expected)
        psd do not change
        pdf do change
    """
    file = f'{data_path}amps.npz'
    amp = ['exp', 'pareto', 'gam', 'unif', 'ray', 'deg']  # 'alap'
    p = slf.FPPProcess()
    N = int(1e5)
    dt = 1e-2
    g = 1.
    figs = ['psd1', 'psd2', 'pdf']
    data1 = []
    data2 = []
    amps = []
    if not data:
        for i, a in enumerate(amp):
            p.set_params(gamma=g, K=int(N * g * dt), dt=dt, amp=a, snr=0.)
            s = p.create_realisation(fit=False)
            sig = s[-1]
            sig = (sig - sig.mean()) / sig.std()
            # p.plot_realisation('plot_real', parameter=s)
            if i < 3:
                data1.append([s[0], sig])
            else:
                data2.append([s[0], sig])
            # x, y = tools.est_pdf(s[-1])
            y, _, x = sa.distribution(sig, 100)
            amps.append([x, y])

            # plt.figure(figs[0])
            # plt.subplot(3, 2, i + 1)
            # plt.title(f'Adist = {a}')
            # p.plot_realisation('plot_psd', parameter=s[-1], fs=dt, new_fig=False)
            # # tools.psd(s[-1], new_fig=False)

            # plt.figure(figs[1])
            # plt.subplot(3, 2, i + 1)
            # plt.title(f'Adist = {a}')
            # p.plot_realisation('plot_pdf', parameter=s[-1], new_fig=False)
            # # tools.pdf(s[-1], new_fig=False)

        np.savez(file, data1=data1, data2=data2, amps=amps)
    else:
        print(f'Loading data from {file}')
        f = np.load(file, allow_pickle=True)
        data1 = f['data1']
        data2 = f['data2']
        amps = f['amps']

    lab = [f'{a}' for a in amp]
    tools.ridge_plot_psd(data1, dt, xlabel='$ f $', ylabel='PSD', labels=lab[:3], figname=figs[0])
    tools.ridge_plot_psd(data2, dt, xlabel='$ f $', ylabel='PSD', labels=lab[3:], figname=figs[1])
    tools.ridge_plot(amps, xlabel='$ \Phi $', ylabel='PDF', labels=lab, figname=figs[2], y_scale=.72)
    # plt.figure(figs[2])
    # clrs = [(r, 0, 0) for r in np.linspace(0, 1, len(lab))]
    # line_styles = ['-', '--', '-.', ':',
    #                (0, (3, 5, 1, 5, 1, 5)),
    #                (0, (3, 1, 1, 1, 1, 1))]
    # for l, ll, a in zip(clrs, lab, amps):
    #     plt.plot(a[0], a[1], color=l, label=ll, alpha=.8)
    # plt.legend()

    if save:
        for f in figs:
            print(f'Saving to {save_path}adist_{f}.*')
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}adist_{f}.pdf',
                        bbox_inches='tight', format='pdf', dpi=200)
            plt.savefig(f'{save_path}adist_{f}.pgf',
                        bbox_inches='tight', dpi=200)
    else:
        plt.show()


if __name__ == '__main__':
    # fpp_example()
    # fpp_sde_realisations()
    # fpp_sde_real_L()
    # fpp_sde_psdpdf()
    # fpp_tw_real()
    # fpp_tw_psd()
    # fpp_tw_dist(data=False)
    fpp_tw_cox(data=False)
    # sde_tw()
    # fpptw_sde_real()
    # fpptw_sde_psd()
    # power_law_pulse()
    # amplitude_dist()
