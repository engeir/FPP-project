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


def fpp_tw_dist(data: bool = True, save=False):
    file = f'{data_path}fpp_tw_dist.npz'
    rate = ['n-random', 'n-random', 'n-random']
    figs = ['pareto']
    gamma = [.01, .1, 1., 10.]
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
                s = (s[0]) / s[0].mean()
                # y, _, x = sa.distribution(s, 100)
                # s = (x, y)
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

    # Create waiting times
    TW_e = []
    corr = []
    TW_c = []
    TW_t = []
    while len(fpp_e) > 0:
        e = fpp_e.pop()
        c = np.correlate(e, e, 'same') / np.correlate(e, e)
        corr.insert(0, (np.linspace(0, 1, len(e)), c))
        y, _, x = sa.distribution(e, 100)
        s = (x, y)
        # tw_e = p.get_tw(parameter=e)
        TW_e.insert(0, s)
        # c = fpp_c.pop()
        # tw_c = p.get_tw(parameter=c)
        # TW_c.insert(0, tw_c)
        # t = fpp_t.pop()
        # tw_t = p.get_tw(parameter=t)
        # TW_t.insert(0, tw_t)
    fpp_e = TW_e

    lab = [f'$\gamma = {g}$' for g in gamma]
    plt_type = 'semilogy'
    # tools.ridge_plot_psd(fpp_vr, dt, xlabel='$ f $', ylabel='$ S $', labels=lab, figname=figs[0])
    tools.ridge_plot(corr, 'grid', xlabel='$ \\tau_\mathrm{w} $', plt_type='plot',
                     ylabel='Correlate', labels=lab, figname='corr')
    tools.ridge_plot(fpp_e, 'grid', xlabel='$ \\tau_\mathrm{w} $', plt_type=plt_type,
                     ylabel='$ P_{\\tau_{\mathrm{w}}} $', labels=lab, figname=figs[0])
    # tools.ridge_plot(fpp_c, 'grid', xlabel='$ \\tau_\mathrm{w} $', plt_type=plt_type,
    #                  ylabel='$ P_{\\tau_{\mathrm{w}}} $', labels=lab, figname=figs[1])
    # tools.ridge_plot(fpp_t, 'grid', xlabel='$ \\tau_\mathrm{w} $', plt_type=plt_type,
    #                  ylabel='$ P_{\\tau_{\mathrm{w}}} $', labels=lab, figname=figs[2])

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


def fpp_tw_pareto(data=True, save=False):
    file = f'{data_path}fpp_tw_pareto.npz'
    rate = 'n-random'
    tw_ = 'pareto'
    figs = ['fpp', 'sde']
    gamma = .1
    dt = 1e-2
    if not data:
        p = slf.FPPProcess()
        ps = slf.SDEProcess()
        N = int(5e5)  # 2e6 in saved arr
        snr = .0
        sig = []
        force = []
        amp = []
        tw = []
        p.set_params(gamma=gamma, K=int(N * gamma * dt), dt=dt,
                     tw=tw_, snr=snr, rate=rate, amp='exp', kern='1exp')
        ps.set_params(gamma=gamma, K=int(N * gamma * dt), dt=dt)

        t, f, r, a, ta = p.create_realisation(fit=False, full=True)
        s = (t, r)
        trw = np.diff(ta)
        TW = np.sort(trw)[::-1]
        sig.append(s)
        force.append(f)
        amp.append(a)
        tw.append(TW)

        t, r, ta, a, f = ps.get_tw()
        s = (t, r)
        trw = np.diff(ta)
        TW = np.sort(trw)[::-1]
        sig.append(s)
        force.append(f)
        amp.append(a)
        tw.append(TW)

        # np.savez(file, sig=sig, force=force, amp=amp, tw=tw)
    else:
        print(f'Loading data from {file}')
        f = np.load(file, allow_pickle=True)
        sig = f['sig']
        # force = f['force']
        # amp = f['amp']
        # tw = list(f['tw'])
        del f

    # # Create histogram of waiting times
    # for _ in range(len(tw)):
    #     c = tw.pop()
    #     c /= c.mean()
    #     y, _, x = sa.distribution(c, 100)
    #     tw.insert(0, (x, y))

    lab = [f'{f}' for f in figs]
    tools.ridge_plot(sig, xlabel='$ t $', ylabel='$ \Phi $', labels=lab, figname='sig')
    # tools.ridge_plot(tw, xlabel='$ t $', ylabel='$ \Phi $', labels=lab, figname='tw', plt_type='semilogy')
    tools.ridge_plot_psd(sig, dt, 'squeeze', xlabel='$ f $',
                     ylabel='$ S $', labels=lab, figname='psd')
    # tools.ridge_plot_psd(, dt, xlabel='$ f $',
    #                  ylabel='$ S $', labels=lab, figname=figs[1])

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


def test_FPP(data=False, save=False):
    file = f'{data_path}test_fpp.npz'
    rate = 'n-random'
    tw_ = 'pareto'
    figs = ['fpp_1']
    gamma = [.1]
    dt = 1e-2
    if not data:
        p = slf.FPPProcess()
        N = int(5e6)  # 2e6 in saved arr
        snr = .0
        sig = []
        force = []
        amp = []
        tw = []
        for g in gamma:
            p.set_params(gamma=g, K=int(N * g * dt), dt=dt, TWkappa=.5,
                         tw=tw_, snr=snr, rate=rate, amp='pareto', kern='1exp')

            t, f, r, a, ta = p.create_realisation(fit=False, full=True)
            s = (t, r)
            trw = np.diff(ta)
            TW = np.sort(trw)[::-1]
            sig.append(s)
            force.append(f)
            amp.append(a)
            tw.append(TW)

        # np.savez(file, sig=sig, force=force, amp=amp, tw=tw)
    else:
        print(f'Loading data from {file}')
        f = np.load(file, allow_pickle=True)
        sig = f['sig']
        # force = f['force']
        # amp = f['amp']
        # tw = list(f['tw'])
        del f

    # # Create histogram of waiting times
    # for _ in range(len(tw)):
    #     c = tw.pop()
    #     c /= c.mean()
    #     y, _, x = sa.distribution(c, 100)
    #     tw.insert(0, (x, y))

    lab = [f'$\gamma = {g}$' for g in gamma]
    # tools.ridge_plot(sig, xlabel='$ t $', ylabel='$ \Phi $',
    #                  labels=lab, figname='sig')
    # tools.ridge_plot(tw, xlabel='$ t $', ylabel='$ \Phi $', labels=lab, figname='tw', plt_type='semilogy')
    tools.ridge_plot_psd(sig, dt, 'squeeze', xlabel='$ f $',
                         ylabel='$ S $', labels=lab, figname=figs[0])
    # tools.ridge_plot_psd(, dt, xlabel='$ f $',
    #                  ylabel='$ S $', labels=lab, figname=figs[1])

    if save:
        for f in figs:
            print(f'Saving to {save_path}test_FPP_{f}.*')
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}test_FPP_{f}.pdf',
                        bbox_inches='tight', format='pdf', dpi=200)
            plt.savefig(f'{save_path}test_FPP_{f}.pgf',
                        bbox_inches='tight', dpi=200)
    else:
        plt.show()


if __name__ == '__main__':
    # fpp_tw_dist(data=False)
    # fpp_tw_pareto(data=False)
    # fpptw_sde_real()
    # fpptw_sde_psd()
    # power_law_pulse()
    test_FPP()
