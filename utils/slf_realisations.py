import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
import scipy.signal as ssi

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
    # file = f'{data_path}fpp_tw_dist.npz'
    file = f'{data_path}fpp_tw_dist_ampPareto.npz'
    rate = ['n-random', 'n-random', 'n-random', 'n-random']
    tw_distr = ['exp', 'pareto', 'cox', 'tick']
    figs = ['tw_exp', 'tw_pareto', 'tw_cox', 'tw_tick', 'corr_exp', 'corr_pareto', 'corr_cox', 'corr_tick']
    gamma = [.01, .1, 1., 10.]
    if not data:
        p = slf.FPPProcess()
        dt = 1e-2
        N = int(1e7)
        snr = .0
        fpps = [[], [], [], []]
        corrs = [[], [], [], []]
        for i, (r, tw) in enumerate(zip(rate, tw_distr)):
            for g in gamma:
                p.set_params(gamma=g, K=int(N * g * dt), dt=dt,
                             tw=tw, snr=snr, rate=r, amp='pareto')
                s = p.get_tw()
                fpps[i].append(s[0])
                s_ = s[0]
                s_ = (s_ - s_.mean()) / s_.std()
                corrs[i].append([np.linspace(-1, 1, len(s_)),
                                np.correlate(s_, s_, 'same') / np.correlate(s_, s_)])

        fpp_e = fpps[0]
        fpp_p = fpps[1]
        fpp_c = fpps[2]
        fpp_t = fpps[3]
        corr_e = corrs[0]
        corr_p = corrs[1]
        corr_c = corrs[2]
        corr_t = corrs[3]
        del fpps
        np.savez(file, fpp_e=fpp_e, fpp_p=fpp_p, fpp_c=fpp_c, fpp_t=fpp_t, corr_e=corr_e, corr_p=corr_p, corr_c=corr_c, corr_t=corr_t)
    else:
        print(f'Loading data from {file}')
        f = np.load(file, allow_pickle=True)
        fpp_e = f['fpp_e']
        fpp_p = f['fpp_p']
        fpp_c = f['fpp_c']
        fpp_t = f['fpp_t']
        corr_e = f['corr_e']
        corr_p = f['corr_p']
        corr_c = f['corr_c']
        corr_t = f['corr_t']
        del f

    # Calculate waiting time PDFs
    TW_e = []
    TW_p = []
    TW_c = []
    TW_t = []
    for e, p, c, t in zip(fpp_e, fpp_p, fpp_c, fpp_t):
        e /= e.std()
        y, _, x = sa.distribution(e, 50)
        s = (x, y)
        TW_e.append(s)
        p /= p.std()
        y, _, x = sa.distribution(p, 50)
        s = (x, y)
        TW_p.append(s)
        c /= c.std()
        y, _, x = sa.distribution(c, 50)
        s = (x, y)
        TW_c.append(s)
        t /= t.std()
        y, _, x = sa.distribution(t, 50)
        s = (x, y)
        TW_t.append(s)
    del fpp_e
    del fpp_p
    del fpp_c
    del fpp_t

    lab = [f'$\gamma = {g}$' for g in gamma]
    plt_type = 'loglog'
    plt_type_c = 'plot'
    xlim_c = (-.2, .2)
    # tools.ridge_plot_psd(fpp_vr, dt, xlabel='$ f $', ylabel='$ S $', labels=lab, figname=figs[0])
    tools.ridge_plot(TW_e, 'grid', 'dots', xlabel='$ \\tau_\mathrm{w} $', plt_type=plt_type,
                     ylabel='$ P_{\\tau_{\mathrm{w}}} $', labels=lab, figname=figs[0])
    tools.ridge_plot(TW_p, 'grid', 'dots', xlabel='$ \\tau_\mathrm{w} $', plt_type=plt_type,
                     ylabel='$ P_{\\tau_{\mathrm{w}}} $', labels=lab, figname=figs[1])
    tools.ridge_plot(TW_c, 'grid', 'dots', xlabel='$ \\tau_\mathrm{w} $', plt_type=plt_type,
                     ylabel='$ P_{\\tau_{\mathrm{w}}} $', labels=lab, figname=figs[2])
    tools.ridge_plot(TW_t, 'grid', 'dots', xlabel='$ \\tau_\mathrm{w} $', plt_type=plt_type,
                     ylabel='$ P_{\\tau_{\mathrm{w}}} $', labels=lab, figname=figs[3])
    tools.ridge_plot(corr_e, 'grid', xlabel='$ \\tau_\mathrm{w} $', plt_type=plt_type_c,
                     ylabel='Correlate', labels=lab, figname=figs[4], xlim=xlim_c)
    tools.ridge_plot(corr_p, 'grid', xlabel='$ \\tau_\mathrm{w} $', plt_type=plt_type_c,
                     ylabel='Correlate', labels=lab, figname=figs[5], xlim=xlim_c)
    tools.ridge_plot(corr_c, 'grid', xlabel='$ \\tau_\mathrm{w} $', plt_type=plt_type_c,
                     ylabel='Correlate', labels=lab, figname=figs[6], xlim=xlim_c)
    tools.ridge_plot(corr_t, 'grid', xlabel='$ \\tau_\mathrm{w} $', plt_type=plt_type_c,
                     ylabel='Correlate', labels=lab, figname=figs[7], xlim=xlim_c)

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


def test_FPP(data=True, save=False):
    file = f'{data_path}test_fpp.npz'
    rate = 'n-random'
    tw_ = 'cox'
    figs = ['fpp_1']
    gamma = [.01]
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
            TW = np.diff(ta)
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


def test_compare(data=True, save=False):
    file = f'{data_path}test_compare.npz'
    rate = 'n-random'
    tw_ = 'cox'
    figs = ['fpp_1']
    gamma = [1e-1]
    dt = 1e-2
    if not data:
        pf = slf.FPPProcess()
        ps = slf.SDEProcess()
        N = int(5e6)  # 2e6 in saved arr
        snr = .0
        sig = []
        for g in gamma:
            pf.set_params(gamma=g, K=int(N * g * dt), dt=dt, TWkappa=.5,
                         tw=tw_, snr=snr, rate=rate, amp='pareto', kern='1exp')
            ps.set_params(gamma=g, K=int(N * g * dt), dt=dt)

            t, f, r = pf.create_realisation(fit=False)
            s = (t, r)
            sig.append(s)
            t, r = ps.create_realisation(fit=False)
            s = (t, r)
            sig.append(s)

        np.savez(file, sig=sig)
    else:
        print(f'Loading data from {file}')
        f = np.load(file, allow_pickle=True)
        sig = f['sig']
        del f

    # Create PSD
    psd = []
    first = True
    for s in sig:
        s = s[1]
        Xn = (s - s.mean()) / s.std()
        if first:
            # Create Lorentz spectrum
            fp, P_Xn = ssi.periodogram(Xn, fs=dt)
            w = 2 * np.pi * fp[1:]
            t_d = (1 / dt)**2
            lor = 2 * t_d / (1 + (t_d * w)**2)
            first = False
            psd.append([fp[1:], lor])

        f, S_Xn = ssi.welch(Xn, fs=dt, nperseg=2**18)
        psd.append([f[1:], S_Xn[1:]])

    lab = ['Lorentz spectrum', f'FPP — {tw_}', 'SDE']
    plot_f = getattr(plt, 'loglog')
    plt.figure()
    for i, p in enumerate(psd[::-1]):
        alpha = 1 if i == 2 else .7
        l = '--k' if i == 2 else '-'
        plot_f(p[0], p[1], l, alpha=alpha)
    plt.legend(lab[::-1])

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
    # fpp_tw_dist()
    # fpp_tw_pareto(data=False)
    # fpptw_sde_real()
    # fpptw_sde_psd()
    # power_law_pulse()
    # test_FPP()
    test_compare(data=False)
