import sys
sys.path.append('/Users/eirikenger/uit_scripts')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec

import slf
import tools
import plot_utils as pu

import uit_scripts.stat_analysis as sa
from uit_scripts.plotting import figure_defs as fd

fd.set_rcparams_article_thickline(plt.rcParams)
plt.rcParams['font.family'] = 'DejaVu Sans'

data_path = '/home/een023/Documents/FPP_SOC_Chaos/report/data/'
# save_path = '/home/een023/Documents/FPP_SOC_Chaos/report/figures/'
save_path = ''


# Figure 1
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
        print(f'Saving to {save_path}{figs}.*')
        plt.tight_layout()
        plt.savefig(f'{save_path}{figs}.pdf',
                    bbox_inches='tight', format='pdf', dpi=600)
        plt.savefig(f'{save_path}{figs}.pgf', bbox_inches='tight')
    else:
        plt.show()


# Figure 2
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
            if i < 3:
                data1.append([s[0], sig])
            else:
                data2.append([s[0], sig])
            y, _, x = sa.distribution(sig, 100)
            amps.append([x, y])

        np.savez(file, data1=data1, data2=data2, amps=amps)
    else:
        print(f'Loading data from {file}')
        f = np.load(file, allow_pickle=True)
        data1 = f['data1']
        data2 = f['data2']
        amps = f['amps']

    lab = [f'{a}' for a in amp]
    tools.ridge_plot_psd(data1, dt, xlabel='$ f $',
                         ylabel='PSD', labels=lab[:3], figname=figs[0])
    tools.ridge_plot_psd(data2, dt, xlabel='$ f $',
                         ylabel='PSD', labels=lab[3:], figname=figs[1])
    tools.ridge_plot(amps, xlabel='$ \Phi $', ylabel='PDF',
                     labels=lab, figname=figs[2], y_scale=.72)

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


# Figure 3
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
    tools.ridge_plot(fpp, xlabel='$ t $', ylabel='$ \Phi $',
                     labels=lab, figname=figs[0])
    tools.ridge_plot(sde, xlabel='$ t $', ylabel='$ \Phi $',
                     labels=lab, figname=figs[1])

    if save:
        for f in figs:
            print(f'Saving to {save_path}{f}.*')
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}{f}.pdf',
                        bbox_inches='tight', format='pdf', dpi=200)
            plt.savefig(f'{save_path}{f}.pgf', bbox_inches='tight')
    else:
        plt.show()


# Figure 4
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
    tools.ridge_plot(fpp, xlabel='$ t $', ylabel='$\Phi$',
                     labels=lab, figname=figs[0])
    tools.ridge_plot(sde, xlabel='$ t $', ylabel='$\Phi$',
                     labels=lab, figname=figs[1])

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


# Figure 5
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
        N = int(1e6)
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
    tools.ridge_plot_psd(fpp, dt, xlabel='$ f $',
                         ylabel='$ S $', labels=lab, figname=figs[0])
    tools.ridge_plot_psd(sde, dt, xlabel='$ f $',
                         ylabel='$ S $', labels=lab, figname=figs[1])

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


#Figure 6
def sde_tw(data=True, save=False):
    # file_grab = f'{data_path}fpp_sde.npz'
    # gamma = [.1, 1., 10.]
    # N = int(1e4)
    # file_grab = f'{data_path}fpp_sde_L.npz'
    gamma = [.01, .1, 1., 10.]
    N = int(1e7)
    dt = 1e-2
    file = 'sde.npz'
    # file = f'{data_path}sde_w11.npz'
    figs = ['sde', 'tw', 'amp', 'corr']
    if not data:
        p = slf.SDEProcess()
        # print(f'Loading data from {file}')
        # f = np.load(file_grab, allow_pickle=True)
        # sde = list(f['sde'])
        # del f

        sde = []
        ta = []
        amp = []
        force = []
        print(gamma[::-1])
        for g in gamma[::-1]:
            p.set_params(gamma=g, K=int(N * g * dt), dt=dt)
            # s1, s2 = sde.pop()
            # s2[s2 < 1e-1] = 0
            t, s, TA, AMP, f = p.get_tw()  # sde.pop()
            print(len(TA), p.K, p.gamma)
            # y, _, x = sa.distribution(s1, 10)
            # s = (x, y)
            sde.insert(0, (t, s))
            ta.insert(0, TA)
            amp.insert(0, AMP)
            force.insert(0, f)

        np.savez(file, sde=sde, ta=ta, amp=amp, f=force)
    else:
        print(f'Loading data from {file}')
        f = np.load(file, allow_pickle=True)
        sde = f['sde']
        ta = f['ta']
        amp = f['amp']
        force = f['f']

    # F = []
    # for ff in force:
    #     F.append((np.linspace(0, N * dt, len(ff)), ff))
    # force = F
    TW = []
    AMP = []
    corr = []
    for i, (s, a) in enumerate(zip(ta, amp)):
        if i == 4:
            a = (a - a.mean()) / a.std()
            mask = a > 0
            plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(a[mask], '-or')
            plt.subplot(3, 1, 2)
            plt.plot(s[mask], '-or')
            plt.subplot(3, 1, 3)
            plt.plot(np.diff(s[mask]))
            plt.show()

        a = (a - 0) / a.std()
        # s = np.diff(s[a > 1e-10])
        s = np.diff(s)
        co = (s - s.mean()) / s.std()
        s = (s - 0) / s.std()
        corr.append((np.linspace(-1, 1, len(co)),
                     np.correlate(co, co, 'same') / np.correlate(co, co)))
        y, _, x = sa.distribution(a, 100)
        AMP.append((x, y))
        y, _, x = sa.distribution(s, 100)
        TW.append((x, y))
    tw = TW
    amp = AMP
    lab = [f'$\gamma = {g}$' for g in gamma]
    # f0 = force[0][0]
    # f1 = force[0][1]
    # # f1[f1 != 0] = 1
    # # print(len(f1[f1 > 1e-10]), len(f1))
    # plt.figure()
    # plt.scatter(f0[f1 == 0], f1[f1 == 0] - .02 * f1.max(), color='k')
    # plt.scatter(f0[(0 < f1) & (f1 < 1e-10)], f1[(0 < f1) & (f1 < 1e-10)] - .01 * f1.max(), color='r')
    # plt.scatter(f0[f1 > 1e-10], f1[f1 > 1e-10], color='b')
    # # plt.figure()
    # # plt.plot(sde[1][0], sde[1][1])
    plt.rcParams['lines.linewidth'] = .4
    tools.ridge_plot(sde, 'grid', xlabel='$ t $', ylabel='$ \Phi $',
                     labels=lab, figname=figs[0], plt_type='plot')  # , xlim=[0, 200])
    plt.rcParams['lines.linewidth'] = 1.5
    tools.ridge_plot(tw, 'grid', 'squeeze', 'dots', xlabel='$ t_k $', ylabel='$ P_{t_k} $',
                     labels=lab, figname=figs[1], plt_type='loglog')  # , xlim=[0, 200])
    tools.ridge_plot(amp, 'grid', 'squeeze', 'dots', xlabel='$ A_k $', ylabel='$ P_{A_k} $',
                     labels=lab, figname=figs[2], plt_type='loglog')  # , xlim=[0, 200])
    tools.ridge_plot(corr, 'slalomaxis', xlabel='Lag', ylabel='Correlation',
                     labels=lab, figname=figs[3], plt_type='plot')  # , xlim=[0, 200])

    if save:
        for f in figs:
            print(f'Saving to {save_path}{f}.*')
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(f'{save_path}sde_anlz_{f}.pdf',
                        bbox_inches='tight', format='pdf', dpi=200)
            # plt.savefig(f'{save_path}sde_anlz_{f}.pgf', bbox_inches='tight')
    else:
        plt.show()


# Figure 7
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

        fpp_c = fpps[0]
        fpp_t = fpps[1]
        del fpps
        np.savez(file, fpp_c=fpp_c, fpp_t=fpp_t)
    else:
        print(f'Loading data from {file}')
        f = np.load(file, allow_pickle=True)
        fpp_c = f['fpp_c']
        fpp_t = f['fpp_t']

    plt.rcParams['lines.linewidth'] = .4
    lab = [f'$\gamma = {g}$' for g in gamma]
    tools.ridge_plot(fpp_c, xlabel='$ t $', ylabel='$\Phi$',
                     labels=lab, figname=figs[0])
    tools.ridge_plot(fpp_t, xlabel='$ t $', ylabel='$\Phi$',
                     labels=lab, figname=figs[1])

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


# Figure 8
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
    tools.ridge_plot_psd(fpp_c, dt, xlabel='$ f $',
                         ylabel='$ S $', labels=lab, figname=figs[0])
    tools.ridge_plot_psd(fpp_t, dt, xlabel='$ f $',
                         ylabel='$ S $', labels=lab, figname=figs[1])

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

if __name__ == '__main__':
    # fpp_example()  # 1
    # amplitude_dist()  # 2
    # fpp_sde_realisations()  # 3
    # fpp_sde_real_L()  # 4
    # fpp_sde_psdpdf()  # 5
    sde_tw(save=True)  # 6
    # fpp_tw_real()  # 7
    # fpp_tw_psd()  # 8
    # fpp_tw_dist(data=False)
    # fpp_tw_cox()
    # fpptw_sde_real()
    # fpptw_sde_psd()
    # power_law_pulse()