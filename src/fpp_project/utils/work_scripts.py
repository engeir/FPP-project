import matplotlib.pyplot as plt
import numpy as np
import uit_scripts.stat_analysis as sa
from scipy.optimize import curve_fit
from uit_scripts.plotting import figure_defs as fd

import fpp_project.utils.slf as slf
import fpp_project.utils.tools as tools

# import matplotlib.gridspec as grid_spec
# import plot_utils as pu

fd.set_rcparams_article_thickline(plt.rcParams)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["pgf.texsystem"] = "pdflatex"

data_path = "/home/een023/Documents/work/FPP_SOC_Chaos/report/data/"
save_path = "/home/een023/Documents/work/FPP_SOC_Chaos/report/figures/"
# save_path = ''


# Figure 1
def fpp_example(data=True, save=False):
    """Example of FPP process, exponential everything."""
    figs = "fpp_example"
    file = f"{data_path}{figs}.npz"
    p = slf.FPPProcess()
    if not data:
        p = slf.SDEProcess()
        N = int(1e4)
        dt = 1e-2
        gamma = 0.1
        snr = 0.0
        p.set_params(gamma=gamma, K=int(N * gamma * dt), dt=dt, snr=snr)
        s = p.create_realisation(fit=False)
        pulse = p.fit_pulse(s[0], s[1], s[2])[0]

        np.savez(file, s=s, pulse=pulse)
    else:
        print(f"Loading data from {file}")
        with np.load(file, allow_pickle=True) as f:
            s = f["s"]
            pulse = f["pulse"]

    p.plot_realisation("plot_real", parameter=s, fit=False)
    plt.subplot(2, 1, 1)
    plt.plot(s[0], pulse, "r", label="Pulse")
    plt.legend()

    if save:
        print(f"Saving to {save_path}{figs}.*")
        plt.tight_layout()
        plt.savefig(
            f"{save_path}{figs}.pdf", bbox_inches="tight", format="pdf", dpi=600
        )
        plt.savefig(f"{save_path}{figs}.pgf", bbox_inches="tight")
    else:
        plt.show()


# Figure 2
def amplitude_dist(data=True, save=False):
    """Plot psd and pdf for different amp distributions.

    Observe: (as expected)
        psd do not change
        pdf do change
    """
    file = f"{data_path}amps.npz"
    amp = ["exp", "pareto", "gam", "unif", "ray", "deg"]  # 'alap'
    p = slf.FPPProcess()
    N = int(1e5)
    dt = 1e-2
    g = 1.0
    figs = ["psd1", "psd2", "pdf"]
    data1 = []
    data2 = []
    amps = []
    if not data:
        for i, a in enumerate(amp):
            p.set_params(gamma=g, K=int(N * g * dt), dt=dt, amp=a, snr=0.0)
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
        print(f"Loading data from {file}")
        with np.load(file, allow_pickle=True) as f:
            data1 = f["data1"]
            data2 = f["data2"]
            amps = f["amps"]

    lab = [f"{a}" for a in amp]
    tools.ridge_plot_psd(
        data1,
        dt,
        "squeeze",
        xlabel="$ f $",
        ylabel="PSD",
        labels=lab[:3],
        figname=figs[0],
    )
    tools.ridge_plot_psd(
        data2,
        dt,
        "squeeze",
        xlabel="$ f $",
        ylabel="PSD",
        labels=lab[3:],
        figname=figs[1],
    )
    tools.ridge_plot(
        amps,
        xlabel=r"$ \Phi $",
        ylabel="PDF",
        labels=lab,
        figname=figs[2],
        y_scale=0.64,
    )

    if save:
        for f in figs:
            print(f"Saving to {save_path}adist_{f}.*")
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(
                f"{save_path}adist_{f}.pdf", bbox_inches="tight", format="pdf", dpi=200
            )
            plt.savefig(f"{save_path}adist_{f}.pgf", bbox_inches="tight", dpi=200)
    else:
        plt.show()


# Figure 3
def fpp_sde_realisations(data=True, save=False):
    """Example of FPP and SDE realisations with varying gamma.

    Parameters
    ----------
    data: bool
        Use saved data (default) or generate new
    save: bool
        save if True, show if False. Defaults to False.
    """
    file = f"{data_path}fpp_sde.npz"
    gamma = [0.1, 1.0, 10.0]
    figs = ["fpp_gamma", "sde_gamma"]
    if not data:
        pf = slf.FPPProcess()
        ps = slf.SDEProcess()
        N = int(1e4)
        dt = 1e-2
        snr = 0.0
        fpp = []
        sde = []
        for g in gamma:
            pf.set_params(gamma=g, K=int(N * g * dt), dt=dt, snr=snr)
            ps.set_params(gamma=g, K=int(N * g * dt), dt=dt)

            s1, _, s2 = pf.create_realisation(fit=False)
            if g == 10.0:
                s2[:100] = np.nan
            s = (s1, s2)
            fpp.append(s)

            s = ps.create_realisation(fit=False)
            sde.append(s)

        np.savez(file, fpp=fpp, sde=sde)
    else:
        print(f"Loading data from {file}")
        with np.load(file, allow_pickle=True) as f:
            fpp = f["fpp"]
            sde = f["sde"]

    lab = [fr"$\gamma = {g}$" for g in gamma]
    tools.ridge_plot(
        fpp, xlabel="$ t $", ylabel=r"$ \Phi $", labels=lab, figname=figs[0]
    )
    tools.ridge_plot(
        sde, xlabel="$ t $", ylabel=r"$ \Phi $", labels=lab, figname=figs[1]
    )

    if save:
        for f in figs:
            print(f"Saving to {save_path}{f}.*")
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(
                f"{save_path}{f}.pdf", bbox_inches="tight", format="pdf", dpi=200
            )
            plt.savefig(f"{save_path}{f}.pgf", bbox_inches="tight")
    else:
        plt.show()


# Figure 4
def fpp_sde_real_L(data=True, save=False):
    """Example of FPP and SDE realisations with varying gamma.

    Args:
        save (bool, optional): save if True, show if False. Defaults to False.
    """
    file = f"{data_path}fpp_sde_L.npz"
    gamma = [0.01, 0.1, 1.0]
    figs = ["fpp_gamma_L", "sde_gamma_L"]
    if not data:
        pf = slf.FPPProcess()
        ps = slf.SDEProcess()
        N = int(1e6)
        dt = 1e-2
        snr = 0.0
        fpp = []
        sde = []
        for g in gamma:
            pf.set_params(gamma=g, K=int(N * g * dt), dt=dt, snr=snr)
            ps.set_params(gamma=g, K=int(N * g * dt), dt=dt)

            s1, _, s2 = pf.create_realisation(fit=False)
            if g == 10.0:
                s2[:100] = np.nan
            s = (s1, s2)
            fpp.append(s)

            s = ps.create_realisation(fit=False)
            sde.append(s)

        np.savez(file, fpp=fpp, sde=sde)
    else:
        print(f"Loading data from {file}")
        with np.load(file, allow_pickle=True) as f:
            fpp = f["fpp"]
            sde = f["sde"]

    plt.rcParams["lines.linewidth"] = 0.4
    lab = [fr"$\gamma = {g}$" for g in gamma]
    tools.ridge_plot(fpp, xlabel="$ t $", ylabel=r"$\Phi$", labels=lab, figname=figs[0])
    tools.ridge_plot(sde, xlabel="$ t $", ylabel=r"$\Phi$", labels=lab, figname=figs[1])

    if save:
        for f in figs:
            print(f"Saving to {save_path}{f}.*")
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(
                f"{save_path}{f}.pdf", bbox_inches="tight", format="pdf", dpi=200
            )
            plt.savefig(f"{save_path}{f}.pgf", bbox_inches="tight", dpi=200)
    else:
        plt.show()


# Figure 5
def fpp_sde_psdpdf(data=True, save=False):
    # file = f'{data_path}fpp_sde_psdpdf.npz'
    # file = f'{data_path}fpp_sde.npz'
    file = f"{data_path}fpp_sde_L.npz"
    gamma = [0.01, 0.1, 1.0]
    figs = ["fpp_psd", "sde_psd"]
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
        print(f"Loading data from {file}")
        with np.load(file, allow_pickle=True) as f:
            fpp = f["fpp"]
            sde = f["sde"]

    lab = [fr"$\gamma = {g}$" for g in gamma]
    tools.ridge_plot_psd(
        fpp, dt, "squeeze", xlabel="$ f $", ylabel="$ S $", labels=lab, figname=figs[0]
    )
    tools.ridge_plot_psd(
        sde, dt, "squeeze", xlabel="$ f $", ylabel="$ S $", labels=lab, figname=figs[1]
    )

    if save:
        for f in figs:
            print(f"Saving to {save_path}{f}.*")
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(
                f"{save_path}{f}.pdf", bbox_inches="tight", format="pdf", dpi=200
            )
            plt.savefig(f"{save_path}{f}.pgf", bbox_inches="tight", dpi=200)
    else:
        plt.show()


# Figure 6
def sde_tw(data=True, save=False):
    # file_grab = f'{data_path}fpp_sde.npz'
    # gamma = [.1, 1., 10.]
    # N = int(1e4)
    # file_grab = f'{data_path}fpp_sde_L.npz'
    gamma = [0.01, 0.1, 1.0, 10.0]
    N = int(1e7)
    dt = 1e-2
    # file = 'sde.npz'
    file = f"{data_path}sde_L.npz"
    # file = f'{data_path}sde_w11.npz'
    figs = ["sde", "tw", "amp", "corr"]
    if not data:
        p = slf.SDEProcess()
        # print(f'Loading data from {file}')
        # with np.load(file_grab, allow_pickle=True) as f:
        #     sde = list(f['sde'])

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
        print(f"Loading data from {file}")
        with np.load(file, allow_pickle=True) as f:
            # sde = f['sde']
            ta = f["ta"]
            amp = f["amp"]
            # force = f['f']

    # F = []
    # for ff in force:
    #     F.append((np.linspace(0, N * dt, len(ff)), ff))
    # force = F
    TW = []
    AMP = []
    corr = []
    tw_exp = []
    amp_exp = []
    tw_std = []
    # twtxt_x = 20
    # twtxt_y = [5e4, 5e1, 1e-1, 2e-4]
    tw_fit = []
    amp_std = []
    # amptxt_y = [1e7, 5e3, 2e0, 2e-4]
    amp_fit = []
    for i, (s, a) in enumerate(zip(ta, amp)):
        if i == 4:
            a = (a - a.mean()) / a.std()
            mask = a > 0
            plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(a[mask], "-or")
            plt.subplot(3, 1, 2)
            plt.plot(s[mask], "-or")
            plt.subplot(3, 1, 3)
            plt.plot(np.diff(s[mask]))
            plt.show()

        amp_std.append(a.std())
        a = (a - 0) / a.std()
        # s = np.diff(s[a > 1e-10])
        s = np.diff(s)
        co = (s - s.mean()) / s.std()
        tw_std.append(s.std())
        s = (s - 0) / s.std()
        corr.append(
            (
                np.linspace(-1, 1, len(co)),
                np.correlate(co, co, "same") / np.correlate(co, co),
            )
        )
        y, _, x = sa.distribution(a, 70)
        AMP.append((x, y))
        mask = (x > 2.6) & (x < 10)
        x1 = x[mask]
        y1 = y[mask]
        if i in (2, 3):
            popt, _ = curve_fit(tools.exp_func, x1, y1)
            amp_exp.append((x, tools.exp_func(x, *popt)))
            amp_fit.append(fr"$\mathrm{{exp}} = - {popt[1]:2.2f}$")
        else:
            popt, _ = curve_fit(tools.pow_func, x1, y1)
            amp_exp.append((x, tools.pow_func(x, *popt)))
            amp_fit.append(fr"$\mathrm{{pow}} = - {popt[1]:2.2f}$")
        y, _, x = sa.distribution(s, 71)
        TW.append((x, y))
        mask = (x > 2.7) & (x < 10)
        # mask = (x > 1.3) & (x < 14)
        x1 = x[mask]
        y1 = y[mask]
        if i in (2, 3):
            popt, _ = curve_fit(tools.exp_func, x1, y1)
            tw_exp.append((x, tools.exp_func(x, *popt)))
            tw_fit.append(fr"$\mathrm{{exp}} = - {popt[1]:2.2f}$")
        else:
            popt, _ = curve_fit(tools.pow_func, x1, y1)
            tw_exp.append((x, tools.pow_func(x, *popt)))
            tw_fit.append(fr"$\mathrm{{pow}} = - {popt[1]:2.2f}$")
    # tw = TW
    amp = AMP
    lab = [fr"$\gamma = {g}$" for g in gamma]
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

    # Plotting the four figures that are used
    plt.rcParams["lines.linewidth"] = 0.4
    # tools.ridge_plot(sde, 'grid', xlabel='$ t $', ylabel='$ \Phi $',
    #                  labels=lab, figname=figs[0], plt_type='plot')  # , xlim=[0, 200])
    plt.rcParams["lines.linewidth"] = 1.5
    # tools.ridge_plot(tw_exp, 'squeeze', 'blank', color='k', lt='--',
    #                  labels=lab, figname=figs[1], plt_type='loglog', y_scale=1.13, ylim=(5e-6, 1e0), xlim=[1e-1, 1e2])
    # tools.ridge_plot(tw, 'grid', 'squeeze', 'dots', xlabel='$ \\tau_{\mathrm{w},k} $', ylabel='$ P_{\\tau_{\mathrm{w},k}} $',
    #                  labels=lab, figname=figs[1], plt_type='loglog', y_scale=1.13, ylim=(5e-6, 1e0), xlim=[1e-1, 1e2])
    # for yy, txt, txt2 in zip(twtxt_y, tw_std, tw_fit):
    #     plt.text(twtxt_x, yy, f'{txt2}', horizontalalignment='left',
    #             verticalalignment='bottom', size=5)
    #     plt.text(twtxt_x, yy, f'$ \mathrm{{std}} = {txt:2.2f}$', horizontalalignment='left',
    #             verticalalignment='top', size=5)
    # tools.ridge_plot(amp_exp, 'squeeze', 'blank', color='k', lt='--',
    #                  labels=lab, figname=figs[2], plt_type='loglog', y_scale=1.13, xlim=[6e-2, 1e2], ylim=(5e-6, 2e1))
    # tools.ridge_plot(amp, 'grid', 'squeeze', 'dots', xlabel='$ A_k $', ylabel='$ P_{A_k} $',
    #                  labels=lab, figname=figs[2], plt_type='loglog', y_scale=1.13, xlim=[6e-2, 1e2], ylim=(5e-6, 2e1))
    # for yy, txt, txt2 in zip(amptxt_y, amp_std, amp_fit):
    #     plt.text(twtxt_x, yy, f'{txt2}', horizontalalignment='left',
    #             verticalalignment='bottom', size=5)
    #     plt.text(twtxt_x, yy, f'$ \mathrm{{std}} = {txt:2.2f}$', horizontalalignment='left',
    #             verticalalignment='top', size=5)
    tools.ridge_plot(
        corr,
        "grid",
        "slalomaxis",
        xlabel="Lag",
        ylabel="Correlation",
        labels=lab,
        figname=figs[3],
        plt_type="plot",
        xlim=[-0.0001, 0.001],
    )  # , ylim=[1e-3, 2e0])

    if save:
        for f in [figs[3]]:
            print(f"Saving to {save_path}sde_anlz_{f}.*")
            plt.figure(f)
            plt.tight_layout()
            # plt.savefig(f'{save_path}sde_anlz_{f}.pdf',
            #             bbox_inches='tight', format='pdf', dpi=200)
            plt.savefig(f"{save_path}sde_anlz_{f}.pgf", bbox_inches="tight")
    else:
        plt.show()


# Figure 8
def fpp_tw_real(data=True, save=False):
    file = f"{data_path}fpp_tw_real.npz"
    # file = f'{data_path}fpp_tw_real_ou.npz'
    rate = ["ou", "ou", "ou"]
    figs = ["cox", "tick", "var_rate"]
    gamma = [0.01, 0.1, 1.0]
    if not data:
        p = slf.FPPProcess()
        dt = 1e-2
        N = int(1e6)
        snr = 0.0
        fpps = [[], [], []]
        for i, (r, tw) in enumerate(zip(rate, figs)):
            for g in gamma:
                p.set_params(
                    gamma=g, K=int(N * g * dt), dt=dt, tw=tw, snr=snr, rate=r, amp="exp"
                )
                out = p.create_realisation(fit=False)
                s1, s2 = out[0], out[-1]
                s = (s1, s2)
                fpps[i].append(s)
                print(p.K)

        fpp_c = fpps[0]
        fpp_t = fpps[1]
        fpp_v = fpps[2]
        del fpps
        np.savez(file, fpp_c=fpp_c, fpp_t=fpp_t, fpp_v=fpp_v)
    else:
        print(f"Loading data from {file}")
        with np.load(file, allow_pickle=True) as f:
            fpp_c = f["fpp_c"]
            fpp_t = f["fpp_t"]
            fpp_v = f["fpp_v"]

    plt.rcParams["lines.linewidth"] = 0.4
    lab = [fr"$\gamma = {g}$" for g in gamma]
    tools.ridge_plot(
        fpp_c, xlabel="$ t $", ylabel=r"$\Phi$", labels=lab, figname=figs[0]
    )
    tools.ridge_plot(
        fpp_t, xlabel="$ t $", ylabel=r"$\Phi$", labels=lab, figname=figs[1]
    )
    tools.ridge_plot(
        fpp_v, xlabel="$ t $", ylabel=r"$\Phi$", labels=lab, figname=figs[2]
    )

    if save:
        for f in figs:
            print(f"Saving to {save_path}fpp_tw_real_nrand_{f}.*")
            # print(f'Saving to {save_path}fpp_tw_real_ou_{f}.*')
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(
                f"{save_path}fpp_tw_real_nrand_{f}.pdf",
                bbox_inches="tight",
                format="pdf",
                dpi=200,
            )
            plt.savefig(
                f"{save_path}fpp_tw_real_nrand_{f}.pgf", bbox_inches="tight", dpi=200
            )
    else:
        plt.show()


# Figure 9
def fpp_tw_psd(save=False):
    file = f"{data_path}fpp_tw_real.npz"
    # file = f'{data_path}fpp_tw_real_ou.npz'
    figs = ["cox_psd", "tick_psd", "var_rate_psd"]
    gamma = [0.01, 0.1, 1.0]
    dt = 1e-2

    print(f"Loading data from {file}")
    with np.load(file, allow_pickle=True) as f:
        fpp_c = f["fpp_c"]
        fpp_t = f["fpp_t"]
        fpp_v = f["fpp_v"]

    lab = [fr"$\gamma = {g}$" for g in gamma]
    tools.ridge_plot_psd(
        fpp_c,
        dt,
        "squeeze",
        xlabel="$ f $",
        ylabel="$ S $",
        labels=lab,
        figname=figs[0],
    )
    tools.ridge_plot_psd(
        fpp_t,
        dt,
        "squeeze",
        xlabel="$ f $",
        ylabel="$ S $",
        labels=lab,
        figname=figs[1],
    )
    tools.ridge_plot_psd(
        fpp_v,
        dt,
        "squeeze",
        xlabel="$ f $",
        ylabel="$ S $",
        labels=lab,
        figname=figs[2],
    )

    if save:
        for f in figs:
            print(f"Saving to {save_path}fpp_tw_psd_nrand_{f}.*")
            # print(f'Saving to {save_path}fpp_tw_psd_ou_{f}.*')
            plt.figure(f)
            plt.tight_layout()
            plt.savefig(
                f"{save_path}fpp_tw_psd_nrand_{f}.pdf",
                bbox_inches="tight",
                format="pdf",
                dpi=200,
            )
            plt.savefig(
                f"{save_path}fpp_tw_psd_nrand_{f}.pgf", bbox_inches="tight", dpi=200
            )
    else:
        plt.show()


if __name__ == "__main__":
    fpp_example()  # 1
    amplitude_dist()  # 2
    fpp_sde_realisations()  # 3
    fpp_sde_real_L()  # 4
    fpp_sde_psdpdf()  # 5
    sde_tw()  # 6
    # Figure 7 is created in slf.py
    fpp_tw_real()  # 8
    fpp_tw_psd()  # 9
    # fpp_tw_dist(data=False)
    # fpp_tw_cox()
    # fpptw_sde_real()
    # fpptw_sde_psd()
    # power_law_pulse()
