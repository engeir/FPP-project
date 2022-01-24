#!/home/een023/.virtualenvs/uit_scripts/bin/python
"""This script gives examples on how to deconvolve using synthetic forcing and forcing data.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
from scipy.signal import fftconvolve
import scipy.optimize as scop
import scipy.integrate as si

from uit_scripts.stat_analysis import deconv_methods as dpy
from uit_scripts.shotnoise import gen_shot_noise as gsn
import data.data_manager as dm


plt.rcParams['axes.grid'] = True
# Customize matplotlib
matplotlib.rcParams.update({
    'text.usetex': True,
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'pgf.texsystem': 'pdflatex'
})


def find_nearest(array, value):
    idx = []
    array = np.asarray(array)
    for i in value:
        idx.append((np.abs(array - i)).argmin())
    return idx


def synthetic_forcing():
    """Try to estimate the amplitude and arrival times based on an assumed known pulse shape.
    """
    t = np.linspace(0, 10, 1000)

    gamma, K = .1, 10
    amp, ta, T = gsn.amp_ta(gamma, K)
    dt = 0.01
    td = np.ones_like(amp)
    time, shot_n = gsn.signal_superposition(amp, ta, T, dt, td)

    # Create volcanoes
    f = np.zeros(1000)
    f[[40, 100, 350, 500, 750]] = 15

    # Climate sensitivity / response function
    # pt_s = np.exp(- np.linspace(0, 200, 500)) / 1
    # pt_s = np.r_[np.zeros(500), pt_s]
    pt_s = np.exp(- np.linspace(0, 50, int(time.size / 2))) / 1
    if time.size % 2:
        padding = int(time.size / 2) + 1
    else:
        padding = int(time.size / 2)
    pt_s = np.r_[np.zeros(padding), pt_s]

    np.random.seed(32)
    # r = fftconvolve(f, pt_s, 'same') + np.random.randn(1000) * 5e0 + 5
    t = time
    max_amp = np.max(amp)
    # TODO: use instead the implemented gen_noise in uit_scripts...
    r = shot_n + np.random.randn(shot_n.size) * max_amp * .06 + max_amp * .06
    f = shot_n

    plt.figure(figsize=(9, 6))
    plt.subplot(4, 1, 1)
    plt.plot(t, pt_s, 'r', label='Spread function')
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(t, f, 'b--', label='Forcing')
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(t, r, 'k', label='Response')
    plt.legend()
    _ = deconv(r, f, pt_s)
    # res = res.reshape((-1,))
    # T_a, A = dpy.find_amp_ta_old(res, np.linspace(0, 10, res.shape[0]))
    # plt.figure(figsize=(9, 6))
    # plt.plot(T_a, A)


def synthetic_spread():
    """Estimate the pulse shape based on known amplitudes and arrival times.
    """
    # Create a signal (inverted temperature response) from an amplitude and arrival time array (volcanoes)
    # Average waiting time is 1 / gamma
    gamma = .1
    K = int(gamma * 100)
    dt = 0.01
    snr = .01

    # Climate sensitivity / response function / pulse shape
    # pt_s = np.exp(- np.linspace(0, 200, 500)) / 1
    # pt_s = np.r_[np.zeros(500), pt_s]

    # np.random.seed(3)
    # max_amp = np.max(amp)
    # r_height = .03
    # r = shot_n + noise[1]  # np.random.randn(shot_n.size) * max_amp * r_height + max_amp * r_height
    t, _, r, amp, ta = gsn.make_signal(
        gamma, K, dt, eps=snr, ampta=True, dynamic=True, kerntype='1-exp', lam=.5)
    # Volcanic eruptions / delta pulses
    f = np.zeros_like(t)
    mask = find_nearest(t, ta)
    f[mask] = amp

    # Look at only second half of the signal
    # mask = int(len(t) / 2)
    # t = t[mask:]
    # f = f[mask:]
    # r = r[mask:]
    # ===

    plt.figure(figsize=(9, 6))
    # gs = gridspec.GridSpec(3, 2)
    plt.subplot(3, 1, 1)
    # plt.subplot(gs[0, :])
    plt.plot(t, f, 'r', label='Forcing')
    plt.legend()
    # plt.subplot(gs[1, :1])
    plt.subplot(3, 1, 2)
    # plt.plot(t, pt_s, 'r', label='Spread function')
    response, error = deconv(r, f, None, pts=False, time=t)
    plt.plot(t, response)
    res_fit = response_fit(response, time=t)
    try:
        plt.plot(t, res_fit, 'r--', label='Response fit')
        plt.legend()
    except Exception:
        pass
    # plt.subplot(gs[1, 1:])
    # _ = deconv(r, f, None, pts=False, shift=True)
    # plt.legend()
    # plt.subplot(gs[2, :])
    plt.subplot(3, 1, 3)
    plt.plot(t, r, 'k', label='Response')
    Temp = fftconvolve(f, res_fit, mode='same')
    plt.plot(t, Temp, 'r--', label='New temp', linewidth=2)
    plt.legend()
    # plt.savefig('deconv.pgf', bbox_inches='tight')
    # res = res.reshape((-1,))
    # T_a, A = dpy.find_amp_ta_old(res, np.linspace(0, 10, res.shape[0]))
    # plt.figure(figsize=(9, 6))
    # plt.plot(T_a, A)
    plt.figure()
    plt.semilogy(error)


def deconv(r, f, p_s, iterations=100, pts=True, time=None, shift=False):
    if time is not None:
        t = time
    else:
        t = np.linspace(0, 10, r.shape[0])
    if pts:
        res, err = dpy.RL_gauss_deconvolve(r, p_s, iterations, shift=shift)
        plt.subplot(4, 1, 3)
        plt.plot(t, res, 'b--', label=f'Deconvolved, shift = {shift}')
        plt.legend()
    else:
        res, err = dpy.RL_gauss_deconvolve(r, f, iterations, shift=shift)
        # plt.plot(t, res, 'k', label=f'Deconvolved, shift = {shift}')

    return res, err


def find_forcing():
    # === LOOK AT DATA ===
    # ctrl.files = ['T_orig', 'C_orig', 'T_yav', 'C_yav', 'scriptname']
    ctrl = np.load('data/control_run.npz', mmap_mode='r')

    # sig_in.files = ['T_orig', 'I_orig', 'O_orig', 'T', 'I', 'O', 'scriptname']
    sig = np.load('data/temp_yav_O.npz', mmap_mode='r')

    # sig_out.files = ['T_orig', 'I_orig', 'O_orig', 'T', 'I', 'O', 'scriptname']
    sig_2 = np.load('data/temp_rep_I.npz', mmap_mode='r')

    c = ctrl['C_yav']
    t = sig['T']
    s_in = sig['I']
    s_out = sig['O']

    plt.figure(figsize=(9, 6))
    plt.subplot(4, 1, 1)
    middle = 509
    shift = t[middle] - .206
    pt_s = (t[middle:] - shift)**(- .679) * .043
    pt_s = np.r_[np.zeros(middle), pt_s]
    # pt_s = np.linspace(1, 1000, 500)**(- .68) * .09
    # pt_s = np.r_[np.zeros(500), pt_s]
    plt.plot(t, pt_s, 'r', label='Spread function / Climate sensitivity')
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(t, s_in, 'b--', label=r'sig\_in / Forcing')
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(t, s_out, 'k', label=r'sig\_out / Response')
    plt.legend()

    # === DECONVOLVE DATA ===
    _ = deconv(s_out, s_in, pt_s, time=t)
    # res = res.reshape((-1,))
    # T_a, A = dpy.find_amp_ta_old(res, np.linspace(0, 10, res.shape[0]))
    # plt.figure(figsize=(9, 6))
    # plt.plot(T_a, A)


def find_sensitivity():
    # === LOOK AT DATA ===
    # ctrl.files = ['T_orig', 'C_orig', 'T_yav', 'C_yav', 'scriptname']
    ctrl = np.load('data/control_run.npz', mmap_mode='r')
    # sig_in.files = ['T_orig', 'I_orig', 'O_orig', 'T', 'I', 'O', 'scriptname']
    sig = np.load('data/temp_yav_O.npz', mmap_mode='r')
    # This one is very noisy. Use the above instead.
    # sig_out.files = ['T_orig', 'I_orig', 'O_orig', 'T', 'I', 'O', 'scriptname']
    sig_2 = np.load('data/temp_rep_I.npz', mmap_mode='r')

    # c = ctrl['C_yav']
    t = sig['T']
    s_in = sig['I']
    s_out = sig['O']
    # === DECONVOLVE DATA ===
    # d_response, error = deconv(s_out, s_in, None, pts=False, time=t, iterations=[414, 1000])
    d_response, error = dpy.RL_gauss_deconvolve(s_out, s_in, [414, 995])

    for response in d_response.T:
        plt.figure(figsize=(9, 6))
        # gs = gridspec.GridSpec(3, 2)
        plt.subplot(3, 1, 1)
        # plt.subplot(gs[0, :])
        plt.plot(t, s_in, 'r', label='sig in / Forcing')
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(t, response, 'k', label='Deconvolved')
        # plt.subplot(gs[1, :1])
        # # Based on numbers from main.pdf
        # # middle = 509
        # middle = np.searchsorted(t, 1285.01, side="left")
        # shift = t[middle] - .206
        # # pt_s = (t[middle:] - shift)**(- .679) * .043
        # pt_s = (t[middle:] - shift)**(- .679) * .01
        # pt_s = np.r_[np.zeros(middle), pt_s]
        # plt.plot(t, pt_s, 'r', label='Spread function / Climate sensitivity')
        res_fit = response_fit(response, time=t)
        response = response.reshape((-1,))
        response[:int(len(response) / 2 - 3)] = 0
        try:
            # plt.plot(t, response, 'r--', label='Response fit')
            plt.plot(t, res_fit, 'r--', label='Response fit')
            plt.legend()
        except Exception:
            pass
        # res = res.reshape((-1,))
        plt.legend()
        # plt.subplot(gs[1, 1:])
        # _ = deconv(s_out, s_in, None, pts=False, time=t, shift=True)
        plt.legend()
        plt.subplot(3, 1, 3)
        # plt.subplot(gs[2, :])
        plt.plot(t, s_out, 'k', label='sig out / Response')
        Temp = fftconvolve(s_in, res_fit, mode='same')
        # Temp = fftconvolve(s_in, response, mode='same')
        plt.plot(t, Temp, 'r--', label='New temp', linewidth=2)
        plt.legend()

        # T_a, A = dpy.find_amp_ta_old(res, np.linspace(0, 10, res.shape[0]))
        # plt.figure(figsize=(9, 6))
        # plt.plot(T_a, A)
        # np.savez('response_yav_O', res=res_fit)
        # plt.figure()
        # plt.semilogy(error)

def one_s_two_exp(t, c, d, amp, sc):
    term1 = c * np.exp(-2 * np.abs(t) * sc / (1 - d))
    term2 = (1 - c) * np.exp(- 2 * np.abs(t) * sc / (1 + d))
    # amp = 1
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


def response_fit(response, time=None):
    if time is not None:
        try:
            response = response.reshape((-1,))
        except Exception:
            assert False, 'You cannot reshape this!'
        assert response.shape == time.shape, f'The response and time arrays are not of equal length or shape. {response.shape} != {time.shape}'
    half = int(len(response) / 2 - 10)
    response[:half] = 0
    p = int(np.argwhere(response == np.max(response)))
    zeros = np.zeros_like(response[:p])
    signal = response[p:]
    if time is not None:
        t = time[p:] - time[p]
    else:
        t = np.arange(0, len(signal))
    # size = 10000
    # t_ = np.linspace(0, np.max(t), size)
    # signal = np.interp(t_, t, signal)
    # t = t_
    # mask = int(len(signal) * .02)

    # Best fit according two a 1-sided double exponential
    sig_cov, _ = scop.curve_fit(one_s_two_exp, t, signal, p0=[1, 0.5, 1, 1], bounds=([0, 0, 0, 0], [1, 1, np.inf, np.inf]))
    # Best constrained fit according two a 1-sided double exponential and integral = 1
    popt2, _ = scop.curve_fit(FuncPen, t, signal, p0=[1, 0.5, 1, 1], bounds=([0, 0, 0, 0], [1, 1, np.inf, np.inf]))
    print(sig_cov)
    print(popt2)
    # sig_fit = one_s_two_exp(t, .9, .9)
    sig_fit = one_s_two_exp(t, *sig_cov)
    sig_fit_2 = one_s_two_exp(t, *popt2)
    # sig_fit = one_s_two_exp(t, 0.01, .999, .5)
    # plt.plot(t, signal)
    # plt.plot(t, sig_fit)

    # Run some checks...
    # sig_fit *= 1.5  # si.simps(sig_fit, t)
    print('Integral of pulse function = %.3e' % si.simps(sig_fit, t))
    print('Integral of pulse function = %.3e' % si.simps(sig_fit_2, t))
    return np.r_[zeros, sig_fit]

def co2x2():
    time = np.linspace(- 100, 2900, 3000)
    forcing = np.heaviside(time, 1) * 2

    sig = np.load('data/temp_yav_O.npz', mmap_mode='r')
    t = sig['T']
    s_in = sig['I']
    s_out = sig['O']


    plt.figure(figsize=(9, 6))
    # gs = gridspec.GridSpec(3, 2)
    plt.subplot(3, 1, 1)
    # plt.subplot(gs[0, :])
    plt.plot(time, forcing, 'r', label='sig in / Forcing')
    plt.legend()
    plt.subplot(3, 1, 2)
    response, _ = deconv(s_out, s_in, None, pts=False, time=t)
    res_fit = response_fit(response, time=t)
    try:
        plt.plot(t, res_fit, 'r--', label='Response fit')
        plt.legend()
    except Exception:
        pass
    plt.legend()
    plt.subplot(3, 1, 3)
    Temp = fftconvolve(forcing, response.reshape((-1,)), mode='same')
    plt.plot(time, Temp, 'k', label='New temp from devonv', linewidth=2)
    Temp = fftconvolve(forcing, res_fit, mode='same')
    plt.plot(time, Temp, 'r--', label='New temp from two exp fit', linewidth=2)
    plt.legend()


def plot_temp(version):
    a = dm.look_at_jones_mann()
    x = a[1][:, 0]
    y = a[1][:, 4]
    file = np.load('response_yav_O.npz', mmap_mode='r')
    res = file['res']
    res = res.reshape((-1,))
    if version == 'zeroed':
        res[:int(len(res) / 2) - 1] = 0
    elif version == '2exp_fit':
        pass
    temp = fftconvolve(y, res, mode='same')

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(x, y, 'r', label='Forcing')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(res, 'b--', label='Response function')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(x, temp, 'k', label='Temperature')
    plt.legend()
    # plt.savefig('reconstruct.pgf', bbox_inches='tight')

if __name__ == '__main__':
    # synthetic_forcing()
    # synthetic_spread()
    # find_forcing()
    # find_sensitivity()
    co2x2()
    # plot_temp('pure')
    # plot_temp('zeroed')
    # dm.look_at_txt(
    #     'data/glannual_anomaly_ts_Amon_NorESM1-M_abrupt4xCO2_r1i1p1_000101-015012.txt')
    a = dm.look_at_jones_mann()
    print(a[0])
    dm.plot_list_data('jones_mann')  # pages_ens
    plt.show()
