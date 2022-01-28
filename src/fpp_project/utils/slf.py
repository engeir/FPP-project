"""Deprecated: use the slf.py script in the uit_scripts folder.
Wrapper script for the uit-scripts library with limited support.
Especially suited for creating time series realizations / processes
with implementation of clustering methods for waiting times.
"""

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import sdepy
import tick.base as tb
import tick.hawkes as th
from scipy.signal import fftconvolve
from sklearn.datasets import make_blobs
from uit_scripts.misc import runge_kutta_SDE as rksde  # pylint: disable=E0401
from uit_scripts.shotnoise import gen_shot_noise as gsn  # pylint: disable=E0401
from uit_scripts.stat_analysis import deconv_methods as dm  # pylint: disable=E0401

import fpp_project.utils.tools as tools

# sys.path.append('/home/een023/Documents/work/FPP_SOC_Chaos/uit_scripts')
# sys.path.append('/home/een023/resolve/uit_scripts')


class Realisation:
    """Create actual realisations of an FPP process."""

    def __init__(self, process="FPP"):
        if process == "SDE":
            self.p = SDEProcess()
        else:
            self.p = FPPProcess()
        self.arr: tuple

    def __update(self):
        self.arr = self.p.create_realisation()

    def set_params(self, **kwargs):
        self.p.set_params(**kwargs)
        self.__update()

    def warning_no_array(self):
        try:
            self.arr[0]
        except Exception:
            print("A realisation has not yet been made. Using default values.")
            self.__update()

    def get_array(self, the_array):
        if self.p.process == "SDE":
            arr_opt = {
                "t": 0,
                "ta": 1,
                "amp": 2,
                "pulse": 3,
                "forcing": 4,
                "error": 5,
                "response": 6,
            }
            # arr_opt = dict((x, i) for i, (x, _) in enumerate(list(arr_opt.items())[0::6]))
        else:
            arr_opt = {
                "t": 0,
                "forcing": 1,
                "pulse": 2,
                "pulse_fit": 3,
                "error": 4,
                "response_fit": 5,
                "response": 6,
            }
        self.warning_no_array()
        if the_array in list(arr_opt.keys()):
            return self.arr[arr_opt[the_array]]
        assert (
            False
        ), f'The argument "{the_array}" did not match any of the possible return values.'

    def plotter(self):
        self.warning_no_array()
        self.p.plotter(*self.arr)

    def plot_psd(self):
        self.warning_no_array()
        signal = self.get_array("response")
        self.p.plot_psd(parameter=signal)


class Process(ABC):
    """Base class for a Process object.
    Arguments:
        ABC {class} -- abstract base class that all Process objects inherit from
    """

    process: str

    def __init__(self):
        self.gamma = 1.0
        self.K = 1000
        self.dt = 0.01
        self.__update_params()

    # @abstractproperty
    # def process(self) -> str:
    #     """The type of the intregrand implementation."""

    def set_params(self, **kwargs):
        """Handles how the parameters are set."""
        for key in kwargs:
            if key in list(self.__dict__.keys()):
                if isinstance(kwargs[key], type(getattr(self, key))):
                    setattr(self, key, kwargs[key])
                else:
                    raise TypeError(f'"{key}" must be a {type(getattr(self, key))}')
            else:
                print(f'"{key}" is not an attribute of {type(self)}')
        self.__update_params()

    def get_params(self, full=False):
        """Print all self keys.

        Default behaviour is to print only the keys. Option to also print values.

        Args:
            full (bool, optional): Print both keys and values of `self`. Defaults to False.
        """
        if full:
            print(self.__dict__)
        else:
            print(self.__dict__.keys())

    def __update_params(self):
        self.T = self.K / self.gamma
        # self.K = int(self.gamma * self.T)

    @abstractmethod
    def create_realisation(self, fit=True) -> np.ndarray:
        """Method that creates a realisation of the process and returns the resulting arrays.

        Returns:
            np.ndarray: see specific classes for number of outputs
        """

    # , all_plots=False, real=False, psd=False):
    def plot_realisation(self, *plot, parameter=None, **kwargs):
        """Plot arrays created by a realisation of the process."""
        for key in plot:
            try:
                plt_func = getattr(self, key)
            except AttributeError:
                print(f'No plots made. "{key}" is not an attribute of Process.')
            else:
                plt_func(parameter=parameter, **kwargs)
        # if all_plots:
        #     arr = self.create_realisation()
        #     self.plotter(*arr)
        #     self.plot_psd(arr[-1])
        # elif real:
        #     arr = self.create_realisation()
        #     self.plotter(*arr)
        # elif psd:
        #     arr = self.create_realisation(fit=False)
        #     self.plot_psd(arr[-1])
        # else:
        #     print('No plots made.')

    @staticmethod
    @abstractmethod
    def plotter(*args, new_fig=True):
        """Plotter method."""

    def plot_all(self, parameter=None, **kwargs):
        if parameter is None:
            parameter = self.create_realisation()
            self.plotter(*parameter)
            self.plot_psd(parameter[-1])
        else:
            self.plotter(*parameter)
            self.plot_psd(parameter[-1])

    def plot_real(self, parameter=None, **kwargs):
        if parameter is None:
            parameter = self.create_realisation(**kwargs)
            self.plotter(*parameter)
        else:
            self.plotter(*parameter)

    def plot_psd(self, parameter=None, **kwargs):
        # TODO: probably need to send self.dt into psd calculation (compare with jupyter nb)
        if parameter is None:
            parameter = self.create_realisation(fit=False)
            tools.psd(parameter[-1], **kwargs)
        else:
            if isinstance(parameter, np.ndarray):
                tools.psd(parameter, **kwargs)
            elif isinstance(parameter, tuple):
                tools.psd(parameter[-1], **kwargs)

    def plot_pdf(self, parameter=None, **kwargs):
        if parameter is None:
            parameter = self.create_realisation(fit=False)
            tools.pdf(parameter[-1], **kwargs)
        else:
            if isinstance(parameter, np.ndarray):
                tools.pdf(parameter, **kwargs)
            elif isinstance(parameter, tuple):
                tools.pdf(parameter[-1], **kwargs)


class SDEProcess(Process):
    """A SDE process."""

    process = "SDE"

    # def get_params(self):
    #     """
    #     docstring
    #     """
    #     print(self.__dict__)

    @staticmethod
    def fit_force(t, response):
        iterations = 200
        kern = gsn.kern(t - t.mean())
        res, err = dm.RL_gauss_deconvolve(response, kern, iterations)
        res = res[:, 0]

        ta_est, amp_est = dm.find_amp_ta_savgol(res, t)
        return amp_est, ta_est, kern, res, err

    def create_realisation(self, fit=True):
        # @sdepy.integrate
        # def rate_process(t, x, gamma=1.):
        #     return {'dt': x * (1 - x / (1 + gamma)), 'dw': 2**(1/2) * x * (1+gamma)**(-1/2)}

        # timeline = np.linspace(0., 10., self.K)
        # t = np.linspace(0., self.T, self.K)
        # x = rate_process(x0=1e-9, gamma=self.gamma)(timeline)  # pylint: disable=E1102,E1123,E1120
        # x = x.reshape((-1,))

        x = rksde.SDE_SLE(
            self.dt, int(self.T / self.dt), x0=self.gamma, gamma=self.gamma, log=True
        )
        # x = (x - x.mean()) / x.std()
        t = np.linspace(0, self.T, int(self.T / self.dt))
        if fit:
            amp, ta, pulse, forcing, error = self.fit_force(t, x)
            return t, ta, amp, pulse, forcing, error, x

        return t, x

    def get_tw(self, parameter=None):
        if parameter is None:
            parameter = self.create_realisation(fit=False)
        else:
            if isinstance(parameter, np.ndarray) and not len(parameter) == 2:
                raise TypeError(
                    '"plot_tw" only works if both the time and forcing arrays are sent in'
                )
        t = parameter[0]
        s = parameter[1]
        pulse_shape = gsn.kern(t - t.mean())
        deconv, _ = dm.RL_gauss_deconvolve(s, pulse_shape, 100)
        f = deconv.reshape((-1,))
        ta, amp = dm.find_amp_ta_savgol(f, t, window_length=11)
        # ta = t[f > 0]
        # ta = ta[amp > 1e-10]
        # tw = np.diff(ta)
        # tw = np.sort(trw)[::-1]
        return t, s, ta, amp, f

    @staticmethod
    def plotter(*args, new_fig=True):
        assert len(args) == 2 or len(args) == 7, f'length of args is "{len(args)}"'
        if new_fig == True:
            plt.figure(figsize=(9, 6))
        if len(args) == 2:
            t, x = args
            plt.xlabel("$t$")
            plt.plot(t, x, "k", label="Response")
            plt.legend()
        else:
            t, ta, amp, pulse, forcing, error, x = args
            plt.subplot(4, 1, 1)
            plt.xlabel("$t$")
            plt.plot(t, forcing, "-ok", label="Forcing (Estimated)")
            plt.scatter(ta, amp, c="r", label="Amplitudes (Estimated)")
            plt.legend()
            plt.subplot(4, 1, 2)
            plt.xlabel("$t$")
            plt.plot(t, pulse, "k", label="Pulse (Synthetic)", linewidth=2)
            plt.legend()
            plt.subplot(4, 1, 3)
            plt.xlabel("$t$")
            plt.plot(t, x, "k", label="Response (SLF)", linewidth=2)
            plt.legend()
            plt.subplot(4, 1, 4)
            plt.xlabel("$t$")
            plt.semilogy(error, "b", label="Error")
            plt.legend()


class FPPProcess(Process):
    """An FPP process."""

    process = "FPP"

    def __init__(self):
        super(FPPProcess, self).__init__()
        self.snr = 0.01
        self.kern_dict = {
            "1-exp": 0,
            "2-exp": 1,
            "lorentz": 2,
            "gauss": 3,
            "sech": 4,
            "power": 5  # ,
            #   '1exp2s': 6
        }
        self.kern = "1-exp"
        self.rate = "n-random"
        self.tw = "exp"
        self.TWkappa = 0.5
        self.amp = "exp"
        self.mA = 1.0

    def create_rate(self, version, k_length=True, tw=False):
        # Use an Ornstein-Uhlenbeck process as the rate
        if version == "ou":
            # From Matthieu Garcin. Hurst exponents and delampertized fractional Brownian motions. 2018. hal-01919754
            # https://hal.archives-ouvertes.fr/hal-01919754/document
            # and https://sdepy.readthedocs.io/en/v1.1.1/intro.html#id2
            # This uses a Wiener process as dW(t)
            @sdepy.integrate
            def rate_process(t, x, mu=1.0, k=0.1, sigma=1.0):
                return {"dt": k * (mu - x), "dw": sigma}

            # k: speed of reversion = .1
            # mu: long-term average position = 1.
            # sigma: volatility parameter = 1.
            size = self.K if k_length else int(1e5)
            # timeline = np.linspace(0.0, 1.0, size)
            t = np.linspace(0.0, self.T, size)
            # fmt: off
            rate = rate_process(x0=1.0, k=0.01, sigma=0.01)(t)  # pylint: disable=E1102,E1123,E1120
            # rate = rate_process(x0=1., k=.01, sigma=.01)(timeline)  # pylint: disable=E1102,E1123,E1120
            # fmt: on
            # rate = rate.reshape((-1,))
            # rate *= self.gamma / rate.mean()
            rate = rate.reshape((-1,))
            rate -= rate.min()
            rate = rate * self.gamma * 0.995 / rate.mean() + 0.005
            rate = rate if not tw else 1 / rate
        # Use n random numbers as the rate, drawn from a gamma distribution
        elif version == "n-random":
            prob = np.random.default_rng()
            size = self.K if k_length else int(1e5)
            # Return rate as a waiting time if True, else as a varying gamma.
            scale = 1 / self.gamma if tw else self.gamma
            # rate = prob.gamma(shape=1., scale=scale, size=size)
            rate = prob.exponential(scale=scale, size=size)
            t = np.linspace(0, np.sum(1 / rate), size)
        else:
            raise KeyError(f"'version' must be 'ou' or 'n-random', not {version}")
        if any(rate < 0):
            print("Warning: Rate process includes negatives. Computing abs(rate).")
            rate = abs(rate)
        return rate, t

    def create_ampta(self, version, Vrate):
        assert version in ["var_rate", "cox", "tick"]
        if version == "var_rate":
            rate, t = self.create_rate(Vrate, k_length=False)
            t = np.linspace(0, np.sum(rate), int(1e5))
            int_thresholds = np.linspace(0, np.sum(rate), self.K)
            c_sum = np.cumsum(rate)
            ta_i = tools.find_nearest(c_sum, int_thresholds)
            ta = t[ta_i]
            # Normalize arrival times to within T_max
            ta = ta * self.T / np.ceil(np.max(ta))
            amp, _, _ = gsn.amp_ta(self.gamma, self.K)
            return amp, ta
        if version == "cox":
            k_length = True
            rate, _ = self.create_rate(Vrate, k_length=k_length, tw=True)
            if not k_length:
                idx = np.round(np.linspace(0, len(rate) - 1, self.K)).astype(int)
                rate = rate[idx]
            amp, ta, self.T = gsn.amp_ta(
                1 / rate, self.K, TWdist="gam", Adist=self.amp, TWkappa=0.1
            )
            return amp, ta
        if version == "tick":
            rate, t = self.create_rate(Vrate, k_length=False)
            tf = tb.TimeFunction((t, rate))  # , dt=self.dt)
            ipp = th.SimuInhomogeneousPoisson([tf], end_time=self.T, verbose=False)
            ipp.track_intensity()  # Accepts float > 0 to set time step of reproduced rate process
            ipp.threshold_negative_intensity()
            ipp.simulate()
            ta = ipp.timestamps[0]
            # # PLOT RATE AND ESTIMATED ARRIVAL TIMES
            # y = ipp.tracked_intensity[0]
            # xx = ipp.intensity_tracked_times
            # idx = tools.find_nearest(xx, ta)
            # sc_x = xx[idx]
            # print(len(sc_x))
            # sc_y = y[idx]
            # lines = []
            # for i in range(len(sc_x)):
            #     pair = [(sc_x[i], 0), (sc_x[i], sc_y[i])]
            #     lines.append(pair)
            # linecoll = matcoll.LineCollection(lines, colors='r', linewidths=.5)
            # plt.figure(figsize=(7, 1))
            # ax = plt.subplot(1, 2, 1)
            # plt.plot(t, rate, 'k', zorder=-1)
            # ax.add_collection(linecoll)
            # plt.scatter(sc_x, sc_y, color='r', zorder=1)
            # plt.text(0, .2, f'Tick — $ K={len(sc_x)} $', size=5)
            # # For var_rate as well
            # t = np.linspace(0, np.sum(rate), int(1e5))
            # int_thresholds = np.linspace(0, np.sum(rate), self.K)

            # c_sum = np.cumsum(rate)
            # ta_i = tools.find_nearest(c_sum, int_thresholds)
            # ta = t[ta_i]
            # # Normalize arrival times to within T_max
            # tt = t * self.T / np.ceil(np.max(ta))  # Needed only for plotting
            # ta = ta * self.T / np.ceil(np.max(ta))
            # # PLOT RATE AND ESTIMATED ARRIVAL TIMES
            # sc_x = ta
            # print(len(sc_x))
            # sc_y = rate[ta_i]
            # lines = []
            # for i in range(len(sc_x)):
            #     pair = [(sc_x[i], 0), (sc_x[i], sc_y[i])]
            #     lines.append(pair)
            # linecoll = matcoll.LineCollection(lines, colors='r', linewidths=.5)
            # ax = plt.subplot(1, 2, 2)
            # plt.plot(tt, rate, 'k', zorder=-1)
            # ax.add_collection(linecoll)
            # plt.scatter(sc_x, sc_y, color='r')
            # plt.text(0, .2, f'Int — $ K={len(sc_x)} $', size=5)
            # save_path = '/home/een023/Documents/FPP_SOC_Chaos/report/figures/'
            # plt.savefig(f'{save_path}rate_sampling.pdf',
            #             bbox_inches='tight', format='pdf', dpi=200)
            # plt.show()
            # sys.exit()
            # # PLOT RATE AND ESTIMATED ARRIVAL TIMES
            self.K = len(ta)
            amp, _, _ = gsn.amp_ta(self.gamma, self.K)
            return amp, ta

    def create_forcing(self):
        """
        docstring
        """
        print("Warning: This method is outdated. Use `create_ampta`.")
        kinds = ["cluster", "var_rate", "cox", "cox_var_rate", "tick"]
        assert self.tw in kinds, f'The "kind" must be on of {kinds}, not {self.tw}'
        if self.tw == "cluster":
            out = make_blobs(
                n_samples=self.K,
                centers=100,
                cluster_std=0.1,
                n_features=1,
                random_state=0,
            )
            x = out[0].reshape((-1,))
            # TW = np.insert(x, 0, 0.)
            # ta = np.cumsum(TW[:-1])
            # self.T = ta[-1] + TW[-1]
            ta = np.sort(x) - np.min(x)
            ta = ta / np.max(ta) * self.T
            amp, _, _ = gsn.amp_ta(self.gamma, self.K)
        elif self.tw == "var_rate":
            # From Matthieu Garcin. Hurst exponents and delampertized fractional Brownian motions. 2018. hal-01919754
            # https://hal.archives-ouvertes.fr/hal-01919754/document
            # and https://sdepy.readthedocs.io/en/v1.1.1/intro.html#id2
            # This uses a Wiener process as dW(t)
            @sdepy.integrate
            def rate_process(t, x, mu=1.0, k=0.1, sigma=1.0):
                return {"dt": k * (mu - x), "dw": sigma}

            # k: speed of reversion = .1
            # mu: long-term average position = 1.
            # sigma: volatility parameter = 1.

            timeline = np.linspace(0.0, 1.0, int(1e5))
            t = np.linspace(0.0, self.T, int(1e5))
            x = rate_process(x0=1.0)(timeline)  # pylint: disable=E1102,E1123,E1120
            x = self.gamma * (x - np.min(x)) / (np.max(x) - np.min(x)) + self.gamma
            x -= np.mean(x)
            x = abs(x)
            int_thresholds = np.linspace(0, np.sum(x), self.K)
            c_sum = np.cumsum(x)
            ta = tools.find_nearest(c_sum, int_thresholds)
            ta = t[ta]
            amp, _, _ = gsn.amp_ta(self.gamma, self.K)
        elif self.tw == "cox":
            prob = np.random.default_rng()
            tw = prob.gamma(shape=1.0, scale=1 / self.gamma, size=self.K)
            # tw = prob.exponential(scale=1 / self.gamma, size=self.K)
            # tw = prob.poisson(lam=1 / self.gamma, size=self.K)
            amp, ta, self.T = gsn.amp_ta(
                1 / tw, self.K, TWdist="exp", Adist=self.amp, TWkappa=0.1
            )
        elif self.tw == "cox_var_rate":
            # From Matthieu Garcin. Hurst exponents and delampertized fractional Brownian motions. 2018. hal-01919754
            # https://hal.archives-ouvertes.fr/hal-01919754/document
            # and https://sdepy.readthedocs.io/en/v1.1.1/intro.html#id2
            # This uses a Wiener process as dW(t)
            @sdepy.integrate
            def rate_process(t, x, mu=2.0, k=0.1, sigma=1.0):
                return {"dt": k * (mu - x), "dw": sigma}

            # k: speed of reversion = .1
            # mu: long-term average position = 1.
            # sigma: volatility parameter = 1.

            timeline = np.linspace(0.0, 1.0, self.K)
            t = np.linspace(0.0, self.T, self.K)
            x = rate_process(x0=2.0)(timeline)  # pylint: disable=E1102,E1123,E1120
            g = x / 2 * self.gamma
            amp, ta, self.T = gsn.amp_ta(
                g, self.K, TWdist="exp", Adist=self.amp, TWkappa=0.1
            )
        elif self.tw == "tick":
            # https://arxiv.org/pdf/1707.03003.pdf
            @sdepy.integrate
            def rate_process(t, x, mu=1.0, k=0.1, sigma=1.0):
                return {"dt": k * (mu - x), "dw": sigma}

            # k: speed of reversion = .1
            # mu: long-term average position = 1.
            # sigma: volatility parameter = 1.

            timeline = np.linspace(0.0, 1.0, self.K)
            t = np.linspace(0.0, self.T, self.K)
            x = rate_process(x0=1.0)(timeline)  # pylint: disable=E1102,E1123,E1120
            x = x.reshape((-1,))
            x *= self.gamma / x.mean()
            # === PLOT RATE ===
            # plt.figure()
            # plt.plot(t, x)
            # === CREATE Inhomogeneous Poisson process ===
            tf = tb.TimeFunction((t, x), dt=self.dt)
            ipp = th.SimuInhomogeneousPoisson([tf], end_time=self.T, verbose=False)
            ipp.track_intensity()  # Accepts float > 0 to set time step of reproduced rate process
            ipp.threshold_negative_intensity()
            ipp.simulate()
            ta = ipp.timestamps[0]
            self.K = len(ta)
            # === PLOT FULL INTENSITY PROCESS (need float in track_intensity) ===
            # xx, yy = ipp.intensity_tracked_times, ipp.tracked_intensity
            # yy = yy[0]
            # y = yy[tools.find_nearest(xx, ta)]
            # === PLOT RETRIEVED DATA ===
            # plt.figure()
            # # plt.plot(xx, yy)
            # plt.scatter(ta, np.arange(len(ta)))
            amp, _, _ = gsn.amp_ta(self.gamma, self.K)
            # print(len(amp), len(ta), ta[-1], self.T, self.K)
            # tp.plot_point_process(ipp)
            # sys.exit()
        else:
            amp, ta, self.T = gsn.amp_ta(
                self.gamma, self.K, TWdist=self.tw, Adist=self.amp
            )

        return amp, ta

    @staticmethod
    def fit_pulse(t, forcing, response):
        # Shift removed. Not needed?
        pulse, error = dm.RL_gauss_deconvolve(response, forcing, 600)
        # pulse, error = dm.RL_gauss_deconvolve(response, forcing, 600, shift=None)
        pulse = pulse.reshape((-1,))
        pulse_fit = tools.optimize(t, np.copy(pulse), pen=False)
        response_fit = fftconvolve(forcing, pulse_fit, mode="same")
        return pulse, pulse_fit, response_fit, error

    def create_realisation(self, fit=True, ampta=False, full=False):
        # Beware that seeds may yield strange results!
        # Equal seeds give correlated amplitude and waiting times.
        # prev. good seeds: (20, 31)
        # try:
        t, _, response, amp, ta = gsn.make_signal(
            self.gamma,
            self.K,
            self.dt,
            mA=self.mA,
            eps=self.snr,
            ampta=True,
            dynamic=True,
            kerntype=self.kern,
            lam=0.5,
            rate=(self.rate, self.tw),
            TWdist="gam",
            Adist=self.amp,
            TWkappa=self.TWkappa,
        )
        # except Exception:
        #     # amp, ta = self.create_forcing()
        #     amp, ta = self.create_ampta(self.tw, self.rate)
        #     t, response = gsn.signal_convolve(
        #         amp, ta, self.T, self.dt, kernsize=2**17, kerntype=self.kern)

        ta_index = np.ceil(ta / self.dt).astype(int)
        forcing = np.zeros(t.size)
        # for i in range(ta_index.size):
        #     forcing[ta_index[i]] += amp[i]
        forcing[ta_index] = amp

        if fit:
            fitted = self.fit_pulse(t, forcing, response)
            return t, forcing, fitted[0], fitted[1], fitted[2], fitted[3], response
        if full:
            return t, forcing, response, amp, ta
        if ampta:
            return amp, ta
        return t, forcing, response

    def get_tw(self, parameter=None):
        if parameter is None:
            out = self.create_realisation(fit=False, ampta=True)
            ta = out[1]
        else:
            t = parameter[0]
            f = parameter[1]
            ta = t[f > 0]
            if isinstance(parameter, np.ndarray) and not len(parameter) == 2:
                raise TypeError(
                    '"plot_tw" only works if both the time and forcing arrays are sent in'
                )
        tw = np.diff(ta)
        # tw = np.sort(trw)[::-1]
        return tw, np.arange(len(tw))

    def plot_tw(self, parameter=None, new_fig=True):
        if parameter is None:
            parameter = self.create_realisation(fit=False)
        else:
            if isinstance(parameter, np.ndarray):
                raise TypeError(
                    '"plot_tw" only works if both the time and forcing arrays are sent in'
                )
        t = parameter[0]
        f = parameter[1]
        ta = t[f > 0]
        trw = np.diff(ta)
        tw = np.sort(trw)[::-1]
        if new_fig:
            plt.figure()
        plt.semilogx(tw, label="waiting times, high to low")

    @staticmethod
    def plotter(*args, new_fig=True):
        assert len(args) == 7 or len(args) == 3, f'length of args is "{len(args)}"'
        if new_fig == True:
            plt.figure(figsize=(6, 4))
        if len(args) == 7:
            t, forcing, pulse, pulse_fit, response_fit, error, response = args
            plt.subplot(4, 1, 1)
            plt.xlabel("$t$")
            plt.plot(t, forcing, "-ok", label="Forcing")
            plt.legend()
            plt.subplot(4, 1, 2)
            plt.xlabel("$t$")
            plt.plot(t, pulse, "k", label="Pulse", linewidth=2)
            plt.plot(t, pulse_fit, "r", label="Pulse fit")
            plt.legend()
            plt.subplot(4, 1, 3)
            plt.xlabel("$t$")
            plt.plot(t, response, "k", label="Response", linewidth=2)
            plt.plot(t, response_fit, "r", label="Response fit")
            plt.legend()
            plt.subplot(4, 1, 4)
            plt.xlabel("$t$")
            plt.semilogy(error, "b", label="Error")
            plt.legend()
        else:
            t, forcing, response = args
            plt.subplot(2, 1, 1)
            plt.xlabel("$t$")
            plt.plot(t, forcing, "-ok", label="Forcing")
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.xlabel("$t$")
            plt.plot(t, response, "k", label="Response")
            plt.legend()
        plt.xlabel("$t$")


if __name__ == "__main__":
    ## === Examples ===
    p = FPPProcess()
    kern = ["1-exp"]  # , '1exp']  # 'power', '1exp', '1exp'
    tw = ["tick"]  # , 'var_rate']  # 'var_rate', 'ray', 'cluster'
    for k, t in zip(kern, tw):
        print(k, t)
        p.set_params(gamma=1.0, K=1000, kern=k, snr=0.0, tw=t, dt=0.01)
        p.plot_realisation("plot_psd")  # =True)
    p.set_params(gamma=1.0, kern="1-exp", snr=0.0, tw="tick")
    p.create_ampta()
    p.plot_realisation("plot_all")
    p.plot_psd()

    # # r_n = Realisation()
    # r_c = Realisation()
    # r_fr = Realisation()
    # r_sde = Realisation(process='SDE')
    # # r_n.set_params(gamma=1., T=10000., kern='1exp', snr=.0, tw='ray')
    # r_c.set_params(gamma=.1, kern='1exp', snr=.0, tw='cluster')
    # r_fr.set_params(gamma=.1, kern='1exp', snr=.0, tw='var_rate')
    # r_sde.set_params(gamma=.1, K=1000, dt=.1)
    # # r_n.plotter()
    # r_c.plotter()
    # r_fr.plotter()
    # r_sde.plotter()
    # # r_n.plot_psd()
    # r_c.plot_psd()
    # r_fr.plot_psd()
    # r_sde.plot_psd()
    # tools.timestamp()
    # plt.show()
    N = int(1e5)
    dt = 1e-2
    gamma = 1.0
    p = FPPProcess()
    p.set_params(tw="tick", gamma=gamma, K=int(N * gamma * dt), dt=dt)
    p.create_realisation()
