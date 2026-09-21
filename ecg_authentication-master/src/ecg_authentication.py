import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, lfilter_zi, filtfilt, cheb2ord, cheby2, iirnotch
import pan_tompkins
import pywt
from statistics import mean

from collections import defaultdict, Counter
import scipy

import filterpy
from filterpy.kalman import ExtendedKalmanFilter, UnscentedKalmanFilter

from scipy.signal import kaiserord, firwin, freqz

import wfdb
from wfdb import processing

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from dtw import dtw

from scipy.spatial import distance

from new__src.biosppy import storage, utils
from new__src.biosppy.signals import ecg, tools


import math


def manhattan_distance(x, y): return np.abs(x - y)


def ecg_log(level, print_string):
    print(print_string)


class KalmanFilter(object):

    def __init__(self, process_variance, estimated_measurement_variance):
        self.process_variance = process_variance
        self.estimated_measurement_variance = estimated_measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def input_latest_noisy_measurement(self, measurement):
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        blending_factor = priori_error_estimate / \
            (priori_error_estimate + self.estimated_measurement_variance)
        self.posteri_estimate = priori_estimate + \
            blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (
            1 - blending_factor) * priori_error_estimate

    def get_latest_estimated_measurement(self):
        return self.posteri_estimate


class ecg_authentication:
    fs = 250
    decompostion_level = 4
    template_method = 'template_avg'
    discrete_wavelet_type = 'db4'
    continuous_wavelet = 'gaus1'
    continuous_wt_scale = 50
    feature_method = 'fiducial'
    segmentation_method = "window"

    def __init__(
            self,
            template_method,
            discrete_wavelet_type,
            continuous_wavelet,
            continuous_wt_scale,
            feature_method,
            segmentation_method):
        self.template_method = template_method
        self.discrete_wavelet_type = discrete_wavelet_type
        self.continuous_wavelet = continuous_wavelet
        self.continuous_wt_scale = continuous_wt_scale
        self.feature_method = feature_method
        self.segmentation_method = segmentation_method
        # ecg_log(5, "Init ECG_Authentication")
        pass
    ##########################################################################
    #
    #                                   PRIVATE FUNCTIONS
    #
    ##########################################################################

    def __create_file_list(self, file_prefix, file_count):
        file_list = []
        for num in range(0, file_count):
            temp = file_prefix.format(num)
            file_list.append(temp)
        return file_list

    def __cheby_lowpass(self, wp, ws, fs, gpass, gstop):
        wp = wp / fs
        ws = ws / fs
        order, wn = cheb2ord(wp, ws, gpass, gstop)
        b, a = cheby2(order, gstop, wn)
        return b, a

    def __cheby_lowpass_filter(self, data, cutoff, fs, gpass, gstop):
        b, a = self.__cheby_lowpass(cutoff[0], cutoff[1], fs, gpass, gstop)
        y = lfilter(b, a, data)
        return y

    def __butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def __butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.__butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def __fir_filter(self, x, fs, stop_band=60, cutoff_freq_in_hz=10):
        nyq_rate = fs / 2.0

        # The desired width of the transition from pass to stop,
        # relative to the Nyquist rate.  We'll design the filter
        # with a 5 Hz transition width.
        width = 5.0 / nyq_rate

        # The desired attenuation in the stop band, in dB.
        ripple_db = stop_band

        # Compute the order and Kaiser parameter for the FIR filter.
        N, beta = kaiserord(ripple_db, width)

        # The cutoff frequency of the filter.
        cutoff_hz = cutoff_freq_in_hz

        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        taps = firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta))

        # Use lfilter to filter x with the FIR filter.
        filtered_x = lfilter(taps, 1.0, x)
        return filtered_x

    def __filfit_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.__butter_bandpass(lowcut, highcut, fs, order=order)

        # Apply the filter to xn.  Use lfilter_zi to choose the initial condition
        # of the filter.
        zi = lfilter_zi(b, a)
        z, _ = lfilter(b, a, data, zi=zi * data[0])

        # Apply the filter again, to have a result filtered at an order
        # the same as filtfilt.
        z2, _ = lfilter(b, a, z, zi=zi * z[0])

        # Use filtfilt to apply the filter.
        y = filtfilt(b, a, data)
        return y

    def __plot_filtered_signal_comp(self, signal, filtered_signal):
        fig, axs = plt.subplots(2)
        fig.suptitle("Heartbeat")
        axs[0].plot(signal)
        axs[1].plot(filtered_signal)
        plt.show()

    def __lfilter_zi(self, b, a):
        # compute the zi state from the filter parameters.
        # Based on:
        # Fredrik Gustafsson, Determining the initial states in forward-backward
        # filtering, IEEE Transactions on Signal Processing, pp. 988--992, April 1996,
        # Volume 44, Issue 4
        n = max(len(a), len(b))
        zin = (scipy.eye(n - 1) - scipy.hstack(
            (-a[1:n, scipy.newaxis], scipy.vstack((scipy.eye(n - 2), scipy.zeros(n - 2))))))
        zid = b[1:n] - a[1:n] * b[0]
        zi_matrix = scipy.linalg.inv(zin) * (scipy.matrix(zid).transpose())
        zi_return = []
        # convert the result into a regular array (not a matrix)
        for i in range(len(zi_matrix)):
            zi_return.append(float(zi_matrix[i][0]))

        return scipy.array(zi_return)

    def __filtfilt(self, b, a, x):
        """
        Filter with given parameters forward and in reverse to eliminate
        phase shifts.
        In addition, initial state is calculated with lfilter_zi and
        mirror images of the sample are added at end and beginning to
        remove edge effects.
        Must be a one-dimensional array only.
        """
        # For now only accepting 1d arrays
        ntaps = max(len(a), len(b))
        edge = ntaps * 3

        if x.ndim != 1:
            raise ValueError("Filtfilt is only accepting 1 dimension arrays.")

        # x must be bigger than edge
        if x.size < edge:
            raise ValueError(
                "Input vector needs to be bigger than 3 * max(len(a),len(b).")

        if len(a) < ntaps:
            a = scipy.r_[a, scipy.zeros(len(b) - len(a))]

        if len(b) < ntaps:
            b = scipy.r_[b, scipy.zeros(len(a) - len(b))]

        zi = self.__lfilter_zi(b, a)

        # Grow the signal to have edges for stabilizing
        # the filter with inverted replicas of the signal
        s = scipy.r_[2 * x[0] - x[edge:1:-1], x, 2 * x[-1] - x[-1:-edge:-1]]
        # in the case of one go we only need one of the extrems
        # both are needed for filtfilt

        (y, zf) = scipy.signal.lfilter(b, a, s, -1, zi * s[0])

        (y, zf) = scipy.signal.lfilter(b, a, scipy.flipud(y), -1, zi * y[-1])

        return scipy.flipud(y[edge - 1:-edge + 1])

    def __remove_baseline(self, signal):
        from scipy.signal import medfilt
        # Sampling frequency
        fs = self.fs
        # Baseline estimation
        win_size = int(np.round(0.2 * fs)) + 1
        baseline = medfilt(signal, win_size)
        win_size = int(np.round(0.6 * fs)) + 1
        baseline = medfilt(baseline, win_size)

        # Removing baseline
        filt_data = signal - baseline
        return filt_data, baseline

    def __ekf_filter(self, signal):
        # # relatively basic implementation for now
        Nyq = self.fs / 2
        wn = [5 / Nyq, 15 / Nyq]
        b, a = scipy.signal.butter(2, wn, btype='bandpass')
        # TODO: filtfilt should be implemented here
        filtered_signal = self.__filtfilt(b, a, signal)
        return filtered_signal

    def __wavelet_filter(self, signal):
        coeffs = pywt.wavedec(signal, 'coif3', level=6)
        new_coeff = []
        for each_coeff in coeffs:
            new_coeff.append(pywt.threshold(each_coeff, 1, 'soft'))
        filtered_signal = pywt.waverec(new_coeff, 'coif3')
        fig, axs = plt.subplots(2)
        fig.suptitle("Heartbeat")
        axs[0].plot(signal)
        axs[1].plot(filtered_signal)
        plt.show()

        # ExtendedKalmanFilter()

    def __kalman_filter(self, data):
        # intial parameters
        n_iter = data.__len__()
        sz = (n_iter,)  # size of array
        # truth value (typo in example at top of p. 13 calls this z)
        x = -0.37727
        # observations (normal about x, sigma=0.1)
        z = np.random.normal(x, 0.1, size=sz)

        Q = 1e-5  # process variance

        # allocate space for arrays
        xhat = np.zeros(sz)      # a posteri estimate of x
        P = np.zeros(sz)         # a posteri error estimate
        xhatminus = np.zeros(sz)  # a priori estimate of x
        Pminus = np.zeros(sz)    # a priori error estimate
        K = np.zeros(sz)         # gain or blending factor

        R = 0.1**2  # estimate of measurement variance, change to see effect

        # intial guesses
        xhat[0] = 0.0
        P[0] = data[0]

        for k in range(1, n_iter):
            # time update
            xhatminus[k] = xhat[k - 1]
            Pminus[k] = P[k - 1] + Q

            # measurement update
            K[k] = Pminus[k] / (Pminus[k] + R)
            xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])
            P[k] = (1 - K[k]) * Pminus[k]
        return xhat

    def hampel_filter(self, data, filtsize=6):
        '''Detect outliers based on hampel filter

        Funcion that detects outliers based on a hampel filter.
        The filter takes datapoint and six surrounding samples.
        Detect outliers based on being more than 3std from window mean.
        See:
        https://www.mathworks.com/help/signal/ref/hampel.html

        Parameters
        ----------
        data : 1d list or array
            list or array containing the data to be filtered
        filtsize : int
            the filter size expressed the number of datapoints
            taken surrounding the analysed datapoint. a filtsize
            of 6 means three datapoints on each side are taken.
            total filtersize is thus filtsize + 1 (datapoint evaluated)
        Returns
        -------
        out :  array containing filtered data
        Examples
        --------
        >>> from .datautils import get_data, load_exampledata
        >>> data, _ = load_exampledata(0)
        >>> filtered = hampel_filter(data, filtsize = 6)
        >>> print('%i, %i' %(data[1232], filtered[1232]))
        497, 496
        '''

        # generate second list to prevent overwriting first
        # cast as array to be sure, in case list is passed
        output = np.copy(np.asarray(data))
        onesided_filt = filtsize // 2
        for i in range(onesided_filt, len(data) - onesided_filt - 1):
            dataslice = output[i - onesided_filt: i + onesided_filt]
            mad = utils.MAD(dataslice)
            median = np.median(dataslice)
            if output[i] > median + (3 * mad):
                output[i] = median
        return output

    def hard_filter_signal(
            self,
            data,
            cutoff,
            sample_rate,
            order=2,
            filtertype='lowpass',
            return_top=False):
        '''Apply the specified filter
        Function that applies the specified lowpass, highpass or bandpass filter to
        the provided dataset.
        Parameters
        ----------
        data : 1-dimensional numpy array or list
            Sequence containing the to be filtered data
        cutoff : int, float or tuple
            the cutoff frequency of the filter. Expects float for low and high types
            and for bandpass filter expects list or array of format [lower_bound, higher_bound]
        sample_rate : int or float
            the sample rate with which the passed data sequence was sampled
        order : int
            the filter order
            default : 2
        filtertype : str
            The type of filter to use. Available:
            - lowpass : a lowpass butterworth filter
            - highpass : a highpass butterworth filter
            - bandpass : a bandpass butterworth filter
            - notch : a notch filter around specified frequency range
            both the highpass and notch filter are useful for removing baseline wander. The notch
            filter is especially useful for removing baseling wander in ECG signals.
        Returns
        -------
        out : 1d array
            1d array containing the filtered data

        '''
        if filtertype.lower() == 'lowpass':
            b, a = butter_lowpass(cutoff, sample_rate, order=order)
        elif filtertype.lower() == 'highpass':
            b, a = butter_highpass(cutoff, sample_rate, order=order)
        elif filtertype.lower() == 'bandpass':
            assert isinstance(
                cutoff, tuple) or list or np.array, 'if bandpass filter is specified, \
    cutoff needs to be array or tuple specifying lower and upper bound: [lower, upper].'
            b, a = butter_bandpass(
                cutoff[0], cutoff[1], sample_rate, order=order)
        elif filtertype.lower() == 'notch':
            b, a = iirnotch(cutoff, Q=0.005, fs=sample_rate)
        else:
            raise ValueError('filtertype: %s is unknown, available are: \
    lowpass, highpass, bandpass, and notch' % filtertype)

        filtered_data = filtfilt(b, a, data)

        if return_top:
            return np.clip(filtered_data, a_min=0, a_max=None)
        else:
            return filtered_data

    def remove_baseline_wander(self, data, sample_rate, cutoff=0.05):
        '''removes baseline wander
        Function that uses a Notch filter to remove baseline
        wander from (especially) ECG signals
        Parameters
        ----------
        data : 1-dimensional numpy array or list
            Sequence containing the to be filtered data
        sample_rate : int or float
            the sample rate with which the passed data sequence was sampled
        cutoff : int, float
            the cutoff frequency of the Notch filter. We recommend 0.05Hz.
            default : 0.05
        Returns
        -------
        out : 1d array
            1d array containing the filtered data
        Examples
        --------
        >>> import heartpy as hp
        >>> data, _ = hp.load_exampledata(0)
        baseline wander is removed by calling the function and specifying
        the data and sample rate.
        >>> filtered = remove_baseline_wander(data, 100.0)
        '''

        return self.hard_filter_signal(
            data=data,
            cutoff=cutoff,
            sample_rate=sample_rate,
            filtertype='notch')

    def hampel_correcter(self, data, sample_rate):
        '''apply altered version of hampel filter to suppress noise.
        Function that returns te difference between data and 1-second
        windowed hampel median filter. Results in strong noise suppression
        characteristics, but relatively expensive to compute.
        Result on output measures is present but generally not large. However,
        use sparingly, and only when other means have been exhausted.
        Parameters
        ----------
        data : 1d numpy array
            array containing the data to be filtered
        sample_rate : int or float
            sample rate with which data was recorded

        Returns
        -------
        out : 1d numpy array
            array containing filtered data
        Examples
        --------
        >>> from .datautils import get_data, load_exampledata
        >>> data, _ = load_exampledata(1)
        >>> filtered = hampel_correcter(data, sample_rate = 116.995)
        '''

        return data - self.hampel_filter(data, filtsize=int(sample_rate))

    def __smooth(self, x, window_len=11, window='hanning'):
        """smooth the data using a window with requested size.

        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.

        input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

        output:
            the smoothed signal

        example:

        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)

        see also:

        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter

        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """

        if x.ndim != 1:
            print("smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            print("Input vector needs to be bigger than window size.")

        if window_len < 3:
            return x

        if window not in [
            'flat',
            'hanning',
            'hamming',
            'bartlett',
                'blackman']:
            print("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        return y

    def __get_all_r_peaks_indices(
            self,
            raw_list,
            qrs_method='pantompkins',
            skip_inital_last_segment=True,
            is_auth=False):
        r_peaks_indices_list = []
        # integrated_ecg_list = []
        for each_raw_list in raw_list:
            temp_array = np.array(each_raw_list)
            tp1 = [0] * len(temp_array)
            tempCombinedData = np.vstack((tp1, temp_array)).T
            # tempCombinedData = np.vstack((temp_array,tp1)).T
            if qrs_method == 'pantompkins':
                qrs_detector = pan_tompkins.QRSDetectorOffline(
                    ecg_data_path="",
                    verbose=False,
                    bps=self.fs,
                    log_data=False,
                    ecg_data_raw=tempCombinedData,
                    plot_data=False,
                    show_plot=False)

                lis = qrs_detector.qrs_peaks_indices
                if is_auth:
                    lis = lis[-25:-1]
                else:
                    lis = lis[0:99]
                r_peaks_indices_list.append(lis)
                # integrated_ecg_list.append(qrs_detector.integrated_ecg_measurements)
            elif qrs_method == 'wfdb':
                # normalized_raw_list = processing.normalize_bound(sig=each_raw_list, lb=0, ub=1)
                peaks = processing.XQRS(sig=each_raw_list, fs=self.fs)
                peaks.detect(verbose=False)
                lis = peaks.qrs_inds
                if skip_inital_last_segment:
                    size_list = lis.__len__()
                    if size_list > 3:
                        lis = lis[2:size_list - 1]
                if len(lis) == 0:
                    qrs_detector = pan_tompkins.QRSDetectorOffline(
                        ecg_data_path="",
                        verbose=False,
                        bps=self.fs,
                        log_data=False,
                        ecg_data_raw=tempCombinedData,
                        plot_data=False,
                        show_plot=False)
                    # wfdb.plot_items(signal=each_raw_list, ann_samp=[peaks.qrs_inds])
                # qrs_inds = processing.gqrs_detect(sig=each_raw_list, fs=self.fs)
                # search_radius = int(self.fs * 60 / 230)
                # corrected_peak_inds = processing.correct_peaks(each_raw_list, peak_inds=qrs_inds,
                    # search_radius=search_radius, smooth_window_size=150)
                # r_peaks_indices_list.append(corrected_peak_inds)
                lis = lis[0:99]
                r_peaks_indices_list.append(lis)
            elif qrs_method == 'gqrs':
                pass

        return r_peaks_indices_list

    def __get_dwt_decomposition(
            self,
            signal_to_decompose,
            which_wavelet='db4'):
        decomp_signal_level_list = pywt.wavedec(
            signal_to_decompose, which_wavelet, level=self.decompostion_level)
        return decomp_signal_level_list

    def __get_continous_wavelet(self, one_segment, file_name='', scale=50):
        # dt = self.fs/500
        # frequencies = pywt.scale2frequency(self.continuous_wavelet, np.arange(1,10)) / dt
        frequencies = np.arange(1, scale)
        coef, freqs = pywt.cwt(one_segment, frequencies,
                               self.continuous_wavelet)
        plt.plot(coef, freqs)
        if not file_name == '':
            plt.savefig(file_name, format='eps')
        # plt.savefig()
        # plt.show()
        return coef, freqs

    def __simple_average_of_list(self, lst):
        return sum(lst[0]) / len(lst[0])
        # return mean(lst)

    def __average_list(self, each_user_template, no_of_segments):
        for each_user in range(each_user_template.__len__()):
            for each_decomposition_level in range(
                    each_user_template[each_user].__len__()):
                each_user_template[each_user][each_decomposition_level] /= no_of_segments[each_user].__len__()
                # each_user_template[each_user][each_decomposition_level] = abs(each_user_template[each_user][each_decomposition_level])
        return each_user_template

    def __average_list_individual(
            self,
            each_user_template,
            each_user,
            no_of_segments):
        for each_decomposition_level in range(
                each_user_template[each_user].__len__()):
            if no_of_segments > 1:
                each_user_template[each_user][each_decomposition_level] /= no_of_segments
            # each_user_template[each_user][each_decomposition_level] = abs(each_user_template[each_user][each_decomposition_level])
        return each_user_template

    def __calculate_entropy(self, list_values):
        counter_values = Counter(list_values).most_common()
        probabilities = [elem[1] / len(list_values) for elem in counter_values]
        entropy = scipy.stats.entropy(probabilities)
        return entropy

    def __calculate_statistics(self, list_values):
        n5 = np.nanpercentile(list_values, 5)
        n25 = np.nanpercentile(list_values, 25)
        n75 = np.nanpercentile(list_values, 75)
        n95 = np.nanpercentile(list_values, 95)
        median = np.nanpercentile(list_values, 50)
        mean = np.nanmean(list_values)
        std = np.nanstd(list_values)
        var = np.nanvar(list_values)
        rms = np.nanmean(np.sqrt(list_values**2))
        max = np.nanmax(list_values)
        return [n5, n25, n75, n95, median, mean, std, var, rms, max]

    def __calculate_crossings(self, list_values):
        zero_crossing_indices = np.nonzero(
            np.diff(np.array(list_values) > 0))[0]
        no_zero_crossings = len(zero_crossing_indices)
        mean_crossing_indices = np.nonzero(
            np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
        no_mean_crossings = len(mean_crossing_indices)
        return [no_zero_crossings, no_mean_crossings]

    def __get_stats_features(self, list_values):
        entropy = self.__calculate_entropy(list_values)
        crossings = self.__calculate_crossings(list_values)
        statistics = self.__calculate_statistics(list_values)
        return [entropy] + crossings + statistics

    def __wavelet_distance(self, template, user_segment):
        diff_0 = abs(template[0]) - abs(user_segment[0])
        avg_diff_0 = self.__simple_average_of_list(diff_0)

        diff_1 = abs(template[1]) - abs(user_segment[1])
        avg_diff_1 = self.__simple_average_of_list(diff_1)

        diff_2 = abs(template[2]) - abs(user_segment[2])
        avg_diff_2 = self.__simple_average_of_list(diff_2)

        diff_3 = abs(template[3]) - abs(user_segment[3])
        avg_diff_3 = self.__simple_average_of_list(diff_3)

        diff_4 = abs(template[4]) - abs(user_segment[4])
        avg_diff_4 = self.__simple_average_of_list(diff_4)

        avg_diff = (abs(avg_diff_0) + abs(avg_diff_1) +
                    abs(avg_diff_2) + abs(avg_diff_3) + abs(avg_diff_4)) / 5
        return abs(avg_diff)

    def __list_offset_normalize(self, list1):
        # offset = list1.min()
        offset = list1[0]
        if offset < 0:
            offset = abs(offset)
            list1 = list1 + offset
        elif offset > 0:
            list1 = list1 - offset
        return list1

    def __do_pca(self, max_number_of_features, template, user_segment):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=max_number_of_features)
        principal_components_template = pca.fit_transform(template)
        principal_components_user_segment = pca.fit_transform(user_segment)
        print(principal_components_template)
        print(principal_components_user_segment)

    def __get_slope(self, point1, point2, one_segment):
        x_axs = (point2 - point1)
        y_axs = one_segment[point2] - one_segment[point1]
        slope = y_axs / x_axs
        return slope

    def __get_distance(self, point1, point2, one_segment):
        x_axs = (point1 - point2) * (point1 - point2)
        y_axs = (one_segment[point1] - one_segment[point2]) * \
            (one_segment[point1] - one_segment[point2])
        distance = math.sqrt(x_axs + y_axs)
        return distance

    def __get_fiducial_points(self, segmented_signal_for_all_users):
        for each_user in range(segmented_signal_for_all_users.__len__()):
            for each_segment in range(
                    segmented_signal_for_all_users[each_user].__len__()):
                one_segment = segmented_signal_for_all_users[each_user][each_segment][0]

                qrs_start_index = int(round(0.28 * one_segment.__len__()))
                qrs_stop_index = int(round(0.5 * one_segment.__len__()))
                for_r_peak = one_segment[qrs_start_index:qrs_stop_index]
                r_peak = int(np.argmax(for_r_peak))
                r_peak = qrs_start_index + r_peak

                temp = one_segment[qrs_start_index:r_peak]
                q_peak = int(qrs_start_index + np.argmin(temp))

                temp = one_segment[r_peak:qrs_stop_index]
                s_peak = int(r_peak + np.argmin(temp))

                p_start_index = int(round(0.1 * one_segment.__len__()))
                p_stop_index = int(round(0.225 * one_segment.__len__()))
                first_half = one_segment[p_start_index:p_stop_index]
                p_peak = int(p_start_index + np.argmax(first_half))

                t_start_index = int(round(0.6 * one_segment.__len__()))
                t_stop_index = int(round(0.75 * one_segment.__len__()))
                second_half = one_segment[t_start_index:t_stop_index]
                t_peak = int(t_start_index + np.argmax(second_half))

                n_peaks = [p_peak, q_peak, r_peak, s_peak, t_peak]
                n_peaks_time_diff = [
                    r_peak - q_peak,
                    s_peak - r_peak,
                    s_peak - q_peak,
                    q_peak - p_peak,
                    t_peak - p_peak,
                    s_peak - p_peak,
                    r_peak - p_peak,
                    t_peak - q_peak,
                    r_peak - t_peak,
                    t_peak - s_peak]
                n_peaks_amp_diff = [
                    one_segment[p_peak] - one_segment[q_peak],
                    one_segment[r_peak] - one_segment[p_peak],
                    one_segment[p_peak] - one_segment[s_peak],
                    one_segment[p_peak] - one_segment[t_peak],
                    one_segment[r_peak] - one_segment[s_peak],
                    one_segment[t_peak] - one_segment[s_peak],
                    one_segment[q_peak] - one_segment[s_peak],
                    one_segment[r_peak] - one_segment[q_peak],
                    one_segment[r_peak] - one_segment[t_peak]]
                n_peaks_amp = [
                    one_segment[p_peak],
                    one_segment[q_peak],
                    one_segment[r_peak],
                    one_segment[s_peak],
                    one_segment[t_peak]]

                pr_slope = self.__get_slope(p_peak, r_peak, one_segment)
                pq_slope = self.__get_slope(p_peak, q_peak, one_segment)
                st_slope = self.__get_slope(s_peak, t_peak, one_segment)
                rt_slope = self.__get_slope(r_peak, t_peak, one_segment)
                qs_slope = self.__get_slope(q_peak, s_peak, one_segment)

                n_peaks_slope = [
                    pr_slope,
                    pq_slope,
                    st_slope,
                    rt_slope,
                    qs_slope]
                fiducial_points = n_peaks
                for each in n_peaks_time_diff:
                    fiducial_points.append(each)
                for each in n_peaks_amp_diff:
                    fiducial_points.append(each)
                for each in n_peaks_amp:
                    fiducial_points.append(each)
                for each in n_peaks_slope:
                    fiducial_points.append(each)
                fiducial_points = np.array(fiducial_points)
                # print(fiducial_points)
                segmented_signal_for_all_users[each_user][each_segment].append(
                    fiducial_points)
        return segmented_signal_for_all_users

    def __get_n_peaks_diff_and_amp_diff(self, all_peaks, one_segment):
        fiducial_points = []
        for i in range(all_peaks.__len__() - 1):
            for j in range(i + 1, all_peaks.__len__()):
                fiducial_points.append(abs(all_peaks[i] - all_peaks[j]))
                fiducial_points.append(
                    abs(one_segment[all_peaks[i]] - one_segment[all_peaks[j]]))

        for each_peak in all_peaks:
            fiducial_points.append(one_segment[each_peak])

        pr_slope = self.__get_slope(all_peaks[-1], all_peaks[1], one_segment)
        pq_slope = self.__get_slope(all_peaks[-1], all_peaks[-2], one_segment)
        st_slope = self.__get_slope(all_peaks[2], all_peaks[3], one_segment)
        rt_slope = self.__get_slope(all_peaks[0], all_peaks[3], one_segment)
        qs_slope = self.__get_slope(all_peaks[2], all_peaks[-2], one_segment)

        pr_distance = self.__get_distance(
            all_peaks[-1], all_peaks[1], one_segment)
        pq_distance = self.__get_distance(
            all_peaks[-1], all_peaks[-2], one_segment)
        st_distance = self.__get_distance(
            all_peaks[2], all_peaks[3], one_segment)
        rt_distance = self.__get_distance(
            all_peaks[0], all_peaks[3], one_segment)
        qs_distance = self.__get_distance(
            all_peaks[2], all_peaks[-2], one_segment)

        n_peaks_slope = [pr_slope, pq_slope, st_slope, rt_slope, qs_slope]
        for each_value in n_peaks_slope:
            fiducial_points.append(each_value)
        n_peaks_distance = [
            pr_distance,
            pq_distance,
            st_distance,
            rt_distance,
            qs_distance]
        for each_value in n_peaks_distance:
            fiducial_points.append(each_value)

        return fiducial_points

    def __get_fiducial_points_single_rr(self, one_segment, plot=True):
        segment_len = one_segment.__len__()
        if segment_len <= 0:
            return -1
        r_start_index = 0
        r_end_index = segment_len - 1
        s_peak_offset = int(round(0.5 * one_segment.__len__()))
        s_peak_range = one_segment[0:s_peak_offset]
        s_peak = np.argmin(s_peak_range)

        t_peak_offset = int(round(0.4 * self.fs))
        t_peak_range = one_segment[s_peak:t_peak_offset]
        t_peak = s_peak + np.argmax(t_peak_range)

        q_peak_offset = int(round(0.28 * one_segment.__len__()))
        q_peak_range = one_segment[one_segment.__len__() - q_peak_offset:-1]
        q_peak = one_segment.__len__() - q_peak_offset + np.argmin(q_peak_range)

        p_peak_offset = int(round(0.2 * self.fs))
        p_peak_range = one_segment[q_peak - p_peak_offset:q_peak]
        p_peak = q_peak - p_peak_offset + np.argmax(p_peak_range)

        n_peaks = [r_start_index, r_end_index, s_peak, t_peak, q_peak, p_peak]
        n_rpeak = np.array(n_peaks)
        if plot:
            wfdb.plot_items(signal=one_segment, ann_samp=[n_rpeak])
        fiducial_points = self.__get_n_peaks_diff_and_amp_diff(
            n_peaks, one_segment)
        fiducial_points = np.array(fiducial_points)
        return fiducial_points, 0

    def __get_fiducial_points_single(self, one_segment, plot=False):
        # one_segment = segmented_signal_for_all_users[each_user][each_segment][0]
        if "window" in self.segmentation_method:
            qrs_start_index = int(round(0.28 * one_segment.__len__()))
            qrs_stop_index = int(round(0.5 * one_segment.__len__()))
            for_r_peak = one_segment[qrs_start_index:qrs_stop_index]
            r_peak = int(np.argmax(for_r_peak))
            r_peak = qrs_start_index + r_peak

            temp = one_segment[qrs_start_index:r_peak]
            val = qrs_start_index - r_peak
            if val == 0:
                if plot:
                    self.plot_signal(one_segment)
                # self.plot_signal(one_segment)
                # plt.plot(one_segment, ann)
                # plt.show()
                print("segment may not be correct")
                fiducial_points = []
                return fiducial_points, -99
            q_peak = int(qrs_start_index + np.argmin(temp))

            temp = one_segment[r_peak:qrs_stop_index]
            s_peak = int(r_peak + np.argmin(temp))

            p_start_index = int(round(0.1 * one_segment.__len__()))
            p_stop_index = int(round(0.225 * one_segment.__len__()))
            first_half = one_segment[p_start_index:p_stop_index]
            p_peak = int(p_start_index + np.argmax(first_half))

            t_start_index = int(round(0.6 * one_segment.__len__()))
            t_stop_index = int(round(0.75 * one_segment.__len__()))
            second_half = one_segment[t_start_index:t_stop_index]
            t_peak = int(t_start_index + np.argmax(second_half))

            n_peaks = [p_peak, q_peak, r_peak, s_peak, t_peak]

            n_rpeak = np.array(n_peaks)

            if plot:
                wfdb.plot_items(signal=one_segment, ann_samp=[n_rpeak])
            # wfdb.plot(one_segment,ann=n_peaks)
            # plt.plot(n_peaks, marker='o')
            # plt.show()
            n_peaks_time_diff = [
                r_peak - q_peak,
                s_peak - r_peak,
                s_peak - q_peak,
                q_peak - p_peak,
                t_peak - p_peak,
                s_peak - p_peak,
                r_peak - p_peak,
                t_peak - q_peak,
                r_peak - t_peak,
                t_peak - s_peak]
            n_peaks_amp_diff = [
                one_segment[p_peak] - one_segment[q_peak],
                one_segment[r_peak] - one_segment[p_peak],
                one_segment[p_peak] - one_segment[s_peak],
                one_segment[p_peak] - one_segment[t_peak],
                one_segment[r_peak] - one_segment[s_peak],
                one_segment[t_peak] - one_segment[s_peak],
                one_segment[q_peak] - one_segment[s_peak],
                one_segment[r_peak] - one_segment[q_peak],
                one_segment[r_peak] - one_segment[t_peak]]
            n_peaks_amp = [
                one_segment[p_peak],
                one_segment[q_peak],
                one_segment[r_peak],
                one_segment[s_peak],
                one_segment[t_peak]]

            pr_slope = self.__get_slope(p_peak, r_peak, one_segment)
            pq_slope = self.__get_slope(p_peak, q_peak, one_segment)
            st_slope = self.__get_slope(s_peak, t_peak, one_segment)
            rt_slope = self.__get_slope(r_peak, t_peak, one_segment)
            qs_slope = self.__get_slope(q_peak, s_peak, one_segment)

            pr_distance = self.__get_distance(p_peak, r_peak, one_segment)
            pq_distance = self.__get_distance(p_peak, q_peak, one_segment)
            st_distance = self.__get_distance(s_peak, t_peak, one_segment)
            rt_distance = self.__get_distance(r_peak, t_peak, one_segment)
            qs_distance = self.__get_distance(q_peak, s_peak, one_segment)

            n_peaks_slope = [pr_slope, pq_slope, st_slope, rt_slope, qs_slope]
            n_peaks_distance = [
                pr_distance,
                pq_distance,
                st_distance,
                rt_distance,
                qs_distance]
            fiducial_points = n_peaks
            for each in n_peaks_time_diff:
                fiducial_points.append(each)
            for each in n_peaks_amp_diff:
                fiducial_points.append(each)
            for each in n_peaks_amp:
                fiducial_points.append(each)
            for each in n_peaks_slope:
                fiducial_points.append(each)
            for each in n_peaks_distance:
                fiducial_points.append(each)
            fiducial_points = np.array(fiducial_points)
            return fiducial_points, 0
        else:
            return self.__get_fiducial_points_single_rr(
                one_segment, plot=False)

    def __normalize_signal(self, data):
        # data = (data - min(data)) / (max(data) - min(data))
        ret_data = []
        for each_data in data:
            ret_data.append(data + min(each_data))
        return ret_data

    def __get_euclidean_distance(self, template, user_segment, plot=False):
        result_list = []
        for each_decomposition_level in range(self.decompostion_level):
            list1 = template[each_decomposition_level][0]
            list1 = self.__list_offset_normalize(list1)
            slice_len = int(list1.__len__() * 0.7)
            list1 = list1[:slice_len]

            list2 = user_segment[each_decomposition_level]
            list2 = self.__list_offset_normalize(list2)

            slice_len = int(list2.__len__() * 0.7)
            list2 = list2[:slice_len]

            list1 = self.__normalize_signal(list1)
            list2 = self.__normalize_signal(list2)

            a = np.array(list1)
            b = np.array(list2)
            # feature_set_a = self.__get_stats_features(a)
            # feature_set_b = self.__get_stats_features(b)
            # plot = True
            if plot:
                plt.plot(a, color='Red')
                plt.plot(b, color='Blue')
                plt.show()
            # d, cost_matrix, acc_cost_matrix, path = dtw(a, b, dist=manhattan_distance)
            # self.__do_pca(5, a, b)

            dist = 0
            if a.size != b.size:
                # print("Length different")
                dist, path = fastdtw(a, b, dist=euclidean)
            else:
                dist = distance.euclidean(a, b)
            # distance, path = fastdtw(a,b, dist=euclidean)
            # distance, path = fastdtw(feature_set_a,feature_set_b, dist=euclidean)
            result_list.append(dist)
        return sum(result_list)

    def __dynamic_time_warping(self, template, user_segment, plot=False):
        result_list = []

        for each_decomposition_level in range(self.decompostion_level):

            list1 = template[each_decomposition_level][0]
            # list1 = self.__list_offset_normalize(list1)
            # slice_len = int(list1.__len__() * 0.7)
            # list1 = list1[:slice_len]

            list2 = user_segment[each_decomposition_level]
            # list2 = self.__list_offset_normalize(list2)

            # slice_len = int(list2.__len__() * 0.7)
            # list2 = list2[:slice_len]

            # list1 = self.__normalize_signal(list1)
            # list2 = self.__normalize_signal(list2)

            # a = np.array(list1)
            # b = np.array(list2)
            # feature_set_a = self.__get_stats_features(a)
            # feature_set_b = self.__get_stats_features(b)
            # plot = True
            if plot:
                plt.plot(a, color='Red')
                plt.plot(b, color='Blue')
                plt.show()
            # d, cost_matrix, acc_cost_matrix, path = dtw(a, b, dist=manhattan_distance)
            # self.__do_pca(5, a, b)
            dist, path = fastdtw(list1, list2, dist=euclidean)
            # distance, path = fastdtw(feature_set_a,feature_set_b, dist=euclidean)
            result_list.append(dist)
        return sum(result_list)

    def __dynamic_time_warping_fiducial(
            self, template, user_segment, plot=False):
        result_list = []

        list1 = template

        list2 = user_segment
        # list1 = self.__normalize_signal(list1)
        # list2 = self.__normalize_signal(list2)

        a = np.array(list1)
        b = np.array(list2)
        # feature_set_a = self.__get_stats_features(a)
        # feature_set_b = self.__get_stats_features(b)
        # plot = True
        if plot:
            plt.plot(a, color='Red')
            plt.plot(b, color='Blue')
            plt.show()
        # d, cost_matrix, acc_cost_matrix, path = dtw(a, b, dist=manhattan_distance)
        # self.__do_pca(5, a, b)
        dist = 0
        try:
            dist = distance.euclidean(a, b)
        except BaseException:
            dist = 0
        result_list.append(dist)
        return sum(result_list)

    def __autocorrelation(self, x):
        """
        Compute the autocorrelation of the signal, based on the properties of the
        power spectral density of the signal.
        """
        xp = x - np.mean(x)
        f = np.fft.fft(xp)
        p = np.array([np.real(v)**2 + np.imag(v)**2 for v in f])
        pi = np.fft.ifft(p)
        return np.real(pi)[:x.size // 2] / np.sum(xp**2)

    def autocorr1(self, x):
        r2 = np.fft.ifft(np.abs(np.fft.fft(x))**2).real
        return r2[:len(x) // 2]

    def __align_signal_segments_user(self, segmented_signal_for_all_users):
        for each_user in range(segmented_signal_for_all_users.__len__()):
            for each_segment in range(
                    segmented_signal_for_all_users[each_user].__len__()):
                one_segment = segmented_signal_for_all_users[each_user][each_segment][0]
                # rms = np.nanmean(np.sqrt(one_segment**2))
                # print(rms)
                rpeak = np.argmax(one_segment)
                minima = 30
                # temp = one_segment[:rpeak]
                # minima = np.argmin(temp)
                first_half = one_segment[:rpeak - minima]
                # temp = one_segment[:rpeak]
                # minima = np.argmin(temp)
                second_half = one_segment[rpeak + minima:]

                # first_half = one_segment[:rpeak-25]
                # second_half = one_segment[rpeak+25:]
                from scipy import signal
                # new_sampling_rate = first_half.__len__() * 5
                # new_sampling_rate = int(round(new_sampling_rate))
                # first_half = signal.resample(first_half, new_sampling_rate)
                # # first_half = self.__butter_bandpass_filter(first_half,0.5, 40,new_sampling_rate, 4)

                # new_sampling_rate = second_half.__len__() * 5
                # new_sampling_rate = int(round(new_sampling_rate))
                # second_half = signal.resample(second_half,new_sampling_rate)
                # second_half = self.__butter_bandpass_filter(second_half,1, 40,new_sampling_rate, 3)

                t_peak_offset = first_half.__len__() + (2 * minima)
                # rpeak = t_peak_offset - minima
                # first_half.__len__()
                # first_half = wfdb.processing.resample_sig(first_half,self.fs,500);
                # second_half = wfdb.processing.resample_sig(second_half,self.fs,500);
                p_peak = np.argmax(first_half)
                t_peak = np.argmax(second_half) + t_peak_offset
                first_half = np.array(first_half)
                first_half = np.array(first_half)
                second_half = np.array(second_half)
                temp = np.array(one_segment[rpeak - minima:rpeak + minima])

                final = np.concatenate(
                    (first_half, temp, second_half), axis=None)
                rpeak = np.argmax(final)

                n_rpeak = np.array([rpeak, p_peak, t_peak])

                # test = self.__autocorrelation(final)
                test = self.autocorr1(final)
                segmented_signal_for_all_users[each_user][each_segment][0] = test
                # print(test)
                # final = self.__butter_bandpass_filter(final,1, 40,new_sampling_rate, 3)

                # all_hard_peaks, all_soft_peaks = processing.find_peaks(one_segment)
                # wfdb.plot_items(signal=final, ann_samp=[n_rpeak])
                # wfdb.plot_items(signal=data, ann_samp=[all_soft_peaks])
        return segmented_signal_for_all_users

    def __find_min_distance_among_template(
            self, template, user_segment, plot=False):
        result_list = []
        for each_template in template:
            a = np.array(each_template[self.decompostion_level - 1][0])
            b = np.array(user_segment[self.decompostion_level - 1])
            a.flatten()
            b.flatten()
            feature_set_a = self.__get_stats_features(a)
            feature_set_b = self.__get_stats_features(b)
            peak_diff = feature_set_a[12] - feature_set_b[12]
            result_list.append(abs(peak_diff))
        # print(min(result_list))
        return min(result_list)

    def __find_min_distance_avgeraged_template(
            self, template, user_segment, plot=False):
        a = np.array(template[self.decompostion_level - 1][0])
        # a = np.array(template[1][0])
        a.flatten()
        b = np.array(user_segment[self.decompostion_level - 1])
        # b = np.array(user_segment[1])
        b.flatten()
        feature_set_a = self.__get_stats_features(a)
        feature_set_b = self.__get_stats_features(b)
        sum_a = sum(feature_set_a)
        sum_b = sum(feature_set_b)
        # print(feature_set_a[12] - feature_set_b[12])
        peak_diff = feature_set_a[12] - feature_set_b[12]
        # print(peak_diff)

        # self.plot_signal(template[0][0])
        if plot:
            plt.plot(a, color='Red')
            plt.plot(b, color='Blue')
            plt.show()
        diff_0 = np.correlate(a, b)
        # avg_diff_0 = self.__simple_average_of_list(diff_0)

        # return abs(diff_0)
        return abs(peak_diff)

    def __avg_template_update(
            self,
            each_user_template,
            decomposed_data_for_each_segment,
            user):
        skipped_segment = 0
        for each_segment in range(decomposed_data_for_each_segment.__len__()):
            for each_level_decomposition in range(self.decompostion_level + 1):
                diff_len = each_user_template[user][each_level_decomposition][0].__len__(
                ) - decomposed_data_for_each_segment[each_segment][0][each_level_decomposition].__len__()
                if (diff_len != 0):
                    skipped_segment += 1
                    continue
                each_user_template[user][each_level_decomposition] += decomposed_data_for_each_segment[each_segment][0][each_level_decomposition]
        for each_level_decomposition in range(self.decompostion_level + 1):
            each_user_template[user][each_level_decomposition] /= (
                decomposed_data_for_each_segment.__len__() - skipped_segment)
        return each_user_template

    def __get_segment_for_template_update(self, file_name):
        ecg_signal_list = self.step1_read_signal(True, file_name, 1)
        ecg_signal_filtered_list = self.step2_filter_signal(
            ecg_signal_list, False)
        segmented_signal_for_all_users = self.step3_segment_signal(
            ecg_signal_filtered_list)
        decomposed_data_for_each_segment = self.step4_feature_extraction(
            segmented_signal_for_all_users)
        return decomposed_data_for_each_segment[0]

    def __template_generate_5_chunks(self, decomposed_data_for_each_segment):
        each_user_template = []
        for each_user in range(decomposed_data_for_each_segment.__len__()):
            skipped_segment = 0
            segment_number = -1
            each_user_template.append([])
            seg_len = math.ceil(
                (decomposed_data_for_each_segment[each_user].__len__()) / 5)
            for each_segment in range(seg_len):
                each_user_template[each_user].append([])
                for each_level_decomposition in range(
                        self.decompostion_level + 1):
                    each_user_template[each_user][each_segment].append([])
            for each_segment in range(
                    decomposed_data_for_each_segment[each_user].__len__()):
                if each_segment % 5 == 0:
                    if segment_number > -1:
                        if skipped_segment < 5:
                            for each_level__decomposition in range(
                                    self.decompostion_level + 1):
                                each_user_template[each_user][segment_number][each_level__decomposition] /= 5 - skipped_segment
                        else:
                            continue
                    segment_number += 1
                    skipped_segment = 0
                    # each_user_template[each_user][segment_number].append([])
                    for each_level_decomposition in range(
                            self.decompostion_level + 1):
                        each_user_template[each_user][segment_number][each_level_decomposition].append(
                            decomposed_data_for_each_segment[each_user][each_segment][0][each_level_decomposition])
                    continue
                for each_level_decomposition in range(
                        self.decompostion_level + 1):
                    diff_len = each_user_template[each_user][segment_number][each_level_decomposition][0].__len__(
                    ) - decomposed_data_for_each_segment[each_user][each_segment][0][each_level_decomposition].__len__()
                    if (diff_len != 0):
                        skipped_segment += 1
                        continue
                    each_user_template[each_user][segment_number][each_level_decomposition] += decomposed_data_for_each_segment[each_user][each_segment][0][each_level_decomposition]
            # each_user_template = self.__average_list_individual(each_user_template,each_user,decomposed_data_for_each_segment[each_user].__len__()-skipped_segment)
        # data = self.__average_list(each_user_template,decomposed_data_for_each_segment)
        return each_user_template

    def __template_generate_avg_way(self, decomposed_data_for_each_segment):
        each_user_template = []
        for each_user in range(decomposed_data_for_each_segment.__len__()):
            skipped_segment = 0
            each_user_template.append([])
            for each_level_decomposition in range(self.decompostion_level + 1):
                each_user_template[each_user].append([])
            for each_segment in range(
                    decomposed_data_for_each_segment[each_user].__len__()):
                if each_segment == 0:
                    for each_level_decomposition in range(
                            self.decompostion_level + 1):
                        each_user_template[each_user][each_level_decomposition].append(
                            decomposed_data_for_each_segment[each_user][0][0][each_level_decomposition])
                    continue
                for each_level_decomposition in range(
                        self.decompostion_level + 1):
                    diff_len = each_user_template[each_user][each_level_decomposition][0].__len__(
                    ) - decomposed_data_for_each_segment[each_user][each_segment][0][each_level_decomposition].__len__()
                    if (diff_len != 0):
                        skipped_segment += 1
                        continue
                    each_user_template[each_user][each_level_decomposition] += decomposed_data_for_each_segment[each_user][each_segment][0][each_level_decomposition]
            each_user_template = self.__average_list_individual(
                each_user_template,
                each_user,
                decomposed_data_for_each_segment[each_user].__len__() -
                skipped_segment)
        # data = self.__average_list(each_user_template,decomposed_data_for_each_segment)
        return each_user_template

    def __template_generate_avg_way_fiducial(
            self, decomposed_data_for_each_segment):
        each_user_template = []
        init_list = []
        last_index = 0
        if "window" in self.segmentation_method:
            last_index = 39
        else:
            last_index = 46
        for i in range(0, last_index):
            init_list.append(0.0)
        for each_user in range(decomposed_data_for_each_segment.__len__()):
            each_user_template.append(np.array(init_list))
        for each_user in range(decomposed_data_for_each_segment.__len__()):
            skipped_segment = 0
            # each_user_template.append([])
            if decomposed_data_for_each_segment[each_user].__len__() == 0:
                print(
                    "No valid segment in user " +
                    str(each_user) +
                    " Initializing Empty Segment")
                each_user_template.append([])
                continue
            for each_segment in range(
                    decomposed_data_for_each_segment[each_user].__len__()):
                try:
                    # if each_segment == 0:
                    #     each_user_template.append(decomposed_data_for_each_segment[each_user][each_segment][1])
                    #     continue
                    each_user_template[each_user] += decomposed_data_for_each_segment[each_user][each_segment][1]
                except BaseException:
                    print(
                        "Skipping the segment for user = " +
                        str(each_user) +
                        " segment = " +
                        str(each_segment))
            # each_user_template = self.__average_list_individual(each_user_template,each_user,decomposed_data_for_each_segment[each_user].__len__()-skipped_segment)
        # data = self.__average_list(each_user_template,decomposed_data_for_each_segment)
            each_user_template[each_user] = each_user_template[each_user] / \
                (decomposed_data_for_each_segment[each_user].__len__() - 1)
        return each_user_template

    def __template_generate_multiple(self, decomposed_data_for_each_segment):
        pass

    ##########################################################################
    #
    #                                   PUBLIC FUNCTIONS
    #
    ##########################################################################

    def simple_average_of_list(self, lst, discard_segment=0):
        data = 0
        iterate = 0
        for i in range(lst.__len__()):
            if(i < discard_segment):
                continue
            data += lst[i]
            iterate += 1
        return data / iterate

    def plot_filtered_signal(
            self,
            unfiltered_signal,
            signal_to_plot,
            markers=None):
        # if markers != None:
        # plt.plot(signal_to_plot,markevery=markers , marker="o")
        # else:
        plt.plot(unfiltered_signal)
        plt.plot(signal_to_plot)
        plt.legend(loc='best')
        plt.show()

    def plot_zoomed_signal(
            self,
            signal_to_plot,
            zoom_signal,
            rr_signal,
            markers=None):
        fig, ax = plt.subplots(3)
        # ax.plot(signal_to_plot, linewidth=5)
        ax[0].plot(signal_to_plot)
        ax[0].set_ylabel('Amplitude', fontsize='large')
        ax[0].set_title('ECG Signal', fontsize='large', fontweight='bold')

        # for xc in rpeaks:
        #     ax[0].axvline(x=xc)

        ax[1].plot(zoom_signal, linewidth=3)
        ax[1].set_ylabel('Amplitude', fontsize='large')
        ax[1].set_title(
            'ECG Segment Window',
            fontsize='large',
            fontweight='bold')

        ax[2].plot(rr_signal, linewidth=3)
        ax[2].set_ylabel('Amplitude', fontsize='large')
        ax[2].set_title(
            'ECG Segment RR Interval',
            fontsize='large',
            fontweight='bold')
        # ax.set(xlabel='Samples', ylabel='Amplitude',
        #     title='Segmented ECG Signal', labelpad=18)
        # ax[1].grid(linestyle='dotted', linewidth=2)

        # leg = plt.legend()
        plt.savefig(
            'src/Result/DB_Segment/zoomed_plot_1',
            format='eps',
            dpi=100)
        plt.show()

    def plot_signal(self, signal_to_plot, markers=None):
        # if markers != None:
        # plt.plot(signal_to_plot,markevery=markers , marker="o")
        # else:
        fig, ax = plt.subplots()
        ax.plot(signal_to_plot, linewidth=7)
        ax.set_xlabel('Samples', fontsize='large')
        ax.set_ylabel('Amplitude', fontsize='large')
        ax.set_title('ECG Segment', fontsize='large', fontweight='bold')
        # ax.set(xlabel='Samples', ylabel='Amplitude',
        #     title='Segmented ECG Signal', labelpad=18)
        ax.grid()

        # leg = plt.legend()
        plt.savefig(
            'src/Result/DB_Segment/segmented_signal_plot',
            format='eps')
        plt.show()

    def read_wfdb_files_2(self, file_name_list, count):
        file_name_list = file_name_list[:count]
        raw_list = []
        for eachFile in file_name_list:
            data = []
            for iter in range(1, 3):
                if iter == 2:
                    eachFile.replace('rec_1', 'rec_2')
                # temp = eachFile.format(iter)
            # raw_list.append([])
                temp = self.read_signal_wfdb(eachFile)
                data.extend(temp)
            data = np.array(data)
            # offset = abs(data.min())
            # data = data+offset
            # data = (data/3.3)*4095
            raw_list.append(data)
        return raw_list

    def read_wfdb_files(self, file_name_list, count):
        file_name_list = file_name_list[:count]
        raw_list = []
        for eachFile in file_name_list:
            # raw_list.append([])
            data = self.read_signal_wfdb(eachFile)
            raw_list.append(data)
        return raw_list

    def read_signal_wfdb(self, file_name):
        record = wfdb.rdrecord(file_name, channels=[0])
        # record = wfdb.rdrecord(file_name)
        # record, fields = wfdb.rdsamsp(file_name,channels=[1])
        # self.fs = fields['fs']
        self.fs = record.fs
        # data = record.p_signal.flatten()
        data = record.p_signal[:, 0]
        # data = record[:,0]
        data = processing.normalize_bound(sig=data, lb=0, ub=1)

        # offset = abs(data.min())
        # data = data+offset
        voltage = 3
        adc_resolution = record.adc_res[0]
        if adc_resolution == 0:
            adc_resolution = 16
        resolution = 2 ** adc_resolution
        data = (data / voltage) * resolution
        data = np.round(data)
        # data = processing.normalize_bound(sig=data, lb=0, ub=1)
        # peaks = processing.XQRS(sig=record[:,0], fs=fields['fs'])
        # peaks.detect()
        # lis = peaks.qrs_inds
        # for each_data in data:
        #     new_data.append(each_data[0])
        return data

    def ecg_wave_detector(self, ecg, rpeaks):
        """
        Returns the localization of the P, Q, T waves. This function needs massive help!
        Parameters
        ----------
        ecg : list or ndarray
            ECG signal (preferably filtered).
        rpeaks : list or ndarray
            R peaks localization.
        Returns
        ----------
        ecg_waves : dict
            Contains wave peaks location indices.
        Example
        Notes
        ----------
        *Details*
        - **Cardiac Cycle**: A typical ECG showing a heartbeat consists of a P wave, a QRS complex and a T wave.The P wave represents the wave of depolarization that spreads from the SA-node throughout the atria. The QRS complex reflects the rapid depolarization of the right and left ventricles. Since the ventricles are the largest part of the heart, in terms of mass, the QRS complex usually has a much larger amplitude than the P-wave. The T wave represents the ventricular repolarization of the ventricles. On rare occasions, a U wave can be seen following the T wave. The U wave is believed to be related to the last remnants of ventricular repolarization.

        """
        q_waves = []
        p_waves = []
        q_waves_starts = []
        s_waves = []
        t_waves = []
        t_waves_starts = []
        t_waves_ends = []
        for index, rpeak in enumerate(rpeaks[:-3]):

            try:
                epoch_before = np.array(ecg)[int(rpeaks[index - 1]):int(rpeak)]
                epoch_before = epoch_before[int(
                    len(epoch_before) / 2):len(epoch_before)]
                epoch_before = list(reversed(epoch_before))

                q_wave_index = np.min(find_peaks(epoch_before))
                q_wave = rpeak - q_wave_index
                p_wave_index = q_wave_index + \
                    np.argmax(epoch_before[q_wave_index:])
                p_wave = rpeak - p_wave_index

                inter_pq = epoch_before[q_wave_index:p_wave_index]
                inter_pq_derivative = np.gradient(inter_pq, 2)
                q_start_index = find_closest_in_list(
                    len(inter_pq_derivative) / 2, find_peaks(inter_pq_derivative))
                q_start = q_wave - q_start_index

                q_waves.append(q_wave)
                p_waves.append(p_wave)
                q_waves_starts.append(q_start)
            except ValueError:
                pass
            except IndexError:
                pass

            try:
                epoch_after = np.array(ecg)[int(rpeak):int(rpeaks[index + 1])]
                epoch_after = epoch_after[0:int(len(epoch_after) / 2)]

                s_wave_index = np.min(find_peaks(epoch_after))
                s_wave = rpeak + s_wave_index
                t_wave_index = s_wave_index + \
                    np.argmax(epoch_after[s_wave_index:])
                t_wave = rpeak + t_wave_index

                inter_st = epoch_after[s_wave_index:t_wave_index]
                inter_st_derivative = np.gradient(inter_st, 2)
                t_start_index = find_closest_in_list(
                    len(inter_st_derivative) / 2, find_peaks(inter_st_derivative))
                t_start = s_wave + t_start_index
                t_end = np.min(find_peaks(epoch_after[t_wave_index:]))
                t_end = t_wave + t_end

                s_waves.append(s_wave)
                t_waves.append(t_wave)
                t_waves_starts.append(t_start)
                t_waves_ends.append(t_end)
            except ValueError:
                pass
            except IndexError:
                pass

    def step1_read_signal(self, file, file_name, read_file_count):
        '''
        # Read Signal from files
        '''
        if file:
            if read_file_count > 1:
                return self.step1_read_files(file_name, read_file_count, True)
            else:
                return self.step1_read_files(file_name, read_file_count, False)
        else:
            pass

    def step1_read_files(self, file_name, read_file_count, is_list):
        file_list = []
        raw_list = []
        if is_list:
            for num in range(1, read_file_count + 1):
                raw_list.append([])
                temp = file_name.format(num)
                file_list.append(temp)
        else:
            raw_list.append([])
            file_list.append(file_name)

        # ecg_log(5,file_list)
        # ecg_log(5,raw_list)

        file_count = 0
        for eachFile in file_list:
            try:
                fp = open(eachFile, 'r')
                eachLine = fp.readline()
                while eachLine:
                    eachLine = fp.readline()
                    try:
                        temp = int(eachLine)
                    except BaseException:
                        # ecg_log("Except = " + eachLine)
                        continue
                    raw_list[file_count].append(temp)
                # raw_list[file_count] = processing.normalize_bound(sig=raw_list[file_count], lb=0, ub=1)
                file_count += 1
            finally:
                fp.close()
        return raw_list

    def step2_filter_signal(
            self,
            signal_to_filter_list,
            plot=False,
            which_filter="bandpass"):
        band_pass_filtered_signal_list = []
        for signal_to_filter in signal_to_filter_list:
            filtered_signal = signal_to_filter
            if "filfit" in which_filter:
                filtered_signal = self.__filfit_filter(
                    filtered_signal, 0.1, 10, self.fs, 3)
            elif "bandpass" in which_filter:
                filtered_signal = self.__butter_bandpass_filter(
                    filtered_signal, 0.1, 15, self.fs, 3)
            elif "FIR" in which_filter:
                filtered_signal = self.__fir_filter(
                    filtered_signal, self.fs, 15, 15)
            elif "kalman" in which_filter:
                filtered_signal = self.__kalman_filter(signal_to_filter)
            # filtered_signal = self.__smooth(filtered_signal, 9, 'flat')
            # filtered_signal = processing.normalize_bound(sig=filtered_signal, lb=0, ub=1)
            band_pass_filtered_signal_list.append(filtered_signal)
            if plot:
                self.plot_signal(filtered_signal)
        return band_pass_filtered_signal_list

    def step2_filter_signal__one(
            self,
            signal_to_filter,
            which_filter='bandpass',
            plot=False):
        filtered_signal = signal_to_filter
        if "filfit" in which_filter:
            filtered_signal = self.__filfit_filter(
                filtered_signal, 0.1, 10, self.fs, 3)
        elif "bandpass" in which_filter:
            filtered_signal = self.__butter_bandpass_filter(
                filtered_signal, 0.1, 15, self.fs, 3)
        elif "FIR" in which_filter:
            filtered_signal = self.__fir_filter(
                filtered_signal, self.fs, 15, 15)
        elif "kalman" in which_filter:
            filtered_signal = self.__kalman_filter(signal_to_filter)
        # filtered_signal = self.__smooth(filtered_signal, 9, 'flat')
        # filtered_signal = processing.normalize_bound(sig=filtered_signal, lb=0, ub=1)
        if plot:
            self.plot_signal(filtered_signal)
        return filtered_signal

        # return self.__cheby_lowpass_filter(signal_to_filter, 50,360,1,10)

    def step3_segment_signal_ex(
            self,
            signal_to_segment_list,
            plot=False,
            qrs_method='pantompkins'):
        r_peaks_indices_list = self.__get_all_r_peaks_indices(
            signal_to_segment_list, qrs_method=qrs_method)
        segmented_signal_for_all_users = []
        user_iterate = 0
        for signal_to_segment in signal_to_segment_list:
            segmented_signal_for_all_users.append([])
            segment_iterate = 0
            for each_r_peak in range(
                    r_peaks_indices_list[user_iterate].__len__()):
                if segment_iterate == r_peaks_indices_list[user_iterate].__len__(
                ) - 1:
                    continue
                # segmented_list[each_user].append(raw_list[each_user][each_r_peak-100:each_r_peak+100])
                # data = signal_to_segment[int(each_r_peak-(0.25 * self.fs)):int(each_r_peak+(0.45 * self.fs))]
                data = signal_to_segment[r_peaks_indices_list[user_iterate][
                    each_r_peak]:r_peaks_indices_list[user_iterate][each_r_peak + 1]]
                if data.any():
                    # data = processing.normalize_bound(sig=data, lb=0, ub=1)
                    # data = self.__normalize_signal(data)
                    segmented_signal_for_all_users[user_iterate].append([])
                    segmented_signal_for_all_users[user_iterate][segment_iterate].append(
                        data)
                    plot = True
                    # if user_iterate == 12 or user_iterate == 14:
                    #     plot = True
                    # else:
                    #     plot = False
                    if plot:
                        # self.plot_signal(signal_to_segment)
                        self.plot_signal(data)
                    segment_iterate += 1
                else:
                    pass
            user_iterate += 1

        # return r_peaks_indices_list,integrated_signal_list
        return segmented_signal_for_all_users

    def step3_get_rpeaks_wfdb(
            self,
            signal,
            plot=False,
            skip_inital_last_segment=True):
        peaks = processing.XQRS(sig=signal, fs=self.fs)
        peaks.detect(verbose=False)
        lis = peaks.qrs_inds
        if skip_inital_last_segment:
            size_list = lis.__len__()
            if size_list > 3:
                lis = lis[2:size_list - 1]
        return lis

    def step3_segment_signal(
            self,
            signal_to_segment_list,
            plot=False,
            qrs_method='pantompkins',
            is_auth=False):
        r_peaks_indices_list = self.__get_all_r_peaks_indices(
            signal_to_segment_list, qrs_method=qrs_method, is_auth=is_auth)
        segmented_signal_for_all_users = []
        user_iterate = 0
        for signal_to_segment in signal_to_segment_list:
            segmented_signal_for_all_users.append([])
            segment_iterate = 0
            for each_r_peak in r_peaks_indices_list[user_iterate]:
                if segment_iterate == r_peaks_indices_list[user_iterate].__len__(
                ) - 1:
                    continue
                # segmented_list[each_user].append(raw_list[each_user][each_r_peak-100:each_r_peak+100])
                data = signal_to_segment[int(
                    each_r_peak - (0.3 * self.fs)):int(each_r_peak + (0.5 * self.fs))]
                if data.any():
                    data = processing.normalize_bound(sig=data, lb=0, ub=1)
                    # data = self.__normalize_signal(data)
                    segmented_signal_for_all_users[user_iterate].append([])
                    segmented_signal_for_all_users[user_iterate][segment_iterate].append(
                        data)
                    # plot = True
                    # if user_iterate == 5:
                    #     plot = True
                    # else:
                    #     plot = False
                    # if is_auth:
                    #     plot = True
                    if plot:
                        # self.plot_signal(signal_to_segment)
                        self.plot_signal(data)
                    segment_iterate += 1
                else:
                    pass
            user_iterate += 1

        # return r_peaks_indices_list,integrated_signal_list
        return segmented_signal_for_all_users

    def step4_feature_extraction(self, segmented_signal_for_all_users):
        data = []
        user_iterate = 0

        for each_user in segmented_signal_for_all_users:
            data.append([])
            segment_iterate = 0
            for each_user_each_segment in each_user:
                one_segment = each_user_each_segment[0]
                # ecg_authenticate.plot_signal(one_segment[0])

                # rpeak = np.argmax(one_segment)
                # t_peak_offset = rpeak+20
                # first_half = one_segment[:rpeak-20]
                # second_half = one_segment[rpeak+20:]
                # p_peak = np.argmax(first_half)
                # t_peak = np.argmax(second_half) + t_peak_offset

                # n_rpeak = np.array([rpeak,p_peak,t_peak])

                # all_hard_peaks, all_soft_peaks = processing.find_peaks(one_segment)
                # wfdb.plot_items(signal=one_segment, ann_samp=[n_rpeak])
                # wfdb.plot_items(signal=data, ann_samp=[all_soft_peaks])
                # file_name = "Images/" + str(user_iterate) + "_" + str(segment_iterate)  + ".eps"
                file_name = ''
                decomp_each_segment_level_list = self.__get_dwt_decomposition(
                    one_segment, self.discrete_wavelet_type)
                data[user_iterate].append([])
                data[user_iterate][segment_iterate].append(
                    decomp_each_segment_level_list)
                # contious_decomp_each_segment = self.__get_continous_wavelet(one_segment, scale = self.continuous_wt_scale, file_name=file_name)
                if self.feature_method == 'fiducial':
                    fiducial_points, error_code = self.__get_fiducial_points_single(
                        one_segment)
                    if error_code == -99:
                        continue
                    else:
                        data[user_iterate][segment_iterate].append(
                            fiducial_points)

                # data[user_iterate][segment_iterate].append(contious_decomp_each_segment)
                # print(decomp_each_segment_level_list)
                segment_iterate += 1
            user_iterate += 1
        return data

    def step4_feature_extraction_ex(self, segmented_signal_for_all_users):
        data = []
        user_iterate = 0

        for each_user in segmented_signal_for_all_users:
            data.append([])
            segment_iterate = 0
            for each_user_each_segment in each_user:
                one_segment = each_user_each_segment
                # ecg_authenticate.plot_signal(one_segment[0])

                # rpeak = np.argmax(one_segment)
                # t_peak_offset = rpeak+20
                # first_half = one_segment[:rpeak-20]
                # second_half = one_segment[rpeak+20:]
                # p_peak = np.argmax(first_half)
                # t_peak = np.argmax(second_half) + t_peak_offset

                # n_rpeak = np.array([rpeak,p_peak,t_peak])

                # all_hard_peaks, all_soft_peaks = processing.find_peaks(one_segment)
                # wfdb.plot_items(signal=one_segment, ann_samp=[n_rpeak])
                # wfdb.plot_items(signal=data, ann_samp=[all_soft_peaks])
                # file_name = "Images/" + str(user_iterate) + "_" + str(segment_iterate)  + ".eps"
                # if (user_iterate == 1) and (segment_iterate == 24):
                #     print("error")
                #     pass
                file_name = ''
                decomp_each_segment_level_list = self.__get_dwt_decomposition(
                    one_segment, self.discrete_wavelet_type)
                data[user_iterate].append([])
                data[user_iterate][segment_iterate].append(
                    decomp_each_segment_level_list)
                # contious_decomp_each_segment = self.__get_continous_wavelet(one_segment, scale = self.continuous_wt_scale, file_name=file_name)
                if self.feature_method == 'fiducial':
                    fiducial_points, error_code = self.__get_fiducial_points_single(
                        one_segment)
                    if error_code == -99:
                        pass
                    else:
                        data[user_iterate][segment_iterate].append(
                            fiducial_points)

                # data[user_iterate][segment_iterate].append(contious_decomp_each_segment)
                # print(decomp_each_segment_level_list)
                segment_iterate += 1
            if user_iterate == 5:
                print("debug")
            user_iterate += 1
        return data

    def step5_template_generate(self,
                                decomposed_data_for_each_segment,
                                which_template_generate='template_avg'):
        if "template_avg" in which_template_generate:
            if self.feature_method == 'fiducial':
                return self.__template_generate_avg_way_fiducial(
                    decomposed_data_for_each_segment)
            else:
                return self.__template_generate_avg_way(
                    decomposed_data_for_each_segment)
        elif 'template_5_set' in which_template_generate:
            return self.__template_generate_5_chunks(
                decomposed_data_for_each_segment)

    def get_threshold(self, template, user_segment, plot=False):
        if self.feature_method == 'fiducial':
            return self.__dynamic_time_warping_fiducial(
                template, user_segment, plot=plot)
        else:
            return self.__dynamic_time_warping(
                template, user_segment, plot=plot)

    def step6_matching(self, template, user_segment, threshold, plot=False):
        # data = threshold
        if self.feature_method == 'fiducial':
            r_data = self.__dynamic_time_warping_fiducial(
                template, user_segment, plot=plot)
        else:
            r_data = self.__dynamic_time_warping(
                template, user_segment, plot=plot)
        # r_data = self.__get_euclidean_distance(template,user_segment, plot=plot)
        # print(r_data)
        # if 'template_avg' in self.template_method:
        #     data = self.__find_min_distance_avgeraged_template(template,user_segment, plot=plot)
        # elif 'template_5_set' in self.template_method:
        #     data = self.__find_min_distance_among_template(template,user_segment, plot=plot)
        # print("Threshold = " + str(r_data)+ "," + str(threshold))
        if r_data < threshold:
            return True
        else:
            return False
        # return data[0]
        # return self.__wavelet_distance(template,user_segment)

    def step7_template_update(self, each_user_template, file_name, user):
        decomposed_data_for_each_segment = self.__get_segment_for_template_update(
            file_name)
        return self.__avg_template_update(
            each_user_template, decomposed_data_for_each_segment, user)

    def get_new_segmented_signal(
            self,
            ecg_signal_list,
            qrs_method='pantompkins',
            which_filter="FIR",
            segmentation_method="window",
            auth=False,
            plot=False):
        result = []
        other_result = []
        segmentation_method = self.segmentation_method
        # data = self.__normalize_signal(ecg_signal_list)
        order = int(0.3 * self.fs)
        for each_signal in ecg_signal_list:
            filtered_signal = []
            r_peaks = []

            # filtered = self.hampel_correcter(each_signal, self.fs)
            # filtered, noise = self.__remove_baseline(each_signal)
            # self.__plot_filtered_signal_comp(each_signal, filtered)
            # self.plot_signal(noise)

            if which_filter in [
                "FIR",
                "butter",
                "cheby1",
                "cheby2",
                "ellip",
                    "bessel"]:
                filtered_signal, _, _ = tools.filter_signal(signal=each_signal,
                                                            ftype=which_filter,
                                                            band='bandpass',
                                                            order=order,
                                                            frequency=[3, 45],
                                                            sampling_rate=self.fs)
            else:
                filtered_signal = self.__ekf_filter(each_signal)
                # filtered_signal = each_signal
            # plot = True
            if plot:
                fig, axs = plt.subplots(2)
                fig.suptitle("Heartbeat")
                axs[0].plot(each_signal)
                axs[1].plot(filtered_signal)
                plt.show()

            if qrs_method == 'pantompkins':
                rpeaks, = ecg.hamilton_segmenter(
                    signal=filtered_signal, sampling_rate=self.fs)
            elif qrs_method == "gamboa":
                rpeaks, = ecg.gamboa_segmenter(
                    signal=filtered_signal, sampling_rate=self.fs, tol=0.002)
            elif qrs_method == "engzee":
                rpeaks, = ecg.engzee_segmenter(
                    signal=filtered_signal, sampling_rate=self.fs, threshold=0.48)
            elif qrs_method == "christov":
                rpeaks, = ecg.christov_segmenter(
                    signal=filtered_signal, sampling_rate=self.fs)
            elif qrs_method == "ssf":
                rpeaks, = ecg.ssf_segmenter(
                    signal=filtered_signal, sampling_rate=self.fs, threshold=20, before=0.03, after=0.01)
            elif qrs_method == "pekkanen":
                rpeaks = ecg.segmenter_pekkanen(
                    ecg=filtered_signal,
                    sampling_rate=self.fs,
                    window_size=5.0,
                    lfreq=5.0,
                    hfreq=15.0)
            else:
                rpeaks = self.step3_get_rpeaks_wfdb(filtered_signal)

            rpeaks, = ecg.correct_rpeaks(signal=filtered_signal,
                                         rpeaks=rpeaks,
                                         sampling_rate=self.fs,
                                         tol=0.05)
            rpeaks_len = rpeaks.__len__()
            rpeaks_auth = int(rpeaks_len * 0.8)
            if auth:
                rpeaks = rpeaks[rpeaks_auth:]
            else:
                rpeaks = rpeaks[:rpeaks_auth]

            templates, rpeaks_1 = ecg.extract_heartbeats(signal=filtered_signal,
                                                         rpeaks=rpeaks,
                                                         sampling_rate=self.fs,
                                                         before=0.2,
                                                         after=0.4,
                                                         segmentation_method=self.segmentation_method)
            templates_2, rpeaks_2 = ecg.extract_heartbeats(signal=filtered_signal,
                                                           rpeaks=rpeaks,
                                                           sampling_rate=self.fs,
                                                           before=0.2,
                                                           after=0.4,
                                                           segmentation_method='RR')
            i = 0
            t_segs = []
            rmax = []
            for each_template in templates:
                if i < 6:
                    for each_s in each_template:
                        t_segs.append(each_s)
                        rmax.append(np.argmax(each_s))
                i += 1
                if i == 6:
                    self.plot_zoomed_signal(
                        t_segs, each_template, templates_2[0])

            # args = (filtered_signal, rpeaks, templates)
            # names = ('filtered', 'rpeaks', 'templates')
            # other_result.append(utils.ReturnTuple(args, names))
            # out = ecg.ecg(signal=each_signal, sampling_rate=self.fs, show=True)
            # if templates.__len__() > 0:
            #     result.append(templates)
            result.append(templates)

        return result

    def trim_signal(self, ecg_signal_list):
        for iterate in range(ecg_signal_list.__len__()):
            temp = ecg_signal_list[iterate]
            max_len = int(25 * self.fs)
            last = ecg_signal_list[iterate].__len__() - (2 * self.fs)
            ecg_signal_list[iterate] = ecg_signal_list[iterate][last -
                                                                max_len - 1:last - 1]
        return ecg_signal_list

    def enroll_users(
            self,
            file_name,
            count,
            qrs_method='pantompkins',
            which_filter="FIR",
            is_ecg_id=1,
            is_wfdb=False):
        if is_wfdb:
            if is_ecg_id == 2:
                ecg_signal_list = self.read_wfdb_files_2(file_name, count)
                pass
            else:
                ecg_signal_list = self.read_wfdb_files(file_name, count)

            # ecg_signal_list = self.read_wfdb_ecg_id(file_name)
        else:
            ecg_signal_list = self.step1_read_signal(True, file_name, count)
        # for each_signal in ecg_signal_list:
        #     self.plot_signal(each_signal)
        # ecg_signal_list = self.trim_signal(ecg_signal_list)

        new_segmented_signal_for_all_users = self.get_new_segmented_signal(
            ecg_signal_list, qrs_method=qrs_method, which_filter=which_filter)

        # self.__get_fiducial_points_single__ex(new_result)
        '''
        Old way of getting
        # ecg_signal_filtered_list = self.step2_filter_signal(ecg_signal_list,plot=False,which_filter=which_filter)
        # segmented_signal_for_all_users = self.step3_segment_signal(ecg_signal_filtered_list,qrs_method=qrs_method,plot=False)
        '''
        # segmented_signal_for_all_users = self.__align_signal_segments_user(segmented_signal_for_all_users)
        self.print_segmented_signal(new_segmented_signal_for_all_users)
        decomposed_data_for_each_segment = self.step4_feature_extraction_ex(
            new_segmented_signal_for_all_users)
        # self.print_decomposed_signal(decomposed_data_for_each_segment)
        # self.print_fiducial_points(decomposed_data_for_each_segment)
        # self.print_continuous_decomposed_data(decomposed_data_for_each_segment)
        each_user_template = self.step5_template_generate(
            decomposed_data_for_each_segment,
            which_template_generate=self.template_method)
        return each_user_template

    def generate_threshold(
            self,
            file_name,
            each_user_template,
            user,
            qrs_method='pantompkins',
            which_filter="bandpass",
            is_wfdb=False,
            plot=False,
            is_auth=False):
        ecg_signal_list = []
        threshold_list = []
        if is_wfdb:
            ecg_signal_list = self.read_wfdb_files(file_name, 1)
        else:
            ecg_signal_list = self.step1_read_signal(True, file_name, 1)
        ecg_signal_list = self.trim_signal(ecg_signal_list)
        segmented_signal_for_all_users = self.get_new_segmented_signal(
            ecg_signal_list, auth=True, qrs_method=qrs_method, which_filter=which_filter)

        '''
        ecg_signal_filtered_list = self.step2_filter_signal(ecg_signal_list,plot=False,which_filter=which_filter)
        segmented_signal_for_all_users = self.step3_segment_signal(ecg_signal_filtered_list,qrs_method=qrs_method,is_auth=True)
        decomposed_data_for_each_segment = self.step4_feature_extraction(segmented_signal_for_all_users)
        '''
        decomposed_data_for_each_segment = self.step4_feature_extraction_ex(
            segmented_signal_for_all_users)
        result = 0
        if self.feature_method == 'fiducial':
            for each_segment in decomposed_data_for_each_segment[0]:
                if (each_user_template[user].__len__() == 0) or (
                        each_segment.__len__() < 2):
                    threshold_list.append(np.NaN)
                else:
                    temp_result = self.get_threshold(
                        each_user_template[user], each_segment[1], plot=plot)
                    threshold_list.append(temp_result)
        else:
            for each_segment in decomposed_data_for_each_segment[0]:
                temp_result = self.get_threshold(
                    each_user_template[user], each_segment[0], plot=plot)
                threshold_list.append(temp_result)
        if threshold_list.__len__() > 0:
            return mean(threshold_list)
        else:
            return np.NaN

    def authenticate_user(
            self,
            file_name,
            each_user_template,
            user,
            qrs_method='pantompkins',
            threshold=0.0026,
            which_filter="bandpass",
            is_wfdb=False,
            plot=False,
            is_auth=False):
        ecg_signal_list = []
        if is_wfdb:
            ecg_signal_list = self.read_wfdb_files(file_name, 1)
        else:
            ecg_signal_list = self.step1_read_signal(True, file_name, 1)
        ecg_signal_list = self.trim_signal(ecg_signal_list)
        segmented_signal_for_all_users = self.get_new_segmented_signal(
            ecg_signal_list, auth=True, qrs_method=qrs_method, which_filter=which_filter)
        '''
        ecg_signal_filtered_list = self.step2_filter_signal(ecg_signal_list,plot=False,which_filter=which_filter)
        segmented_signal_for_all_users = self.step3_segment_signal(ecg_signal_filtered_list,qrs_method=qrs_method,is_auth=is_auth)
        decomposed_data_for_each_segment = self.step4_feature_extraction(segmented_signal_for_all_users)
        '''
        decomposed_data_for_each_segment = self.step4_feature_extraction_ex(
            segmented_signal_for_all_users)
        return decomposed_data_for_each_segment
        result = 0
        for each_segment in decomposed_data_for_each_segment[0]:
            temp_result = self.step6_matching(
                each_user_template[user],
                each_segment[0],
                threshold=threshold,
                plot=plot)
            if temp_result:
                result += 1
        return [result, decomposed_data_for_each_segment[0].__len__()]

    def print_segmented_signal(self, data):
        dump_csv = open("segmented_signal.csv", "w")
        for each_user in range(data.__len__()):
            for each_segment in range(data[each_user].__len__()):
                dump_csv.write(str(each_user) + "," + str(each_segment))
                for each_value in range(
                        data[each_user][each_segment].__len__()):
                    dump_csv.write(
                        "," + str(data[each_user][each_segment][each_value]))
                # self.plot_signal(data[each_user][each_segment][0])
                dump_csv.write("\n")
        return data

    def print_decomposed_signal(self, data):
        dump_csv = open("decompsed_signal.csv", "w")
        for each_user in range(data.__len__()):
            for each_segment in range(data[each_user].__len__()):
                for each_level_decomposition in range(
                        data[each_user][each_segment][0].__len__()):
                    # if each_level_decomposition<4:
                    #     continue
                    # print(str(each_user) + "," + str(each_segment), end = '')
                    dump_csv.write(str(each_user) + "," + str(each_segment))
                    for each_value in range(
                            data[each_user][each_segment][0][each_level_decomposition].__len__()):
                        dump_csv.write(
                            "," + str(data[each_user][each_segment][0][each_level_decomposition][each_value]))
                        # print("," + str(data[each_user][each_segment][0][each_level_decomposition][each_value]), end = '')
                    dump_csv.write("\n")
                    # print()

    def print_fiducial_points(self, data):
        dump_csv = open("fiducial_points.csv", "w")
        header = 'user,segment, p_peak, q_peak, r_peak, s_peak, t_peak, rq_time, sr_time, sq_time, qp_time, tp_time, sp_time, rp_time, tq_time, rt_time, ts_time, pq_amplitude, rp_amplitude, ps_amplitude, pt_amplitude, rs_amplitude, ts_amplitude, qs_amplitude, qr_amplitude, qt_amplitude, p_peak_amplitude, q_peak_amplitude, r_peak_amplitude, s_peak_amplitude, t_peak_amplitude, pr_slope, pq_slope, st_slope, rt_slope, qs_slope, pr_distance, pq_distance, st_distance, rt_distance, qs_distance, \n'
        dump_csv.write(header)
        for each_user in range(data.__len__()):
            for each_segment in range(data[each_user].__len__()):
                # print(str(each_user) + "," + str(each_segment), end = '')
                dump_csv.write(str(each_user) + "," + str(each_segment))
                for each_value in range(
                        data[each_user][each_segment][1].__len__()):
                    dump_csv.write(
                        "," + str(data[each_user][each_segment][1][each_value]))
                    # print("," + str(data[each_user][each_segment][0][each_level_decomposition][each_value]), end = '')
                dump_csv.write("\n")
                # print()

    def print_continuous_decomposed_data(self, data):
        dump_csv = open("continuous_waavelet.csv", "w")
        # header = 'user,segment, p_peak, q_peak, r_peak, s_peak, t_peak, rq_time, sr_time, sq_time, qp_time, tp_time, sp_time, rp_time, tq_time, rt_time, ts_time, pq_amplitude, rp_amplitude, ps_amplitude, pt_amplitude, rs_amplitude, ts_amplitude, qs_amplitude, qr_amplitude, qt_amplitude, p_peak_amplitude, q_peak_amplitude, r_peak_amplitude, s_peak_amplitude, t_peak_amplitude, pr_slope, pq_slope, st_slope, rt_slope, qs_slope, pr_distance, pq_distance, st_distance, rt_distance, qs_distance, \n'
        # dump_csv.write(header)
        for each_user in range(data.__len__()):
            for each_segment in range(data[each_user].__len__()):
                # print(str(each_user) + "," + str(each_segment), end = '')
                for each_value in range(
                        data[each_user][each_segment][2][0].__len__()):
                    dump_csv.write(str(each_user) + "," + str(each_segment))
                    dump_csv.write(
                        "," + str(data[each_user][each_segment][2][1][each_value]))
                    for each_val in range(
                            data[each_user][each_segment][2][0][each_value].__len__()):
                        dump_csv.write(
                            "," + str(data[each_user][each_segment][2][0][each_value][each_val]))
                    # print("," + str(data[each_user][each_segment][0][each_level_decomposition][each_value]), end = '')
                    dump_csv.write("\n")
                    # print()



class ecg_framework():
    def __get_ecg_id_file_list(self, file_name, no_of_users):
        file_name_list = []
        for i in range(1, no_of_users + 1):
            folder_name = file_name.format(i)
            file_name_list.append(glob.glob(folder_name)[0])

        for it in range(file_name_list.__len__()):
            file_name_list[it] = file_name_list[it].replace('.dat', '')
        return file_name_list

    def __get_ecg_ptb_file_list(self, file_name, no_of_users):
        file_name_list = []
        for i in range(1, no_of_users + 1):
            if i < 10:
                temp_file_name = file_name + 'patient00{0}/*.dat'
            elif i < 100:
                temp_file_name = file_name + 'patient0{0}/*.dat'
            else:
                temp_file_name = file_name + 'patient{0}/*.dat'
            folder_name = temp_file_name.format(i)
            try:
                file_name_list.append(glob.glob(folder_name)[0])
            except BaseException:
                print(folder_name)
                pass

        for it in range(file_name_list.__len__()):
            file_name_list[it] = file_name_list[it].replace('.dat', '')
        return file_name_list

    def __get_ecg_qt_db_list(self, file_name):
        file_name_list = glob.glob(file_name)
        for it in range(file_name_list.__len__()):
            file_name_list[it] = file_name_list[it].replace('.dat', '')
        return file_name_list

    def __get_fantasia_file_list(self, file_name):
        file_name_list = glob.glob(file_name)
        for it in range(file_name_list.__len__()):
            file_name_list[it] = file_name_list[it].replace('.dat', '')
        del file_name_list[5]
        return file_name_list

    def enroll_user_framework(
            self,
            ecg_authenticate,
            which_db,
            qrs_method,
            no_of_users,
            which_filter):
        #######################################################################
        '''
        #
        #  ENROLL USER FROM FRAMEWORK
        #  ecg_authenticate = ecg_authenticator object
        #  which_db = "ecg_id" or "ecg_id_csv" or "qt_db"
        #  qrs_method = "wfdb" or "pantompkins"
        #  no_of_users = number of user you want to enroll
        #  which_filer = "bandpass" or "filfit" or "FIR" or "kalman"
        #
        '''
        #######################################################################
        #######################################################################
        #
        #                             ECG ID DATABASE FROM PHYSIONET in CSV
        #
        #######################################################################
        each_user_template = []
        if "ecg_id_csv" in which_db:
            file_name = 'ecgid/4_ecg_filtered/person_0{0}_rec_1_ecg.csv'
            each_user_template = ecg_authenticate.enroll_users(
                file_name,
                no_of_users,
                is_wfdb=False,
                qrs_method=qrs_method,
                which_filter=which_filter)

        #######################################################################
        #
        #                             QT DATABASE FROM PHYSIONET
        #
        #######################################################################
        elif "qt_db" in which_db:
            file_name = 'qt-database-1.0.0/*.dat'
            file_name_list = self.__get_ecg_qt_db_list(file_name)
            each_user_template = ecg_authenticate.enroll_users(
                file_name_list,
                no_of_users,
                is_wfdb=True,
                qrs_method=qrs_method,
                which_filter=which_filter)

        #######################################################################
        #
        #                             ECG ID DATABASE FROM PHYSIONET
        #
        #######################################################################
        elif "ecg_id" in which_db:
            file_name = 'src/data/ecg-id-database-1.0.0/Person_{0}/rec_1.dat'
            file_name_list = self.__get_ecg_id_file_list(
                file_name, no_of_users)
            each_user_template = ecg_authenticate.enroll_users(
                file_name_list,
                no_of_users,
                is_wfdb=True,
                is_ecg_id=1,
                qrs_method=qrs_method,
                which_filter=which_filter)

        #######################################################################
        #
        #                             ECG PTB DATABASE FROM PHYSIONET
        #                               (Need to rename files in DB)
        #                                       DON'T USE
        #
        #######################################################################
        elif "ecg_ptb" in which_db:
            file_name = 'src/data/physionet.org/files/ptbdb/1.0.0/'
            file_name_list = self.__get_ecg_ptb_file_list(
                file_name, no_of_users)
            each_user_template = ecg_authenticate.enroll_users(
                file_name_list,
                no_of_users,
                is_wfdb=True,
                qrs_method=qrs_method,
                which_filter=which_filter)

        elif "fantasia" in which_db:
            file_name = 'src/data/physionet.org/files/fantasia/1.0.0/*.dat'
            file_name_list = self.__get_fantasia_file_list(file_name)
            each_user_template = ecg_authenticate.enroll_users(
                file_name_list,
                no_of_users,
                is_wfdb=True,
                qrs_method=qrs_method,
                which_filter=which_filter)
        elif "mitdb" in which_db:
            file_name = 'src/data/physionet.org/files/mitdb/1.0.0/*.dat'
            file_name_list = self.__get_fantasia_file_list(file_name)
            each_user_template = ecg_authenticate.enroll_users(
                file_name_list,
                no_of_users,
                is_wfdb=True,
                qrs_method=qrs_method,
                which_filter=which_filter)
        elif "cebsdb" in which_db:
            file_name = 'src/data/physionet.org/files/cebsdb/1.0.0/*.dat'
            file_name_list = self.__get_fantasia_file_list(file_name)
            each_user_template = ecg_authenticate.enroll_users(
                file_name_list,
                no_of_users,
                is_wfdb=True,
                qrs_method=qrs_method,
                which_filter=which_filter)
        elif "CYBHi" in which_db or 'ecg_bg' in which_db:
            file_name = 'src/data/converted/{}/data/*.dat'.format(which_db)
            file_name_list = self.__get_fantasia_file_list(file_name)
            each_user_template = ecg_authenticate.enroll_users(
                file_name_list,
                no_of_users,
                is_wfdb=True,
                qrs_method=qrs_method,
                which_filter=which_filter)
        elif "NeomedECGDataset" in which_db:
            file_name = 'src/data/converted/NeomedECGDataset-v0/new_data/*.dat'.format(
                which_db)
            file_name_list = self.__get_fantasia_file_list(file_name)
            each_user_template = ecg_authenticate.enroll_users(
                file_name_list,
                no_of_users,
                is_wfdb=True,
                qrs_method=qrs_method,
                which_filter=which_filter)
        return each_user_template

    def learn_threshold(
            self,
            ecg_authenticate,
            each_user_template,
            qrs_method,
            which_filter,
            which_db,
            all_user):
        threshold_list = []
        file_name = ''
        wfdb = True
        if which_db == 'ecg_id':
            file_name = 'src/data/ecg-id-database-1.0.0/Person_{0}/rec_1'
            wfdb = True
            for i in range(1, all_user + 1):
                file_name_list = []
                temp_name = file_name.format(i)
                file_name_list.append(temp_name)
                result = ecg_authenticate.generate_threshold(
                    file_name_list,
                    each_user_template,
                    i - 1,
                    qrs_method=qrs_method,
                    is_wfdb=wfdb,
                    plot=False,
                    which_filter=which_filter)
                # result = ecg_authenticate.authenticate_user(file_name_list,each_user_template,3-1, threshold=2.48, qrs_method=qrs_method, is_wfdb=True, plot=False, which_filter=which_filter)
                threshold_list.append(result)
        elif which_db == 'qt_db':
            file_name = 'src/data/qt-database-1.0.0/*.dat'
            wfdb = True
        elif (which_db == 'fantasia') or (which_db == 'mitdb') or (which_db == 'cebsdb') or (which_db == 'CYBHi') or (which_db == 'ecg_bg'):
            file_name = ''
            if which_db == 'CYBHi' or which_db == 'ecg_bg':
                file_name = 'src/data/converted/{}/data/*.dat'.format(which_db)
            else:
                file_name = 'src/data/physionet.org/files/{0}/1.0.0/*.dat'.format(
                    which_db)
            wfdb = True
            i = 0
            file_name_list_all = self.__get_fantasia_file_list(file_name)
            for each_file in file_name_list_all:
                if i >= all_user:
                    break
                i += 1
                file_name_list = []
                file_name_list.append(each_file)
                result = ecg_authenticate.generate_threshold(
                    file_name_list,
                    each_user_template,
                    i - 1,
                    qrs_method=qrs_method,
                    is_wfdb=wfdb,
                    plot=False,
                    which_filter=which_filter,
                    is_auth=False)
                # result = ecg_authenticate.authenticate_user(file_name_list,each_user_template,3-1, threshold=2.48, qrs_method=qrs_method, is_wfdb=True, plot=False, which_filter=which_filter)
                threshold_list.append(result)
        elif "ecg_ptb" in which_db:
            file_name = 'src/data/physionet.org/files/ptbdb/1.0.0/'
            wfdb = True
            i = 0
            file_name_list_all = self.__get_ecg_ptb_file_list(
                file_name, all_user)
            all_user = file_name_list_all.__len__()
            for each_file in file_name_list_all:
                if i >= all_user:
                    break
                i += 1
                file_name_list = []
                file_name_list.append(each_file)
                result = ecg_authenticate.generate_threshold(
                    file_name_list,
                    each_user_template,
                    i - 1,
                    qrs_method=qrs_method,
                    is_wfdb=wfdb,
                    plot=False,
                    which_filter=which_filter,
                    is_auth=False)
                # result = ecg_authenticate.authenticate_user(file_name_list,each_user_template,3-1, threshold=2.48, qrs_method=qrs_method, is_wfdb=True, plot=False, which_filter=which_filter)
                threshold_list.append(result)

        return threshold_list

    def authenticate_user_framework(
            self,
            ecg_authenticate,
            each_user_template,
            which_db,
            qrs_method,
            which_filter,
            threshold_template,
            all_user=1,
            which_user=1):
        #######################################################################
        '''
        #
        #  Authenticate User
        #  ecg_authenticate = ecg_authenticator object
        #  each_user_template = template generated from enroll user
        #  which_db = "ecg_id" or "ecg_id_csv" or "qt_db"
        #  qrs_method = "wfdb" or "pantompkins"
        #  which_filer = "bandpass" or "filfit" or "FIR" or "kalman"
        #  all_user = 1 -> authenticate with one user
        #           > 1 -> authenticate with all users -> user numbers
        #
        '''
        #######################################################################
        #######################################################################
        #
        #                             ECG ID DATABASE FROM PHYSIONET CSV
        #
        #######################################################################
        if all_user == 1:
            if "ecg_id_csv" in which_db:
                file_name = 'ecgid/4_ecg_filtered/person_03_rec_3_ecg.csv'
                decomposed_data_for_each_segment = ecg_authenticate.authenticate_user(
                    file_name,
                    each_user_template,
                    3 - 1,
                    qrs_method=qrs_method,
                    is_wfdb=False,
                    plot=False,
                    threshold=threshold_template[
                        3 - 1],
                    which_filter=which_filter)
                for each_segment in decomposed_data_for_each_segment[0]:
                    temp_result = ecg_authenticate.step6_matching(
                        each_user_template[3 - 1], each_segment[0], threshold=(threshold_template[user] * threshold_offset), plot=False)
                    if temp_result:
                        result += 1
                pass_val = result / \
                    decomposed_data_for_each_segment[0].__len__()
                if pass_val > 0.5:
                    return True
                else:
                    return False

        #######################################################################
        #
        #                             ECG ID DATABASE FROM PHYSIONET
        #
        #######################################################################
            elif "ecg_id" in which_db:
                file_name_list = []
                result = 0
                test_file = 'src/data/ecg-id-database-1.0.0/Person_{}/rec_1'.format(
                    which_user)
                file_name_list.append(test_file)
                # result = ecg_authenticate.authenticate_user(file_name_list, each_user_template, 3-1, threshold=2.48, qrs_method=qrs_method, is_wfdb=True, plot=False, which_filter=which_filter)
                decomposed_data_for_each_segment = ecg_authenticate.authenticate_user(
                    file_name_list,
                    each_user_template,
                    3 - 1,
                    qrs_method=qrs_method,
                    is_wfdb=True,
                    plot=False,
                    threshold=threshold_template[
                        3 - 1],
                    which_filter=which_filter)
                for each_segment in decomposed_data_for_each_segment[0]:
                    user = which_user - 1
                    threshold_offset = 1
                    temp_result = ecg_authenticate.step6_matching(
                        each_user_template[user], each_segment[0], threshold=(
                            threshold_template[user] * threshold_offset), plot=False)
                    if temp_result:
                        result += 1

                pass_val = result / \
                    decomposed_data_for_each_segment[0].__len__()
                if pass_val > 0.5:
                    return True
                else:
                    return False
            elif "mitdb" in which_db:
                file_name_list = []
                result = 0
                test_file = 'src/data/ecg-id-database-1.0.0/Person_{}/rec_2'.format(
                    which_user)
                file_name_list.append(test_file)
                # result = ecg_authenticate.authenticate_user(file_name_list, each_user_template, 3-1, threshold=2.48, qrs_method=qrs_method, is_wfdb=True, plot=False, which_filter=which_filter)
                decomposed_data_for_each_segment = ecg_authenticate.authenticate_user(
                    file_name_list,
                    each_user_template,
                    3 - 1,
                    qrs_method=qrs_method,
                    is_wfdb=True,
                    plot=False,
                    threshold=threshold_template[
                        3 - 1],
                    which_filter=which_filter,
                    is_auth=False)
                for each_segment in decomposed_data_for_each_segment[0]:
                    user = which_user - 1
                    threshold_offset = 1
                    temp_result = ecg_authenticate.step6_matching(
                        each_user_template[user], each_segment[0], threshold=(
                            threshold_template[user] * threshold_offset), plot=False)
                    if temp_result:
                        result += 1

                pass_val = result / \
                    decomposed_data_for_each_segment[0].__len__()
                if pass_val > 0.5:
                    return True
                else:
                    return False

        #######################################################################
        #
        #                             Authenticate all users
        #
        #######################################################################
        else:
            if "ecg_id" in which_db:
                ret = []
                file_name = 'src/data/ecg-id-database-1.0.0/Person_{0}/rec_1'
                threshold_iterate = 1
                step = 0.1
                start_thresh = 1.0
                end_thresh = start_thresh + ((step * threshold_iterate) - step)
                correct_user = [0] * threshold_iterate
                false_user = [0] * threshold_iterate
                ret_result = []
                for j in range(1, all_user + 1):
                    for i in range(1, all_user + 1):
                        file_name_list = []
                        temp_name = file_name.format(i)
                        file_name_list.append(temp_name)
                        user = j - 1
                        # print("---------------------------------------------------------------------------------------------------")
                        # thresh = (threshold_template[user]*1.1)
                        thresh = 10
                        decomposed_data_for_each_segment = ecg_authenticate.authenticate_user(
                            file_name_list,
                            each_user_template,
                            user,
                            qrs_method=qrs_method,
                            is_wfdb=True,
                            plot=False,
                            threshold=thresh,
                            which_filter=which_filter)
                        for threshold_offset in np.arange(1.0, 1.1, 1):
                            if decomposed_data_for_each_segment[0].__len__(
                            ) == 0:
                                continue
                            # for threshold_offset in
                            # np.arange(start_thresh,end_thresh,step):
                            result = 0
                            f_method = 0
                            if ecg_authenticate.feature_method == 'fiducial':
                                f_method = 1
                            for each_segment in decomposed_data_for_each_segment[0]:
                                temp_result = ecg_authenticate.step6_matching(
                                    each_user_template[user], each_segment[f_method], threshold=(
                                        threshold_template[user] * threshold_offset), plot=False)
                                if temp_result:
                                    result += 1

                            # result = ecg_authenticate.authenticate_user(file_name_list,each_user_template,3-1, threshold=2.48, qrs_method=qrs_method, is_wfdb=True, plot=False, which_filter=which_filter)
                            # print("---------------------------------------------------------------------------------------------------")
                            pass_val = result / \
                                decomposed_data_for_each_segment[0].__len__()
                            t_i = 0
                            for pass_threshold in np.arange(0, 1, 1):
                                if pass_val >= pass_threshold:
                                    if(i == j):
                                        # print("User is authenticated correctly = " + str(i))
                                        correct_user[t_i] += 1
                                    else:
                                        # print("User is authenticated incorrectly = " + str(j) + " as = " + str(i))
                                        false_user[t_i] += 1
                                t_i += 1
                            # ret_result.append([pass_threshold,correct_user,false_user)
                    # print(ret_result)

                for res in range(threshold_iterate):
                    accuracy = (correct_user[res] / all_user) * 100
                    false_accept_rate = (
                        false_user[res] / (all_user * all_user)) * 100
                    # print("Threshold = " + str(threshold_offset))
                    # print("Accuracy = " + str(accuracy))
                    # print("False Accept Rate = " + str(false_accept_rate))
                    accuracy_list = [res, accuracy, false_accept_rate]
                    ret.append(accuracy_list)
                return ret

            elif ("fantasia" in which_db) or ("mitdb" in which_db) or (which_db == 'cebsdb') or (which_db == 'ecg_ptb') or (which_db == 'CYBHi') or (which_db == 'ecg_bg'):
                ret = []
                unused_user = 0
                # file_name = 'ecg-id-database-1.0.0/Person_{0}/rec_1'
                threshold_iterate = 1
                step = 0.1
                start_thresh = 1.0
                end_thresh = start_thresh + ((step * threshold_iterate) - step)
                correct_user = [0] * threshold_iterate
                false_user = [0] * threshold_iterate
                ret_result = []
                file_name = ''
                file_name_list_all = []
                if which_db == 'ecg_ptb':
                    file_name = 'src/data/physionet.org/files/ptbdb/1.0.0/'
                    file_name_list_all = self.__get_ecg_ptb_file_list(
                        file_name, all_user)
                    all_user = file_name_list_all.__len__()
                elif which_db == 'CYBHi' or which_db == 'ecg_bg':
                    file_name = 'src/data/converted/{}/data/*.dat'.format(
                        which_db)
                    file_name_list_all = self.__get_fantasia_file_list(
                        file_name)
                else:
                    file_name = 'src/data/physionet.org/files/{0}/1.0.0/*.dat'.format(
                        which_db)
                    file_name_list_all = self.__get_fantasia_file_list(
                        file_name)
                start_t = 1.0
                end_t = 1.09
                # if ("mitdb" in which_db):
                #     start_t = 1.2
                #     end_t = 1.29

                for j in range(1, all_user + 1):
                    i = -1
                    for temp_name in file_name_list_all:
                        if i >= all_user:
                            break
                        i += 1
                        file_name_list = []
                        file_name_list.append(temp_name)
                        user = j - 1
                        # print("---------------------------------------------------------------------------------------------------")
                        # thresh = (threshold_template[user]*1.1)
                        thresh = 10
                        decomposed_data_for_each_segment = ecg_authenticate.authenticate_user(
                            file_name_list,
                            each_user_template,
                            user,
                            qrs_method=qrs_method,
                            is_wfdb=True,
                            plot=False,
                            threshold=thresh,
                            which_filter=which_filter,
                            is_auth=False)

                        for threshold_offset in np.arange(start_t, end_t, 0.1):
                            # for threshold_offset in
                            # np.arange(start_thresh,end_thresh,step):
                            result = 0
                            f_method = 0
                            skipped_segment = 0
                            if ecg_authenticate.feature_method == 'fiducial':
                                f_method = 1
                            if decomposed_data_for_each_segment[0].__len__(
                            ) == 0:
                                continue
                            for each_segment in decomposed_data_for_each_segment[0]:
                                if ecg_authenticate.feature_method == 'fiducial':
                                    if (each_segment.__len__() < 2) or (each_user_template[user].__len__() == 0) or (
                                            threshold_template[user] == np.NaN):
                                        skipped_segment += 1
                                        continue
                                else:
                                    if (each_user_template[user][0].__len__() == 0) or (
                                            threshold_template[user] == np.NaN):
                                        skipped_segment += 1
                                        continue
                                temp_result = ecg_authenticate.step6_matching(
                                    each_user_template[user], each_segment[f_method], threshold=(
                                        threshold_template[user] * threshold_offset), plot=False)
                                if temp_result:
                                    result += 1

                            # result = ecg_authenticate.authenticate_user(file_name_list,each_user_template,3-1, threshold=2.48, qrs_method=qrs_method, is_wfdb=True, plot=False, which_filter=which_filter)
                            # print("---------------------------------------------------------------------------------------------------")
                            denominator_pass = (
                                decomposed_data_for_each_segment[0].__len__() - skipped_segment)
                            if denominator_pass > 0:
                                pass_val = result / denominator_pass
                            else:
                                pass_val = 0
                            t_i = 0
                            for pass_threshold in np.arange(0, 1, 1):
                                if pass_val > pass_threshold:
                                    if(i == user):
                                        # print("User is authenticated correctly = " + str(i))
                                        correct_user[t_i] += 1
                                    else:
                                        # print("User is authenticated incorrectly = " + str(j) + " as = " + str(i))
                                        false_user[t_i] += 1
                                        if t_i > 0:
                                            print("d")
                                t_i += 1
                            # ret_result.append([pass_threshold,correct_user,false_user)
                    # print(ret_result)

                for res in range(threshold_iterate):
                    accuracy = (correct_user[res] / all_user) * 100
                    false_accept_rate = (
                        false_user[res] / ((all_user * all_user))) * 100
                    # print("Threshold = " + str(threshold_offset))
                    # print("Accuracy = " + str(accuracy))
                    # print("False Accept Rate = " + str(false_accept_rate))
                    accuracy_list = [res, accuracy, false_accept_rate]
                    ret.append(accuracy_list)
                return ret
                # pass




def main():
    # ecg_authenticate = ecg_authentication(template_method='template_5_set')
    discrete_wavelet_type = 'db4'
    ecg_authenticate = ecg_authentication(
        template_method='template_avg',
        discrete_wavelet_type=discrete_wavelet_type,
        continuous_wavelet='gaus1',
        continuous_wt_scale=50,
        feature_method='non-fiducial',
        segmentation_method="window")

    ecg_framework_object = ecg_framework()

    users = 2

    ##########################################################################
    #
    #  ENROLL USER FROM FRAMEWORK
    #  ecg_authenticate = ecg_authenticator object
    #  which_db = "ecg_id" or "ecg_id_csv" or "qt_db" fantasia, cebsdb, ecg_ptb, 'CYBHi', 'ecg_bg
    #  qrs_method = "wfdb" or "pantompkins"
    #  no_of_users = number of user you want to enroll
    #  which_filer = "bandpass" or "filfit" or "FIR" or "kalman"
    #  segmentation_method = "window" or RR
    #
    ##########################################################################
    # print("Enrolling")
    qrs = 'pantompkins'
    filter = 'FIR'
    db = 'ecg_id'

    each_user_template = ecg_framework_object.enroll_user_framework(
        ecg_authenticate=ecg_authenticate,
        which_db=db,
        qrs_method=qrs,
        no_of_users=users,
        which_filter=filter
    )
    ##########################################################################
    #
    #  Update Templates
    #
    ##########################################################################

    # file_name = 'ecgid/4_ecg_filtered/person_03_rec_3_ecg.csv'
    # ecg_authenticate.step7_template_update(each_user_template,file_name,3-1)
    # print("Learn Threshold")
    threshold_list = ecg_framework_object.learn_threshold(
        ecg_authenticate=ecg_authenticate,
        each_user_template=each_user_template,
        qrs_method=qrs,
        which_db=db,
        which_filter=filter,
        all_user=users)

    ##########################################################################
    #
    #  Authenticate User
    #  ecg_authenticate = ecg_authenticator object
    #  each_user_template = template generated from enroll user
    #  which_db = "ecg_id" or "ecg_id_csv" or "qt_db"
    #  qrs_method = "wfdb" or "pantompkins"
    #  which_filer = "bandpass" or "filfit" or "FIR" or "kalman"
    #  all_user = 1 -> authenticate with one user
    #           > 1 = users -> authenticate with all users -> user numbers
    #
    ##########################################################################
    # print("Authenticate")
    write_file_name = 'src/Result/' + db + '_' + str(users) + '_' + qrs + '_' + filter + '_' + discrete_wavelet_type + \
        '_' + ecg_authenticate.feature_method + '_' + ecg_authenticate.segmentation_method + '.csv'
    dump_op = open(write_file_name, "w")

    result = ecg_framework_object.authenticate_user_framework(
        ecg_authenticate=ecg_authenticate,
        each_user_template=each_user_template,
        which_db=db,
        qrs_method=qrs,
        which_filter=filter,
        threshold_template=threshold_list,
        all_user=users)
    for each_result in result:
        print(str(each_result[0]) + "," +
              str(each_result[1]) + "," + str(each_result[2]))
        dump_op.write(str(each_result[0]) +
                      "," +
                      str(each_result[1]) +
                      "," +
                      str(each_result[2]))
        dump_op.write('\n')

    # result = ecg_framework_object.authenticate_user_framework(ecg_authenticate=ecg_authenticate,each_user_template=each_user_template
    #                                         , which_db="ecg_id", qrs_method='wfdb',which_filter="bandpass"
    #                                         , threshold_template=threshold_list, all_user=1, which_user=4)

    # if result:
    #     print("User Authenticated\n")

    # print(result)


if __name__ == "__main__":
    main()