import mne as mne
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter
import numpy as np
import pandas as pd
from sklearn import decomposition


def butter_bandpass(low_cut, high_cut, fs, order):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, low_cut, high_cut, fs, order):
    """
    Defino un filtro Butterworth de 6to orden que a partir de Scipy.
    Filtro entre 1 y 35 Hz contains a zero-phase 6-th order Butterworth band-pass filter as provided by SciPy.

    :param data: the data to be filtered as a NumPy array of shape (n_samples, n_channels),
    :param low_cut: defining the desired frequency band in Hz
    :param high_cut: defining the desired frequency band in Hz
    :param fs: the EEG sampling rate in Hz.
    :param order:
    :return: Numpy Array filtered.
    """
    b, a = butter_bandpass(low_cut, high_cut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_eeg_data_butter_bandpass_filter(eeg_data, low_cut=2, high_cut=20, order=6, s_rate=256):
    """
    :param eeg_data: EEG Data with all channels.
    :param low_cut: Desired frequency band in Hz.
    :param high_cut: Desired frequency band in Hz.
    :param order:
    :param s_rate: Sampling rate.
    :return: Numpy Array with EEG Data filtered with butter bandpass filter.
    """

    EEGData_filtered = np.zeros_like(eeg_data)
    for i, temp_canal in enumerate(eeg_data):
        EEGData_filtered[i, :] = butter_bandpass_filter(temp_canal, low_cut, high_cut, s_rate, order)

    return EEGData_filtered


def plot_eeg_data(ax, eeg_data, ch_names=None, s_rate=256, start=1, end=4):
    ch = eeg_data.shape[0]
    x_ticks = np.arange(start * s_rate, (end + 1) * s_rate, s_rate)
    # fig, ax = plt.subplots(ncols=1, figsize=(16, 8))
    # fig.suptitle('Series temporales (uV)')
    y_ticks = []

    for c in np.arange(ch):
        temp = eeg_data[c, start * s_rate:end * s_rate]
        d_min = np.min(temp)
        d_max = np.max(temp)
        v_medio = np.mean([d_min, d_max]) + 30 * c
        y_ticks.append(v_medio)
        ax.plot(np.arange(start * s_rate, end * s_rate), v_medio * np.ones_like(temp) + temp, 'k')

    ax.set_xlim([start * s_rate, end * s_rate])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(np.arange(start, end + 1))
    ax.set_yticks(y_ticks)
    if ch_names:
        ax.set_yticklabels(ch_names)
    ax.set_ylabel('channels')
    ax.set_xlabel('Time (s)')
    return ax


def get_egg_data(filename):
    """
    Get EEG Data dataframe, sampling rate, channels, and samples.
    :param filename: filename path
    :return: EEGData dataframe with the data, sampling rate, channels, samples.
    """
    datos = pd.read_csv(filename, sep=',', header=None)
    EEGData = datos.to_numpy()
    s_rate = 128
    ch = EEGData.shape[0]
    samples = EEGData.shape[1]

    return EEGData, s_rate, ch, samples


class Mont1020:
    def __init__(self):
        self.mont1020 = mne.channels.make_standard_montage('standard_1020')
        self.kept_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
                              'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'FC1', 'FC2',
                              'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6', 'AFz', 'Fpz', 'POz']

    def get_ind(self):
        return [i for (i, channel) in enumerate(self.mont1020.ch_names) if channel in self.kept_channels]

    def get_ch_names(self):
        ind = self.get_ind()
        return [self.mont1020.ch_names[x] for x in ind]

    def get_ch_info(self):
        ind = self.get_ind()
        kept_channel_info = [self.mont1020.dig[x + 3] for x in ind]
        return kept_channel_info

    def get_info_eeg(self, s_freq=128, ch_types='eeg'):
        ch_names = self.get_ch_names()
        info_eeg = mne.create_info(ch_names=ch_names, sfreq=s_freq,
                                   ch_types=ch_types).set_montage(self.mont1020)

        return info_eeg


def plot_topology(eeg_data, info_eeg, sujeto, title, s_rate=256, start=1, end=4):
    fig, ax = plt.subplots(figsize=(8, 4),
                           gridspec_kw=dict(top=0.9),
                           sharex=True,
                           sharey=True)

    im, cm = mne.viz.plot_topomap(eeg_data[:, start * s_rate:end * s_rate].mean(axis=1),
                                  info_eeg,
                                  vmin=-0.05,
                                  vmax=0.3,
                                  cmap='coolwarm',
                                  contours=0,
                                  show=True)

    ax.title.set_text(title)
    ax_x_start = 0.95
    ax_x_width = 0.04
    ax_y_start = 0.1
    ax_y_height = 0.9
    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    fig.savefig(f'../images/grafica_topology_sujeto_{sujeto}.png')


def get_pca_eeg_data(eeg_data, n_components=3):
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(eeg_data)
    pcs = pca.transform(eeg_data)
    var = pca.explained_variance_ratio_

    return pcs, var


def plot_pca_eeg_data(pca, var, info_eeg, sujeto, title=''):
    fig2, ax = plt.subplots(ncols=pca.shape[1], figsize=(10, 3), gridspec_kw=dict(top=0.9),
                            sharex=True, sharey=True)
    for p in range(pca.shape[1]):
        mne.viz.plot_topomap(pca[:, p],
                             info_eeg,
                             cmap='coolwarm', contours=0,
                             axes=ax[p], show=False)
        ax[p].set_title('var:' + str(round(var[p] * 100, 2)))
    fig2.suptitle(title, fontsize=16, y=1.1)
    fig2.savefig(f'../images/grafico_pca_sujeto_{sujeto}')


def locmax(x):
    dx = np.diff(x)  # discrete 1st derivative
    zc = np.diff(np.sign(dx))  # zero-crossings of dx
    m = 1 + np.where(zc == -2)[0]  # indices of local max.
    return m


def calculo_gfp(data, srate, do_plot):
    gfp = np.std(data, axis=0)
    gfp_peaks = locmax(gfp)
    gfp_values = gfp[gfp_peaks]
    gfp2 = np.sum(gfp_values ** 2)  # normalizing constant in GEV
    n_gfp = gfp_peaks.shape[0]
    if do_plot == 'yes':
        marcasx = np.arange(3) * srate
        xlab = ['0', '1', '2']
        plt.plot(np.arange(len(gfp)), gfp, c='k')
        plt.scatter(gfp_peaks, gfp_values, c='r')
        plt.xlim(0, srate * 3)
        plt.xticks(marcasx, xlab)
        plt.xlabel('Tiempo (s)')
        plt.ylabel('GFP')
        plt.show()
    return gfp, gfp_peaks, gfp_values, gfp2, n_gfp


# k means modificado
def kmeans3(gfp_maps, gfp_eval, n_maps,
            n_runs=10, maxerr=1e-6, maxiter=500):
    # los inputs son los mapas, el valor del gfp en los picos,
    # la cantidad de clusters, la cantidad de corridas,
    # error máximo, número máximo de iteraciones
    V = gfp_maps.T
    n_gfp = V.shape[0]
    n_ch = V.shape[1]
    sumV2 = np.sum(V ** 2)

    # Guarda resultados de cada corrida
    cv_list = []  # cross-validation criterion for each k-means run
    maps_list = []  # microstate maps for each k-means run
    L_list = []  # microstate label sequence for each k-means run

    for run in range(n_runs):
        # initialize random cluster centroids
        rndi = np.random.permutation(n_gfp)[:n_maps]
        maps = V[rndi, :]
        # normalize row-wise (across EEG channels)
        maps /= np.sqrt(np.sum(maps ** 2, axis=1, keepdims=True))
        # initialize
        n_iter = 0
        var0 = 1.0
        var1 = 0.0
        # convergence criterion: variance estimate (step 6)
        while ((np.abs((var0 - var1) / var0) > maxerr) & (n_iter < maxiter)):
            # (step 3) microstate sequence (= current cluster assignment)
            C = np.dot(V, maps.T)
            C /= (n_ch * np.outer(gfp_eval, np.std(maps, axis=1)))
            L = np.argmax(C ** 2, axis=1)
            # (step 4)
            for k in range(n_maps):
                Vt = V[L == k, :]
                # (step 4a)
                Sk = np.dot(Vt.T, Vt)
                # (step 4b)
                evals, evecs = np.linalg.eig(Sk)
                v = evecs[:, np.argmax(np.abs(evals))]
                v = v.real
                maps[k, :] = v / np.sqrt(np.sum(v ** 2))
                # (step 5)
                var1 = var0
                var0 = sumV2 - np.sum(np.sum(maps[L, :] * V, axis=1) ** 2)
                var0 /= (n_gfp * (n_ch - 1))
                n_iter += 1
                if (n_iter > maxiter):
                    print((f"\tK-means run {run + 1:d}/{n_runs:d} did NOT converge "
                           f"after {maxiter:d} iterations."))
            # CROSS-VALIDATION criterion for this run (step 8)
        cv = var0 * (n_ch - 1) ** 2 / (n_ch - n_maps - 1.) ** 2
        cv_list = np.append(cv_list, cv)
        maps_list.append(maps)
        L_list.append(L)

    # select best run. Lo elige en función del validación cruzada
    k_opt = np.argmin(cv_list)
    maps = maps_list[k_opt]
    L = L_list[k_opt]
    cv = cv_list[k_opt]
    return maps, L, cv
