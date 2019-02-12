import re

import cdo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import xarray as xr
from scipy.signal import detrend, periodogram, tukey
from scipy.stats import chi2, pearsonr


def Sef2014_Fig3_ACF(control,
                     varnamelist,
                     area='Tropical_Pacific',
                     period='ym'):
    """Plot persistence as Autocorrelation function (ACF) from control simulation.

    Reference
    ---------
    - Séférian, Roland, Laurent Bopp, Marion Gehlen, Didier Swingedouw,
    Juliette Mignot, Eric Guilyardi, and Jérôme Servonnat. “Multiyear
    Predictability of Tropical Marine Productivity.” Proceedings of the National
    Academy of Sciences 111, no. 32 (August 12, 2014): 11646–51.
    https://doi.org/10/f6cgs3.

    Parameters
    ----------
    control : Dataset with year dimension
        Input data
    varnamelist : list
        variables to be included
    area : str
        area of interest
    period : str
        period of interest

    """
    plt.figure(figsize=(6, 4))
    df = control.sel(
        area=area, period=period)[varnamelist].to_dataframe()[varnamelist]
    cmap = sb.color_palette("husl", len(varnamelist))
    for i, var in enumerate(df.columns):
        pd.plotting.autocorrelation_plot(df[var], label=var, color=cmap[i])
    plt.xlim([1, 20])
    plt.ylim([-.5, 1])
    plt.ylabel('Persistence (ACF)')
    plt.xlabel('Lag [year]')
    plt.legend(ncol=2, loc='lower center')
    plt.title((' ').join(('Autocorrelation function', area, period)))


def power_spectrum_markov(control,
                          varname,
                          unit=''):
    fig, ax = plt.subplots(figsize=(10, 4))
    s = control.to_dataframe()[varname]
    P, power_spectrum, markov, low_ci, high_ci = create_power_spectrum(s)
    plot_power_spectrum_markov(
        P,
        power_spectrum,
        markov,
        low_ci,
        high_ci,
        color='k',
        ax=ax,
        unit=unit)


def plot_power_spectrum_markov(P,
                               power_spectrum,
                               markov,
                               low_ci,
                               high_ci,
                               ax=None,
                               legend=False,
                               plot_ci=True,
                               **kwargs):
    ax.plot(P, power_spectrum, **kwargs)
    ax.plot(
        P,
        markov,  # label='theoretical Markov spectrum',
        alpha=.5,
        ls='--')
    if plot_ci:
        ax.plot(P, low_ci, c='gray', alpha=.5, linestyle='--')
        ax.plot(P, high_ci, c='gray', alpha=.5, ls='--')
    ax.set_xlabel('Period [yr]')
    ax.set_ylabel('Power [(' + 'unit' + ')$^2$]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([2, 200])
    if legend:
        ax.legend()
    ax.set_title('Power spectrum')


def _get_pcs(anom,
             neofs=15,
             pcscaling=0,
             curv=True):
    from eofs.xarray import Eof

    def get_anom(df):
        return df - df.mean('time')

    coslat = np.cos(np.deg2rad(anom.coords['lat'].values))
    wgts = np.sqrt(coslat)[..., np.newaxis]

    if curv:
        wgts = None
    else:
        coslat = np.cos(np.deg2rad(anom.coords['lat'].values))
        wgts = np.sqrt(coslat)[..., np.newaxis]

    solver = Eof(anom, weights=wgts)
    eofs = solver.eofsAsCorrelation(neofs=neofs)
    # eofcov = solver.eofsAsCovariance(neofs=neofs)
    pcs = solver.pcs(npcs=neofs, pcscaling=pcscaling)
    eofs['mode'] = np.arange(1, eofs.mode.size+1)
    pcs['mode'] = np.arange(1, pcs.mode.size+1)
    return eofs, pcs


def _get_max_peak_period(P, power_spectrum, high_ci):
    significant_peaks = power_spectrum.where(power_spectrum > high_ci)
    max_period = significant_peaks.argmax()
    return P[max_period]


def Sef2013_Fig4_power_spectrum_pcs(control3d,
                                    neofs=5,
                                    plot_eofs=True,
                                    curv=True,
                                    palette='Set2',
                                    print_peak=True
                                    ):
    eofs, pcs = _get_pcs(
        control3d, pcscaling=1, neofs=neofs)
    cmap = sb.color_palette(palette, pcs.mode.size)
    if plot_eofs:
        eofs.plot(col='mode', robust=True, yincrease=not curv)
        plt.show()
        pcs.to_dataframe().unstack().plot(colors=cmap, figsize=(10, 4))

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, mode in enumerate(pcs.mode):
        P, power_spectrum, markov, low_ci, high_ci = create_power_spectrum(
            pcs.sel(mode=mode).to_series())
        plot_power_spectrum_markov(
            P,
            power_spectrum,
            markov,
            low_ci,
            high_ci,
            legend=True,
            plot_ci=False,
            ax=ax,
            color=cmap[i],
            label='PC' + str(int(mode)))
        x = _get_max_peak_period(P, power_spectrum, high_ci)
        if print_peak:
            print('PC'+str(int(mode)), 'max peak at',
                  '{0:.2f}'.format(x), 'years.')
        ax.axvline(
            x=x,
            c=cmap[i],
            ls=':')


def corr_plot_2var(control,
                   varx='fgco2',
                   vary='po4os',
                   area='90S-35S',
                   period='ym'):
    """Plot the correlation between two variables."""
    g = sb.jointplot(
        x=varx,
        y=vary,
        data=control.sel(area=area, period=period).to_dataframe(),
        kind='reg')
    g.annotate(pearsonr)


def corrfunc(x, y, **kws):
    """Corr for corr_pairgrid."""
    r, p = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate(
        "r = {:.2f}, p = {:.5f}".format(r, p),
        xy=(.1, .9),
        xycoords=ax.transAxes)


def corr_pairgrid(control,
                  varnamelist=['tos', 'sos', 'AMO'],
                  area='90S-35S',
                  period='ym'):
    """Plot pairgrid of variables from varnamelist."""
    g = sb.PairGrid(
        control.sel(area=area, period=period).to_dataframe()[varnamelist],
        palette=["red"])
    g.map_upper(plt.scatter, s=10)
    g.map_diag(sb.distplot, kde=False)
    g.map_lower(sb.kdeplot, cmap="Blues_d")
    g.map_lower(corrfunc)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle((' ').join(('Correlations', area, period)))


def show_wavelets(control,
                  unit='',
                  cxmax=None):
    s = control.to_series()
    if cxmax is None:
        cxmax = s.var() * 8
    title = ' '  # (' ').join((varname, area, period))
    label = ' '  # varname + ' ' + area
    import pycwt as wavelet
    from pycwt.helpers import find
    from scipy.signal import detrend
    dt = 1
    N = s.size
    t = s.index
    dat_notrend = detrend(s.values)
    std = dat_notrend.std()  # Standard deviation
    var = std**2  # Variance
    dat_norm = dat_notrend / std  # Normalized dataset
    mother = wavelet.Morlet(6)
    s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 years = 6 months
    dj = 1 / 12  # Twelve sub-octaves per octaves
    J = 7 / dj  # Seven powers of two with dj sub-octaves
    # Lag-1 autocorrelation for red noise
    alpha, _, _ = wavelet.ar1(dat_notrend)
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
        dat_norm, dt, dj, s0, J, mother)
    iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std
    power = (np.abs(wave))**2
    fft_power = np.abs(fft)**2
    period = 1 / freqs
    signif, fft_theor = wavelet.significance(
        1.0, dt, scales, 0, alpha, significance_level=0.95, wavelet=mother)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95
    glbl_power = power.mean(axis=1)
    dof = N - scales  # Correction for padding at edges
    glbl_signif, tmp = wavelet.significance(
        var,
        dt,
        scales,
        1,
        alpha,
        significance_level=0.95,
        dof=dof,
        wavelet=mother)
    sel = find((period >= 2) & (period < 8))
    Cdelta = mother.cdelta
    scale_avg = (scales * np.ones((N, 1))).transpose()
    # As in Torrence and Compo (1998) equation 24
    scale_avg = power / scale_avg
    scale_avg = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
    scale_avg_signif, tmp = wavelet.significance(
        var,
        dt,
        scales,
        2,
        alpha,
        significance_level=0.95,
        dof=[scales[sel[0]], scales[sel[-1]]],
        wavelet=mother)
    figprops = dict(figsize=(11, 8), dpi=72)
    fig = plt.figure(**figprops)
    ax = plt.axes([0.1, 0.75, 0.65, 0.2])
    ax.plot(t, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
    ax.plot(t, dat_notrend, 'k', linewidth=1.5)
    ax.set_title('a) {}'.format(title))
    ax.set_ylabel(r'{} [{}]'.format(label, unit))

    bx = plt.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    bx.contourf(
        t,
        np.log2(period),
        np.log2(power),
        np.log2(levels),
        extend='both',
        cmap=plt.cm.viridis)
    extent = [t.min(), t.max(), 0, max(period)]
    bx.contour(
        t,
        np.log2(period),
        sig95, [-99, 1],
        colors='k',
        linewidths=2,
        extent=extent)
    bx.fill(
        np.concatenate([t, t[-1:] + dt, t[-1:] + dt, t[:1] - dt, t[:1] - dt]),
        np.concatenate([
            np.log2(coi), [1e-9],
            np.log2(period[-1:]),
            np.log2(period[-1:]), [1e-9]
        ]),
        'k',
        alpha=0.3,
        hatch='x')
    bx.set_title('b) {} Wavelet Power Spectrum ({})'.format(
        label, mother.name))
    bx.set_ylabel('Period (years)')
    #
    Yticks = 2**np.arange(
        np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    bx.set_yticks(np.log2(Yticks))
    bx.set_yticklabels(Yticks)
    cx = plt.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
    cx.plot(glbl_signif, np.log2(period), 'k--')
    cx.plot(var * fft_theor, np.log2(period), '--', color='#cccccc')
    cx.plot(
        var * fft_power,
        np.log2(1. / fftfreqs),
        '-',
        color='#cccccc',
        linewidth=1.)
    cx.plot(var * glbl_power, np.log2(period), 'k-', linewidth=1.5)
    cx.set_title('c) Global Wavelet Spectrum')
    cx.set_xlabel(r'Power [({})^2]'.format(unit))
    #cx.set_xlim([0, glbl_power.max() + var])
    cx.set_xlim([0, cxmax])
    cx.set_ylim(np.log2([period.min(), period.max()]))
    cx.set_yticks(np.log2(Yticks))
    cx.set_yticklabels(Yticks)
    plt.setp(cx.get_yticklabels(), visible=False)

    dx = plt.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
    dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
    dx.plot(t, scale_avg, 'k-', linewidth=1.5)
    dx.set_title('d) {}--{} year scale-averaged power'.format(2, 8))
    dx.set_xlabel('Time (year)')
    dx.set_ylabel(r'Average variance [{}]'.format(unit))
    ax.set_xlim([t.min(), t.max()])


def remap(da):
    if not isinstance(da, xr.core.dataset.Dataset):
        da = da.to_dataset()
    remap = cdo.remapbil(
        'r360x180', input=da, returnXDataset=True, options='-P 8')
    return remap


def plot_Hovmoeller(control,
                    varname,
                    lats=[-35, -30, -25, -20, -15],
                    mean_dim='lat',
                    latstep=5,
                    remap='cdo',
                    **kwargs):
    if remap is 'cdo':
        remap = cdo.remapbil(
            'r360x180',
            input=control.to_dataset(),
            returnXArray=varname,
            options='-P 8')
    elif remap is 'xesmf':
        remap = 0
    fig, ax = plt.subplots(
        ncols=len(lats), sharey=True, figsize=(5 * len(lats), 10))
    for i, slat in enumerate(lats):
        nlat = slat + latstep
        if slat == lats[-1]:
            colorbar = True
        else:
            colorbar = False
        remap.sel(
            lon=slice(150, 280), lat=slice(slat, nlat)).mean(mean_dim).plot(
                ax=ax[i],
                cmap='RdBu_r',
                levels=11,
                add_colorbar=colorbar,
                **kwargs)
        ax[i].set_title(str(slat) + '-' + str(nlat))
    plt.tight_layout()


# s='GR15_lon_-150--120_lat_-10--35.mask.nc'
def _area_str_2_lons_lats(area):
    lons = re.search('_lon_(.*)_lat', area).group(1)
    lats = re.search('_lat_(.*).[mw]', area).group(1)
    if lats[0] is '-':
        lats = lats[1:]
    if lons[0] is '-':
        lons = lons[1:]
    if '--' in lons:
        lons = lons.replace('--', '-')
        lonl, lonr = lons.split('-')
        lonr = '-' + lonr
    else:
        lonl, lonr = lons.split('-')
    if '--' in lats:
        lats = lats.replace('--', '-')
        latl, latr = lats.split('-')
        latr = '-' + latr
    else:
        latl, latr = lats.split('-')
    orilon = re.search('_lon_(.*)_lat', area).group(1)
    orilon
    if orilon[0] is '-':
        lonl = '-' + lonl
    orilat = re.search('_lat_(.*).[mw]', area).group(1)
    if orilat[0] is '-':
        latl = '-' + latl
    return int(lonl), int(lonr), int(latl), int(latr)


def _lons_lats_2_area(lonl, lonr, latl, latr, grid='GR15', mask_weight='mask'):
    return '_'.join(
        (grid, 'lon', str(lonl))) + '-' + str(lonr) + '_' + 'lat' + '_' + str(
            latl) + '-' + str(latr) + '.' + mask_weight + '.nc'


def _taper(x, p):
    """
    Description needed here.
    """
    window = tukey(len(x), p)
    y = x * window
    return y


def create_power_spectrum(s, pct=0.1, pLow=0.05):
    """
    Create power spectrum with CI for a given pd.series.

    Reference
    ---------
    - /ncl-6.4.0-gccsys/lib/ncarg/nclscripts/csm/shea_util.ncl

    Parameters
    ----------
    s : pd.series
        input time series
    pct : float (default 0.10)
        percent of the time series to be tapered. (0 <= pct <= 1). If pct = 0,
        no tapering will be done. If pct = 1, the whole series is tapered.
        Tapering should always be done.
    pLow : float (default 0.05)
        significance interval for markov red-noise spectrum

    Returns
    -------
    p : np.ndarray
        period
    Pxx_den : np.ndarray
        power spectrum
    markov : np.ndarray
        theoretical markov red noise spectrum
    low_ci : np.ndarray
        lower confidence interval
    high_ci : np.ndarray
        upper confidence interval
    """
    # A value of 0.10 is common (tapering should always be done).
    jave = 1  # smoothing ### DOESNT WORK HERE FOR VALUES OTHER THAN 1 !!!
    tapcf = 0.5 * (128 - 93 * pct) / (8 - 5 * pct)**2
    wgts = np.linspace(1., 1., jave)
    sdof = 2 / (tapcf * np.sum(wgts**2))
    pHigh = 1 - pLow
    data = s - s.mean()
    # detrend
    data = detrend(data)
    data = _taper(data, pct)
    # periodigram
    timestep = 1
    frequency, power_spectrum = periodogram(data, timestep)
    Period = 1 / frequency
    power_spectrum_smoothed = pd.Series(power_spectrum).rolling(jave, 1).mean()
    # markov theo red noise spectrum
    twopi = 2. * np.pi
    r = s.autocorr()
    temp = r * 2. * np.cos(twopi * frequency)  # vector
    mkov = 1. / (1 + r**2 - temp)  # Markov model
    sum1 = np.sum(mkov)
    sum2 = np.sum(power_spectrum_smoothed)
    scale = sum2 / sum1
    xLow = chi2.ppf(pLow, sdof) / sdof
    xHigh = chi2.ppf(pHigh, sdof) / sdof
    # output
    markov = mkov * scale  # theor Markov spectrum
    low_ci = markov * xLow  # confidence
    high_ci = markov * xHigh  # interval
    return Period, power_spectrum_smoothed, markov, low_ci, high_ci


def create_composites(anomaly_field, timeseries, threshold=1, dim='time'):
    index_comp = xr.full_like(timeseries, 'none', dtype='U4')
    index_comp[timeseries >= threshold] = 'pos'
    index_comp[timeseries <= -threshold] = 'neg'
    composite = anomaly_field.groupby(index_comp.rename('index')).mean(dim=dim)
    return composite


def standardize(ds, dim='time'):
    return (ds-ds.mean(dim))/ds.std(dim)


def composite_analysis(field, timeseries, threshold=1, plot=True, **plot_kwargs):
    index = standardize(timeseries)
    field = field - field.mean('time')
    composite = create_composites(field, index, threshold=threshold)
    if plot:
        composite.sel(index='pos').plot(**plot_kwargs)
        plt.show()
        composite.sel(index='neg').plot(**plot_kwargs)
    else:
        return composite
