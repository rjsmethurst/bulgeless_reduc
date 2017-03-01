import numpy as N
import matplotlib.pyplot as P
import pandas as pd
#import pyfits as F
from astropy.io import fits
#from prefig import Prefig

import os, glob

from astropy.cosmology import FlatLambdaCDM

from astropy import units as un
from astropy.table import Table, vstack, Column

import math
#import linmix

#from gauss_BDS.py import gauss, lorentz, model, model_lorentzian, get_all_params, get_oiii_profile


fitsfiledir = 'combined_sdss_results/best_oiii_fits/'

os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'


P.rc('figure', facecolor='none', edgecolor='none', autolayout=True)
P.rc('path', simplify=True)
P.rc('text', usetex=True)
P.rc('font', family='serif')
P.rc('axes', labelsize='large', facecolor='none', linewidth=0.7, color_cycle = ['k', 'r', 'g', 'b', 'c', 'm', 'y'])
P.rc('xtick', labelsize='medium')
P.rc('ytick', labelsize='medium')
P.rc('lines', markersize=4, linewidth=1, markeredgewidth=0.2)
P.rc('legend', numpoints=1, frameon=False, handletextpad=0.3, scatterpoints=1, handlelength=2, handleheight=0.1)
P.rc('savefig', facecolor='none', edgecolor='none', frameon='False')

params =   {'font.size' : 11,
            'xtick.major.size': 8,
            'ytick.major.size': 8,
            'xtick.minor.size': 3,
            'ytick.minor.size': 3,
            }
P.rcParams.update(params)

cosmo = FlatLambdaCDM(H0=71.0, Om0 = 0.26)
H0 = cosmo.H0.value

msol = 1.989E30 * (un.kg) # mass of sun in kg
lsol = 3.846E33 * (un.erg/un.s) # luminosity of sun in erg/s
mpc = 3.08567758E16 * 1E6 * un.m # mpc in m
c = 299792.458 * (un.km/un.s) #speed of light in km/s


################################################################################
##################################           ###################################
###############################    FUNCTIONS    ################################
##################################           ###################################
################################################################################


def gauss(a, u, s, wave):
    #return a * N.exp(- (((wave - u)**2) /(s**2)))
    return a * N.exp(- (((wave - u)**2) /(2.*s**2)))



def lorentz(a, u, s, wave):
    return a * s**2 / ((wave - u)**2 + s**2)

#
# Get the FWHM of a given emission line
#
# Note: there are shortcuts to this for specific functions, but
#  this way is the most generalized and will work for even a complex
#  line profile (so long as it's always concave downwards, ie not double-peaked)
#
def calc_fwhm(wave, emission):
    max = N.max(emission)
    i = N.argmax(emission)
    hm = max/2
    idx = N.abs(emission-hm)
    fidx = N.argmin(idx[:i])
    sidx = N.argmin(idx[i:]) + i
    fwhm = wave[sidx] - wave[fidx]
    return fwhm

# for converting sigma or FWHM into km/s
def ang_to_kms(x, wavelength):
    # I believe this is analogous to the z ~ v/c approximation and so works great for narrowish lines
    # but starts to become inaccurate for very broad lines?
    #fwhm_v = (fwhm/6562.8) * c  #fwhm in km/s
    #Err_fwhm_v = (Err_fwhm/6562.8) * c
    # it's only different from below by a small factor, but nevertheless
    return c * N.log((x    /wavelength) + 1.)


# from Shen et al. (2011), using a relation between Ha line luminosity and 5100-A continuum from Greene & Ho (2005)
# note the constants are slightly different from Greene & Ho (2005) and this causes
# an upward shift in masses of about 0.5 dex, which based on comparison with masses
# of Shen et al. (who compared with and ultimately used the Vestergaard & Peterson
# Hbeta relation) this is the appropriate relation.
#
# lum should be in erg/s units, fwhm_v should be in km/s units
def Halpha_MBH(lum, Err_lum, fwhm_v, Err_fwhm_v):
    lmbh = 0.379 + 0.43*(N.log10(lum/(un.erg/un.s)) - 42.) + 2.1*(N.log10(fwhm_v/(un.km/un.s)))
    # GH07
    #mbh = 2E6 * ((lum/(1E42*un.erg/un.s))**0.455) * ((fwhm_v/(1E3*un.km/un.s))**2.06)
    # This part is still from GH07; Shen et al. didn't publish error bars, just
    # an offset (0.08 dex) and dispersion (0.18 dex) from VP06
    Err_mbh = 10.**lmbh * ( (0.55/3)**2 + (0.45*Err_lum/lum).value**2 + (0.03*N.log(lum.value/1E42))**2 + (2.06*Err_fwhm_v/fwhm_v).value**2 + (0.06*N.log(fwhm_v.value/1E3))**2 )**(0.5)

    return 10.**lmbh, Err_mbh


#

# gonna do a little bootstrappin'
# with asymmetric error bars 'cause that's how I roll
# if asymmetric, errors are a tuple/list of (lo, hi)
# if symmetric, just pass 1 value
#
# timings:
# nsamp = 1000
# 1000 loops, best of 3: 1.46 ms per loop
#
# nsamp = 10000
# 100 loops, best of 3: 3.36 ms per loop
#
# nsamp=100000
# 10 loops, best of 3: 23.9 ms per loop

def bootstrap_mbh_errors(lum, fwhm_v, err_lum, err_fwhm_v, nsamp=10000):

    # if the error bars are asymmetric, great. If not, fine.
    try:
        dlum_lo = err_lum[0]
        dlum_hi = err_lum[1]
        Err_lum = 0.5*(dlum_lo.value + dlum_hi.value) * un.erg/un.s
    except:
        dlum_lo = dlum_hi = err_lum
        Err_lum = err_lum

    try:
        dfwhm_v_lo = err_fwhm_v[0]
        dfwhm_v_hi = err_fwhm_v[1]
        Err_fwhm_v = 0.5*(dfwhm_v_lo.value + dfwhm_v_hi.value) * un.km/un.s
    except:
        dfwhm_v_lo = dfwhm_v_hi = err_fwhm_v

        Err_lum = N.mean(err_lum.value)
        Err_fwhm_v = err_fwhm_v



    #print(err_lum)
    #print(dlum_lo)
    #print(dlum_hi)

    dlum  = N.zeros(nsamp)
    dfwhm = N.zeros(nsamp)
    # draw from the normal distribution
    fdlum  = N.random.randn(nsamp)
    fdfwhm = N.random.randn(nsamp)

    # figure out which indices draw from the -sigma side & which from +s
    dlum_is_pos  = fdlum  > 0.00
    dfwhm_is_pos = fdfwhm > 0.00
    dlum_is_neg  = fdlum  < 0.00
    dfwhm_is_neg = fdfwhm < 0.00

    dlum[dlum_is_pos]   =  fdlum[dlum_is_pos]  * dlum_hi    #*(un.erg/un.s)
    dlum[dlum_is_neg]   =  fdlum[dlum_is_neg]  * dlum_lo    #*(un.erg/un.s)
    dfwhm[dfwhm_is_pos] = fdfwhm[dfwhm_is_pos] * dfwhm_v_hi #*(un.km/un.s)
    dfwhm[dfwhm_is_neg] = fdfwhm[dfwhm_is_neg] * dfwhm_v_lo #*(un.km/un.s)

    lum_boot    = (lum.value    + dlum)*(un.erg/un.s)
    fwhm_v_boot = (fwhm_v.value + dfwhm)*(un.km/un.s)

    #fwhm_v_boot = ang_to_kms(fwhm_boot, 6562.8)

    mbh_boot, Err_mbh_boot = Halpha_MBH(lum_boot, Err_lum*N.ones(len(lum_boot)), fwhm_v_boot, Err_fwhm_v*N.ones(len(fwhm_v_boot)))
    mlo, mbh_b, mhi = N.percentile(mbh_boot, [16, 50, 84])
    dmbh_b_lo = mbh_b - mlo
    dmbh_b_hi = mhi - mbh_b

    return mbh_b, [mlo, mhi], [dmbh_b_lo, dmbh_b_hi]


#
# Use the Halpha luminosity+FWHM relation to compute BH masses and uncertainties
#
def bhmass(ld, Err_ld, fwhm, Err_fwhm, dfwhm_lo, dfwhm_hi, wave, fit, fitp, fitm):

    fwhm_v     = ang_to_kms(fwhm, 6562.8)
    Err_fwhm_v = ang_to_kms(Err_fwhm, 6562.8)
    dfwhm_v_lo = ang_to_kms(dfwhm_lo, 6562.8)
    dfwhm_v_hi = ang_to_kms(dfwhm_hi, 6562.8)

    print('fwhm = ', fwhm_v)

    #lum = flux*1E-17*6562.8*(4*N.pi*(d**2)) # luminosity in erg/s but flux in erg/cm2/s/A - so integrate over all wavelengths - i.e. multiply by the wavelength
    # don't comment this next line back in unless you're only using Gaussians (the error function only = the integral of the Normal function)
    #total_flux = (-0.5 * N.sqrt(N.pi) * wave[-1] * a * s * math.erf((u-wave[-1])/s)) - (-0.5 * N.sqrt(N.pi) * wave[0] * a * s * math.erf((u-wave[0])/s))  #integrate over the entire gaussian rather than taking the maximum of the Halpha flux which includes the narrow component

    # integrate to get total flux as well as what we'll use to get the plus and minus errors
    total_flux = N.abs(N.trapz(fit, wave))  * (un.erg/(un.cm**2)/un.s)
    tfp        = N.abs(N.trapz(fitp, wave)) * (un.erg/(un.cm**2)/un.s)
    tfm        = N.abs(N.trapz(fitm, wave)) * (un.erg/(un.cm**2)/un.s)

    # luminosity in erg/s but flux in erg/cm2/s - so need d in cm2
    lum     = total_flux*1E-17*(4*N.pi*(ld.to(un.cm)**2))
    dlum_lo =        tfp*1E-17*(4*N.pi*(ld.to(un.cm)**2)) - lum
    dlum_hi =  lum - tfm*1E-17*(4*N.pi*(ld.to(un.cm)**2))

    dtotalflux_lo = N.abs(total_flux-tfp).value *(un.erg/un.s/un.cm**2)
    dtotalflux_hi = N.abs(total_flux-tfm).value *(un.erg/un.s/un.cm**2)

    Err_total_flux =  N.mean([N.abs(total_flux-tfp).value, N.abs(total_flux-tfm).value])*(un.erg/un.s/un.cm**2)
    Err_lum = lum * ( (Err_total_flux/total_flux).value**2 + (2*Err_ld/ld.value)**2 )**(0.5)

    print('luminosity %.2f +/- %.2f' % (N.log10(lum.value), Err_lum/(lum*N.log(10))))

    # get the black hole masses
    mbh, Err_mbh = Halpha_MBH(lum, Err_lum, fwhm_v, Err_fwhm_v)

    # get estimates using bootstrapping and considering asymmetric error bars
    mbh_b, mbh_b_lohi, dmbh_b = bootstrap_mbh_errors(lum, fwhm_v, (dlum_lo, dlum_hi), (dfwhm_v_lo, dfwhm_v_hi), nsamp=100000)


    print(' ------------------------------------\n', N.log10(mbh.value), ' +- ', Err_mbh/(mbh*N.log(10)))
    print('or: %.2e + %.2e - %.2e\n ------------------------------------' % (mbh_b, dmbh_b[1], dmbh_b[0]))

    return (lum/lsol), (Err_lum/lsol), (dlum_lo/lsol), (dlum_hi/lsol), N.log10(mbh), Err_mbh/(mbh*N.log(10)), mbh_b, mbh_b_lohi, dmbh_b







def get_fittype(filename):
    if filename.find('lorentz') >= 0:
        return 'lorentzian'
    else:
        return 'gaussian'






def get_spectral_props(filename):
    bf = fits.open(filename)
    hdr = bf[0].header
    #bf[0].data - spectrum
    #bf[1].data - fit, total (continuum + emission-line)
    #bf[2].data - emission-line-only fit
    #bf[3].data - Gandalf residual (?)
    #bf[4].data - good pixels, I think

    #flux = bf[2].data
    #continuum = bf[1].data - bf[2].data
    #spectrum_nocont = bf[0].data - continuum
    #flux = spectrum_nocont

    # Description from SDSS DR7 data model
    #
    # Line measurements and redshift determinations, as well as the spectrum, for a single object, summing over all of its exposures through a given mapped plate. Each file contains the following HDUs:
    # Primary HDU image: spectrum, continuum-subtracted spectrum, noise in spectrum, mask array.
    # HDU 1 and 2: Line detections and measurements. Under most circumstances, the line measurements in HDU 2 should be used.
    # HDU 3: Emission-line redshift measurements
    # HDU 4: Cross-correlation redshift measurement
    # HDU 5: Line index measurements
    # HDU 6: Mask and per-pixel resolution
    #hdulist = fits.open(gfitlocdir+source[n].replace("_fits.fits", ".fit"))
    #fluxerr = hdulist[0].data[2]
    #hdulist.close()

    z = hdr['Z']

    # define observed and rest-frame wavelength arrays
    lam = hdr['CRVAL1'] + hdr['CD1_1']*(N.arange(hdr['NAXIS1'] -  hdr['CRPIX1'] + 1))
    wave = (10**lam)/(1+z)

    bf.close()

    return wave, z






################################################################################
#################################              #################################
#############################   END OF FUNCTIONS   #############################
#################################              #################################
################################################################################



# I want to use the fits from emcee to compute MBH based on those parameters
# but I want this to be general and take parameters from whatever numpy save files
# are in a particular directory, not *just* 1 per source or just 1 survey or whatever.

npydir = '/Users/vrooje/Documents/Astro/bbcint_paper/data/spectra/combined_sdss_spectra/emcee_gauss_fits_4_components_Brooke_oiii_width/'

# then again there are also .npy files containing the walkers so just look for
# best-fit save files
npyfiles = glob.glob(npydir+'*best_fit.npy')

file_prop = pd.DataFrame(npyfiles)
file_prop.columns = ['emcee_fitfile_npy']
nfiles = len(file_prop)

# whether the fit is lorentz or gaussian (we can reconstruct this with the
# filename list but let's make it explicit)
file_prop['specfile']    = ['' for q in file_prop.emcee_fitfile_npy]
file_prop['which_fit']   = [get_fittype(q) for q in file_prop.emcee_fitfile_npy]
# redshift
file_prop['z']   = N.zeros(nfiles)

# now set up the columns we know we'll want to break out of the numpy files
# and the columns that we'll want to save related to the BH mass
#
# recall the best-fit parameters output by emcee are
# theta = [broad Halpha a, broad Ha mu, broad Ha s, narrow Ha a, narrow Ha mu, narrow [NII] a]
# where a is amplitude, mu is central wavelength, s is sigma (for a Gaussian) or the width parameter (for a Lorentzian)
# using these parameters and the previously-fit narrow-line profiles we can construct
# the entire best-fit model spectrum
# (though for the BH mass we just need a, u, s for broad Halpha)
#
# a, u, s, a1, u1, an1 = theta
file_prop['broad_Ha_amplitude']   = N.zeros(nfiles)
file_prop['broad_Ha_wavelength']  = N.zeros(nfiles)
file_prop['broad_Ha_sig']         = N.zeros(nfiles)
file_prop['narrow_Ha_amplitude']  = N.zeros(nfiles)
file_prop['narrow_Ha_wavelength'] = N.zeros(nfiles)
file_prop['narrow_NII_amplitude'] = N.zeros(nfiles)
# then there are the 16th & 84th percentile values on these
file_prop['broad_Ha_amplitude_lo']   = N.zeros(nfiles)
file_prop['broad_Ha_wavelength_lo']  = N.zeros(nfiles)
file_prop['broad_Ha_sig_lo']         = N.zeros(nfiles)
file_prop['narrow_Ha_amplitude_lo']  = N.zeros(nfiles)
file_prop['narrow_Ha_wavelength_lo'] = N.zeros(nfiles)
file_prop['narrow_NII_amplitude_lo'] = N.zeros(nfiles)
file_prop['broad_Ha_amplitude_hi']   = N.zeros(nfiles)
file_prop['broad_Ha_wavelength_hi']  = N.zeros(nfiles)
file_prop['broad_Ha_sig_hi']         = N.zeros(nfiles)
file_prop['narrow_Ha_amplitude_hi']  = N.zeros(nfiles)
file_prop['narrow_Ha_wavelength_hi'] = N.zeros(nfiles)
file_prop['narrow_NII_amplitude_hi'] = N.zeros(nfiles)

# we will want to output broad Halpha line luminosity, FWHM, MBH, and errors.
file_prop['Ha_lum']        = N.zeros(nfiles)
file_prop['dHa_lum_lo']    = N.zeros(nfiles)
file_prop['dHa_lum_hi']    = N.zeros(nfiles)
file_prop['Ha_FWHM']       = N.zeros(nfiles)
file_prop['dHa_FWHM_lo']   = N.zeros(nfiles)
file_prop['dHa_FWHM_hi']   = N.zeros(nfiles)
file_prop['MBH']           = N.zeros(nfiles)
file_prop['dMBH']          = N.zeros(nfiles)
file_prop['lMBH_boot']     = N.zeros(nfiles)
file_prop['dlMBH_boot_lo'] = N.zeros(nfiles)
file_prop['dlMBH_boot_hi'] = N.zeros(nfiles)

# I wish we could just do
# file_prop['{all the columns}'.split()] = N.zeros((nfiles, ncols))
# but that throws an error
# maybe someday


for i, fullrow in enumerate(file_prop.iterrows()):
    if i < 1e6: # set to a real value if you're just testing this
        i_row = fullrow[0]
        row   = fullrow[1]

        is_row = file_prop.index == i_row

        thefile = row['emcee_fitfile_npy']
        fittype = row['which_fit']

        # extract the core of the filename for FITS/profile reading purposes
        # and decide which function to use to create the broad-line profile
        if fittype == 'lorentzian':
            thesuffix = '_fits.fits_lorentz_best_fit.npy'
            profile_func = lorentz
        else:
            thesuffix = '_fits.fits_best_fit.npy'
            profile_func = gauss

        filebase = thefile.split("/")[-1].replace(thesuffix, '')

        # I know, I know, I'm adding back part of what I just removed
        # but what if I later want to read the original spectrum or something?
        fitsfile = fitsfiledir+filebase+"_fits.fits"

        # load the numpy save file containing best-fit parameters
        bestfit = N.load(thefile)

        # need to construct the rest-frame wavelength array
        # the information for which is in the headers of the FITS file
        wave, z = get_spectral_props(fitsfile)

        ld = cosmo.luminosity_distance(z)
        Err_ld = N.mean([(cosmo.luminosity_distance(z+0.001*z)-ld).value, (ld-cosmo.luminosity_distance(z+0.001*z)).value])


        # broad Ha {a, u, s} are [0:2][] of this, with [][0] being the best fit,
        # [][1] being the upper value, and [][2] being the lower value
        broad  = profile_func(bestfit[0][0],                 bestfit[1][0],                 bestfit[2][0], wave)
        broadp = profile_func(bestfit[0][0] + bestfit[0][1], bestfit[1][0] + bestfit[1][1], bestfit[2][0] + bestfit[2][1], wave)
        broadm = profile_func(bestfit[0][0] - bestfit[0][2], bestfit[1][0] - bestfit[1][2], bestfit[2][0] - bestfit[2][2], wave)

        # get the FWHMs
        int_fwhm = calc_fwhm(wave, broad)
        fwhmp    = calc_fwhm(wave, broadp)
        fwhmm    = calc_fwhm(wave, broadm)

        # get asymmetric and symmetric errors
        dfwhm_lo = N.abs(int_fwhm-fwhmm)
        dfwhm_hi = N.abs(fwhmp-int_fwhm)
        Err_int_fwhm = N.mean([dfwhm_lo, dfwhm_hi])

        fwhm_v         = ang_to_kms(int_fwhm,     6562.8)
        dfwhm_v_lo     = ang_to_kms(dfwhm_lo,     6562.8)
        dfwhm_v_hi     = ang_to_kms(dfwhm_hi,     6562.8)
        Err_int_fwhm_v = ang_to_kms(Err_int_fwhm, 6562.8)

        int_lum, Err_int_lum, dlum_lo, dlum_hi, int_mbh, Err_int_mbh, mbh_b, mbh_b_lohi, dmbh_b = bhmass(ld, Err_ld, int_fwhm, Err_int_fwhm, dfwhm_lo, dfwhm_hi, wave, broad, broadp, broadm)

        # get errors in log space
        lmbh_b      = N.log10(mbh_b)
        lmbh_b_lohi = N.log10(mbh_b_lohi)
        dlmbh_b = N.abs(lmbh_b - lmbh_b_lohi)


        # row is a series, file_prop is a dataframe
        # the small differences in how you handle these continues to irk me
        # e.g. we could do file_prop.columns or row.axes[0]
        # but NOT NOT NOT the other way around. Because obviously.
        # this ought to work but it keeps filling the entire row of file_prop with NaN
        #
        #file_prop.loc[is_row, row.axes[0]] = row[row.axes[0]]
        #
        # which means we have to do each value manually
        # which is much slower. Lovely
        file_prop.loc[is_row, 'specfile']                = filebase+".fit"
        file_prop.loc[is_row, 'z']                       = z
        file_prop.loc[is_row, 'broad_Ha_amplitude']      = bestfit[0][0]
        file_prop.loc[is_row, 'broad_Ha_wavelength']     = bestfit[1][0]
        file_prop.loc[is_row, 'broad_Ha_sig']            = bestfit[2][0]
        file_prop.loc[is_row, 'narrow_Ha_amplitude']     = bestfit[3][0]
        file_prop.loc[is_row, 'narrow_Ha_wavelength']    = bestfit[4][0]
        file_prop.loc[is_row, 'narrow_NII_amplitude']    = bestfit[5][0]
        file_prop.loc[is_row, 'broad_Ha_amplitude_lo']   = bestfit[0][0] - bestfit[0][2]
        file_prop.loc[is_row, 'broad_Ha_wavelength_lo']  = bestfit[1][0] - bestfit[1][2]
        file_prop.loc[is_row, 'broad_Ha_sig_lo']         = bestfit[2][0] - bestfit[2][2]
        file_prop.loc[is_row, 'narrow_Ha_amplitude_lo']  = bestfit[3][0] - bestfit[3][2]
        file_prop.loc[is_row, 'narrow_Ha_wavelength_lo'] = bestfit[4][0] - bestfit[4][2]
        file_prop.loc[is_row, 'narrow_NII_amplitude_lo'] = bestfit[5][0] - bestfit[5][2]
        file_prop.loc[is_row, 'broad_Ha_amplitude_hi']   = bestfit[0][0] + bestfit[0][1]
        file_prop.loc[is_row, 'broad_Ha_wavelength_hi']  = bestfit[1][0] + bestfit[1][1]
        file_prop.loc[is_row, 'broad_Ha_sig_hi']         = bestfit[2][0] + bestfit[2][1]
        file_prop.loc[is_row, 'narrow_Ha_amplitude_hi']  = bestfit[3][0] + bestfit[3][1]
        file_prop.loc[is_row, 'narrow_Ha_wavelength_hi'] = bestfit[4][0] + bestfit[4][1]
        file_prop.loc[is_row, 'narrow_NII_amplitude_hi'] = bestfit[5][0] + bestfit[5][1]
        file_prop.loc[is_row, 'Ha_lum']                  = int_lum.value
        file_prop.loc[is_row, 'dHa_lum_lo']              = dlum_lo.value
        file_prop.loc[is_row, 'dHa_lum_hi']              = dlum_hi.value
        file_prop.loc[is_row, 'Ha_FWHM']                 = fwhm_v.value
        file_prop.loc[is_row, 'dHa_FWHM_lo']             = dfwhm_v_lo.value
        file_prop.loc[is_row, 'dHa_FWHM_hi']             = dfwhm_v_hi.value
        file_prop.loc[is_row, 'MBH']                     = int_mbh.value
        file_prop.loc[is_row, 'dMBH']                    = Err_int_mbh.value
        file_prop.loc[is_row, 'lMBH_boot']               = lmbh_b
        file_prop.loc[is_row, 'dlMBH_boot_lo']           = dlmbh_b[0]
        file_prop.loc[is_row, 'dlMBH_boot_hi']           = dlmbh_b[1]


        # these are AGN properties but we don't need them to compute the BH mass
        #int_lbol = 8 * (4*N.pi*(ld.to(un.cm).value)**2) * 1E-23 * (299792458/12E-6) * 31.674 * 10**(-ints['w3mag']/2.5)
        #Err_int_lbol = int_lbol * ( (8*4*N.pi*Err_ld/ld.value)**2 + ( ((1E-23 * (299792458/12E-6) * 31.674)/-2.5) * N.log(10) * ints['w3sigm'] )**2 )**(0.5)

outfile = fitsfiledir+'best_fit_properties_emcee_all_in_dir.csv'
file_prop.to_csv(outfile)
print("\nDone. Printed outputs to %s\n" % outfile)


#
