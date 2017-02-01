"""File to use emcee to fit a double gaussian with a narrow and broad peak to emission data from GANDALF."""

import numpy as N
import pylab as P
from astropy.io import fits, ascii
#import pyfits as F
import emcee
import glob
import time
from scipy.optimize import minimize, curve_fit
from astropy.table import Table
import os, warnings

# ugh actually this prints a fuckload of warnings
# could add N.seterr(under='ignore') afterward but then I believe that's already the default
#N.seterr(all='warn')
c = 299792.458 # km/s

ha_wave    = 6562.8
hb_wave    = 4861.32

nii_wave   = 6583.34
niib_wave  = 6547.96
n2_dfac    = 0.34 # amplitude ratio between doublet lines ([NII] 6548)/([NII] 6583)

oiii_wave  = 5006.77
oiiib_wave = 4958.83
o3_dfac    = 0.35 # ([OIII] 4959)/([OIII] 5007)

#

def walker_plot(samples, nwalkers, limit, ndim, source, dir):
    # theta:
    # broad ha a, broad ha mu, broad ha s, narrow ha a, narrow ha mu, narrow nii a

    s = samples.reshape(nwalkers, -1, ndim)
    s = s[:,:limit, :]
    fig = P.figure(figsize=(8,18))
    ax1 = P.subplot(ndim,1,1)
    ax2 = P.subplot(ndim,1,2)
    ax3 = P.subplot(ndim,1,3)
    ax4 = P.subplot(ndim,1,4)
    ax5 = P.subplot(ndim,1,5)
    ax6 = P.subplot(ndim,1,6)
    ax1.plot(s[:,:,0].T, 'k', rasterized=True)
    ax2.plot(s[:,:,1].T, 'k', rasterized=True)
    ax3.plot(s[:,:,2].T, 'k', rasterized=True)
    ax4.plot(s[:,:,3].T, 'k', rasterized=True)
    ax5.plot(s[:,:,4].T, 'k', rasterized=True)
    ax6.plot(s[:,:,5].T, 'k', rasterized=True)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.set_ylabel(r'amp B $H\alpha$')
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.set_ylabel(r'wave B $H\alpha$')
    ax3.tick_params(axis='x', labelbottom='off')
    ax3.set_ylabel(r'stdev B $H\alpha$')
    ax4.tick_params(axis='x', labelbottom='off')
    ax4.set_ylabel(r'amp N $H\alpha$')
    ax5.tick_params(axis='x', labelbottom='off')
    ax5.set_ylabel(r'wave N $H\alpha$')
    ax6.set_ylabel(r'amp N $H\alpha$')
    P.subplots_adjust(hspace=0.05)
    #P.tight_layout()
    fig.savefig(dir+'./walkers_steps_'+source+'.png')
    P.close()


def lnlikeoiii(theta, wave, flux, fluxerr):
    a, u, s = theta
    pred_f = gauss(a, u, s, wave)
    #print pred_f
    chi =  -0.5*N.log(2*N.pi*fluxerr**2)-0.5*((flux-pred_f)**2/fluxerr**2)
    #print N.sum(chi)
    return N.sum(chi)

def lnprioroiii(theta, wave, flux):
    a, u, s = theta
    if a > 0 and s > 0:
        return 0.0
    else:
        return -N.inf

def lnproboiii(theta, wave, flux, fluxerr):
    lp = lnprioroiii(theta, wave, flux)
        #print lp
    if not N.isfinite(lp):
        return -N.inf
    #print lp + lnlike(theta, wave, flux, fluxerr)
    return lp + lnlikeoiii(theta, wave, flux, fluxerr)




# returns everything you need to make narrow line profiles for Halpha & [NII] given theta
# (ignores broad-line parmeters a, u, s so this works whether the broad profile is fitted
#  as a gaussian or a lorentzian)
def get_all_params(theta, oiii_profile):
    # theta:
    # broad ha a, broad ha mu, broad ha s, narrow ha a, narrow ha mu, narrow nii a

    #a, u, s, a1, u1, s1, an1, sn1, an2, sn2 = theta
    a, u, s, a1, u1, an1 = theta

    # it may seem counterintuitive to designate the higher-wavelength [NIII] line as
    # line "1" but it's the primary line of the doublet
    un1 = u1 + (nii_wave - ha_wave)
    un2 = u1 - (ha_wave  - niib_wave)
    n2_dfac = 0.34 # amplitude ratio between doublet lines ([NII] 6548)/([NII] 6583)

    # use theta and oiii_profile to construct the narrow profiles as copies of the [OIII] profile but at the amplitudes of each line

    # specifically: given  a1,     an1,        u1
    # we need s1, s2, s3,  a2, a3, an1b, an1c, u2, u3, un1b, un1c, un2b, un2c

    # yes, the [NII] widths should technically be slightly different as u1 and un1 are
    # slightly different but this is a tiny effect & I'm choosing to neglect it here
    s1 = u1*oiii_profile['widthfac'][0]
    s2 = u1*oiii_profile['widthfac'][1]
    s3 = u1*oiii_profile['widthfac'][2]

    # amplitudes of any additional line components have a fixed ratio between them
    # all given relative to the 1st-component amplitude for that particular line
    a2   = a1  * oiii_profile['arat'][1]
    a3   = a1  * oiii_profile['arat'][2]

    an1b = an1 * oiii_profile['arat'][1]
    an1c = an1 * oiii_profile['arat'][2]

    # wavelengths of additional line components are also fixed relative to 1st component
    u2   = u1  + oiii_profile['dwave'][1]
    u3   = u1  + oiii_profile['dwave'][2]

    un1b = un1 + oiii_profile['dwave'][1]
    un1c = un1 + oiii_profile['dwave'][2]

    un2b = un2 + oiii_profile['dwave'][1]
    un2c = un2 + oiii_profile['dwave'][2]

    return un1, un2, n2_dfac, s1, s2, s3, a2, a3, an1b, an1c, u2, u3, un1b, un1c, un2b, un2c


# difference between calling model() and a version of it that only returns the summed profile:
# In [2340]: timeit themodel, haa, ha, niia, niib = model(start, oiii_profile, wave)
# 1000 loops, best of 3: 367 microsec per loop
#
# In [2339]: timeit themodel = get_profile(start, oiii_profile, wave)
# 1000 loops, best of 3: 362 microsec per loop
#
# So, 5 microsec or about a 1.4% savings
# Given this is going to run for hours on the whole dataset, that's a savings of about
# a few minutes - aka not enough to offset the annoyance of having two functions that
# do very nearly exactly the same thing
#
# the default is to assume the broad-line model is Gaussian; if it's Lorentzian use model_lorentzian()
def model(theta, oiii_profile, wave):
    '''
    We have several components here: Broad Halpha, narrow Halpha, 2 x narrow [NII] the narrow lines should all have the same profiles (different amplitudes permitted) and their wavelengths relative to each other should be fixed.

    We've already fit the narrow-line profile to the [OIII] lines, and have those stored for each object. Sometimes they're complex; they can be built from up to 3 Gaussian profiles. But the shape of those profiles is assumed to be the same for all the narrow lines.

    So in order to fit the narrow Ha and [NII] doublet we'll need up to 9 different Gaussians - but their parameters aren't all free. In fact, the only things that can vary are 1 of the wavelengths (the others are all at a fixed dwave from that), the narrow Halpha total amplitude, and the [NII] 6583 total amplitude (the [NII] 6547 amplitude is fixed to 0.34 x A_Nii6583).

    The broad Halpha is likely to be at the same wavelength as the narrow Halpha, but I'm not sure that's always true (?), so its component (which we're keeping to Gaussian for now) has 3 additional parameters: wavelength, amplitude, sigma.

    That means that although we may have up to 9 Gaussians for the narrow lines and 1 Gaussian for the broad line, we really only have 6 free parameters (no matter how complex we let the narrow-line profile get).
    '''
    # theta:
    # broad ha a, broad ha mu, broad ha s, narrow ha a, narrow ha mu, narrow nii a

    #a, u, s, a1, u1, s1, an1, sn1, an2, sn2 = theta
    a, u, s, a1, u1, an1 = theta

    # get all the other amplitudes, wavelengths, sigmas we need for the other profiles
    # based on the a1, u1, an1 in theta and the given narrow-line profile (from [OIII])
    un1, un2, n2_dfac, s1, s2, s3, a2, a3, an1b, an1c, u2, u3, un1b, un1c, un2b, un2c = get_all_params(theta, oiii_profile)

    broad_ha        = gauss(a,           u,   s,  wave)
    narrow_ha       = gauss(a1,          u1,  s1, wave) + gauss(a2,           u2,   s2, wave) + gauss (a3,           u3,   s3, wave)
    narrow_nii_6583 = gauss(an1,         un1, s1, wave) + gauss(an1b,         un1b, s2, wave) + gauss (an1c,         un1c, s3, wave)
    narrow_nii_6545 = gauss(n2_dfac*an1, un2, s1, wave) + gauss(n2_dfac*an1b, un2b, s2, wave) + gauss (n2_dfac*an1c, un2c, s3, wave)

    # return the summed profile plus all the individual profiles
    return broad_ha + narrow_ha + narrow_nii_6583 + narrow_nii_6545, broad_ha, narrow_ha, narrow_nii_6583, narrow_nii_6545


# this is really bad programming but bear with me
oiii_profile=[]

'''

    So. modelonly() and modelonly_lorentzian() are both used by scipy.optimize to fit the Ha+[NII] profiles. In the same way as emcee uses model() and model_lorentzian(), they use the information about the shape of the narrow-line profiles as well as the set of free parameters in the fit to return the model that will be compared against the data. The key here is that the narrow-line profile changes from object to object, which means that even for the same set of fit parameters the model changes from object to object, and all that information is stored in the oiii_profile dict.

    Except, scipy.optimize can only take model functions with a specific set of arguments. The x-variable (wavelength for us) has to be the first argument, and then all the other fit parameters have to be individually passed as scalars. I can't pass a dict or a pointer to a dict or anything like that. (This is much more restrictive than emcee.) So unless I deconstruct oiii_profile and pass all those parameters individually, then hack them back into oiii_profile so I can call get_all_params() -- or possibly write a separate version of get_all_params() with a different set of inputs that does the same thing, which is even more gross -- I have to rely on the fact that oiii_profile is a global variable. (There is quite possibly another option, but I don't know it.)

    Doing this with global variables is frowned upon, or at least it is in other languages. So far in all my tests this is working just fine, but it could break anytime and it could be very difficult to track that bug. The first indicator that this is not the way Python is meant to work is that without the line above this comment that initializes oiii_profile as empty, the script will crash when I try to run it because when it tries to ingest the function below it goes "no wait you've called something you're not passing to me and that doesn't already exist; barf". That happens even if I never actually call the function modelonly() in the rest of the script.

    I was all set to spend a couple of shameful hours rewriting and testing this, but the fix isn't that much less awkward, and this currently works fine, so, eff it. It stays until it breaks (but bear in mind you should check to see if it's broken before trusting the scipy fits for anything).

'''

def modelonly(wave, a, u, s, a1, u1, an1):
    # theta:
    # broad ha a, broad ha mu, broad ha s, narrow ha a, narrow ha mu, narrow nii a

    #a, u, s, a1, u1, an1 = theta
    theta = [a, u, s, a1, u1, an1]

    un1, un2, n2_dfac, s1, s2, s3, a2, a3, an1b, an1c, u2, u3, un1b, un1c, un2b, un2c = get_all_params(theta, oiii_profile)

    broad_ha        = gauss(a,           u,   s,  wave)
    narrow_ha       = gauss(a1,          u1,  s1, wave) + gauss(a2,           u2,   s2, wave) + gauss (a3,           u3,   s3, wave)
    narrow_nii_6583 = gauss(an1,         un1, s1, wave) + gauss(an1b,         un1b, s2, wave) + gauss (an1c,         un1c, s3, wave)
    narrow_nii_6545 = gauss(n2_dfac*an1, un2, s1, wave) + gauss(n2_dfac*an1b, un2b, s2, wave) + gauss (n2_dfac*an1c, un2c, s3, wave)

    # return the summed profile plus all the individual profiles
    return broad_ha + narrow_ha + narrow_nii_6583 + narrow_nii_6545



def model_lorentzian(theta, oiii_profile, wave):
    # theta:
    # broad ha a, broad ha mu, broad ha s, narrow ha a, narrow ha mu, narrow nii a

    #a, u, s, a1, u1, s1, an1, sn1, an2, sn2 = theta
    a, u, s, a1, u1, an1 = theta

    # get all the other amplitudes, wavelengths, sigmas we need for the other profiles
    # based on the a1, u1, an1 in theta and the given narrow-line profile (from [OIII])
    un1, un2, n2_dfac, s1, s2, s3, a2, a3, an1b, an1c, u2, u3, un1b, un1c, un2b, un2c = get_all_params(theta, oiii_profile)

    # the next line is the only line that's different between model() and model_lorentzian()
    broad_ha      = lorentz(a,           u,   s,  wave)
    #
    narrow_ha       = gauss(a1,          u1,  s1, wave) + gauss(a2,           u2,   s2, wave) + gauss (a3,           u3,   s3, wave)
    narrow_nii_6583 = gauss(an1,         un1, s1, wave) + gauss(an1b,         un1b, s2, wave) + gauss (an1c,         un1c, s3, wave)
    narrow_nii_6545 = gauss(n2_dfac*an1, un2, s1, wave) + gauss(n2_dfac*an1b, un2b, s2, wave) + gauss (n2_dfac*an1c, un2c, s3, wave)

    # return the summed profile plus all the individual profiles
    return broad_ha + narrow_ha + narrow_nii_6583 + narrow_nii_6545, broad_ha, narrow_ha, narrow_nii_6583, narrow_nii_6545


def modelonly_lorentzian(wave, a, u, s, a1, u1, an1):
    # theta:
    # broad ha a, broad ha mu, broad ha s, narrow ha a, narrow ha mu, narrow nii a

    #a, u, s, a1, u1, an1 = theta
    theta = [a, u, s, a1, u1, an1]

    un1, un2, n2_dfac, s1, s2, s3, a2, a3, an1b, an1c, u2, u3, un1b, un1c, un2b, un2c = get_all_params(theta, oiii_profile)

    broad_ha      = lorentz(a,           u,   s,  wave)
    narrow_ha       = gauss(a1,          u1,  s1, wave) + gauss(a2,           u2,   s2, wave) + gauss (a3,           u3,   s3, wave)
    narrow_nii_6583 = gauss(an1,         un1, s1, wave) + gauss(an1b,         un1b, s2, wave) + gauss (an1c,         un1c, s3, wave)
    narrow_nii_6545 = gauss(n2_dfac*an1, un2, s1, wave) + gauss(n2_dfac*an1b, un2b, s2, wave) + gauss (n2_dfac*an1c, un2c, s3, wave)

    # return the summed profile plus all the individual profiles
    return broad_ha + narrow_ha + narrow_nii_6583 + narrow_nii_6545






def lnlike(theta, wave, oiii_profile, flux, fluxerr):
    # we only need pred_f from model here
    pred_f, ha_br, na_nr, niia_nr, niib_nr = model(theta, oiii_profile, wave)
    #print pred_f
    chi =  -0.5*N.log(2*N.pi*fluxerr**2)-0.5*((flux-pred_f)**2/fluxerr**2)
    #print N.sum(chi)
    return N.sum(chi)


def lnprior(theta, wave, flux):
    # theta:
    # broad ha a, broad ha mu, broad ha s, narrow ha a, narrow ha mu, narrow nii a

    a, u, s, a1, u1, an1 = theta

    if a > 0 and a1 > 0 and an1 >= 0 and u > 6555 and u < 6580 and u1 > 6555 and u1 < 6580 and N.abs(u-u1)<30. and s > 0 and s < 2000:
    	return 0.0
    else:
    	return -N.inf


def lnprob(theta, wave, flux, fluxerr, oiii_profile):
    lp = lnprior(theta, wave, flux)
    #print lp
    if not N.isfinite(lp):
        return -N.inf
    return lp + lnlike(theta, wave, oiii_profile, flux, fluxerr)







def lnlike_lorentz(theta, wave, oiii_profile, flux, fluxerr):
    # we only need pred_f from model here
    pred_f, ha_br, na_nr, niia_nr, niib_nr = model_lorentzian(theta, oiii_profile, wave)
    #print pred_f
    chi =  -0.5*N.log(2*N.pi*fluxerr**2)-0.5*((flux-pred_f)**2/fluxerr**2)
    #print N.sum(chi)
    return N.sum(chi)



def lnprob_lorentz(theta, wave, flux, fluxerr, oiii_profile):
    lp = lnprior(theta, wave, flux)
    #print lp
    if not N.isfinite(lp):
        return -N.inf
    return lp + lnlike_lorentz(theta, wave, oiii_profile, flux, fluxerr)





# okay so this doesn't matter much as we're interested in FWHM and you're
# measuring it below in a way independent of sigma as defined here, but
# unless s has been redefined to be sqrt(2)*sigma, the function below
# is missing a factor of 2 in the denominator of the exponent.
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




def get_oiii_profile(filename):
    # First load up the width of the narrow [OIII] component output by Brooke from GANDALF
    oiiip = Table.read(filename, format='ascii')

    # how many components does the narrow [OIII] profile actually have?
    n_oiiip = len(oiiip) # okay turns out I may not need to explicitly define this for later use
    print('Source %s has    %d    components in narrow [OIII]' % (source[n], n_oiiip))

    # Note: the table above should list components in order of strongest (well,
    # highest-amplitude) first, so we can use [0] as the denominator in flux ratio
    # [OIII] is composed of no more than 3 Gaussian components
    oiii_widths   = N.zeros(3)
    oiii_amps     = N.zeros(3)
    oiii_waves    = N.zeros(3)
    oiii_widthfac = N.zeros(3) + 1.0e-6

    oiii_widths[0:n_oiiip] = oiiip['sigma_Ang']
    oiii_amps[0:n_oiiip]   = oiiip['observed_A']
    oiii_waves[0:n_oiiip]  = oiiip['lambda_fit']

    # amplitude relative to the 0th component (to be fixed while fitting)
    # note non-existent components have a relative amplitude of 0
    oiii_arat  = oiii_amps/oiii_amps[0]
    # wavelength relative to the 0th component (to be fixed while fitting)
    # note non-existent components have a relative amplitude of ~ -5007
    oiii_dwave = oiii_waves - oiii_waves[0]


    # put this all in one dict so we can pass it through emcee more easily
    # (we need a bunch of this to create the model each time)
    oiii_profile = {}
    oiii_profile['n_comp'] = n_oiiip
    oiii_profile['width']  = oiii_widths
    oiii_profile['amp']    = oiii_amps
    oiii_profile['wave']   = oiii_waves
    oiii_profile['arat']   = oiii_arat
    oiii_profile['dwave']  = oiii_dwave

    # factor to multiply by wavelength to convert to width at a new wavelength
    # set the default value to > 0.0 so as not to cause div0 errors later
    # (not all the components will exist for all objects, but they'll also have
    #  amplitude = 0 in that case so the width here shouldn't matter)
    # (though if for some reason you see components with really low sigma in your
    #  best-fit model, something's gone wrong)
    # so we don't div0 in the next line, just worry about the non-zero profiles
    ok_comp = (oiii_profile['width'] > 1.0e-6) & (oiii_profile['wave'] > 1000.)

    oiii_widthfac[ok_comp] = oiii_profile['width'][ok_comp]/oiii_profile['wave'][ok_comp]
    oiii_profile['widthfac'] = oiii_widthfac

    return oiii_profile, n_oiiip






# use the narrow-line profile and Gandalf best-fit parameters to get initial guesses to feed into emcee
#
# note: the narrow-line profiles may be very different than what Gandalf settled on (for technical reasons
# too boring to get into but basically if we could have gotten Gandalf to use these profiles we wouldn't
# need to use emcee in the first place), so the initial guesses might actually be kind of crap.

def get_initial_guesses(oiii_profile, thesource):
    # oiii_profile['n_comp']   = n_oiiip
    # oiii_profile['width']    = oiii_widths
    # oiii_profile['amp']      = oiii_amps
    # oiii_profile['wave']     = oiii_waves
    # oiii_profile['arat']     = oiii_arat
    # oiii_profile['dwave']    = oiii_dwave
    # oiii_profile['widthfac'] = oiii_widthfac

    # the gandalf fits will give us initial guesses for Halpha and [NII] amplitudes
    haprofilefile = Haprofiledir+thesource.split('_fits.fits')[0]+'_linemeas.csv'
    # try this one first, but if it doesn't exist then try the version with/without "_bluecutoff"
    if not os.path.isfile(haprofilefile):
        haprofilefile_old = haprofilefile
        if thesource.endswith('_bluecutoff_fits.fits'):
            haprofilefile = Haprofiledir+thesource.split('_bluecutoff_fits.fits')[0]+'_linemeas.csv'
        else:
            haprofilefile = Haprofiledir+thesource.split('_fits.fits')[0]+'_bluecutoff_linemeas.csv'

    # now, if you can read a file, great. If not, go with some basic default initial guesses
    if os.path.isfile(haprofilefile):
        allp = ascii.read(haprofilefile)

        # (reverse) sort by amplitude so that the strongest lines are first in any sub-selection
        # For details on the [::-1] array-mirroring thing see http://stackoverflow.com/a/26984520
        allp[::-1].sort('observed_A')

        is_habr = allp['name'] == "Ha_br"
        is_ha   = allp['name'] == "Ha"
        is_nii  = allp['name'] == "[NII]"

        a_habr = allp[is_habr]['observed_A'][0]
        s_habr = allp[is_habr]['sigma_Ang'][0]

        a_ha   = allp[is_ha]['observed_A'][0]

        a_nii  = allp[is_nii]['observed_A'][0]

        # sanity-check these values & replace if the gandalf fit went weird
        if a_habr < 3. or a_habr > 1000.:
            a_habr = 0.5*oiii_profile['amp'][0]

        if s_habr < 1. or s_habr > 130.:
            s_habr = 10.0*ha_wave*oiii_profile['widthfac'][0]

        if a_ha < 3. or a_ha > 1000.:
            a_ha   = 0.8*oiii_profile['amp'][0]

        if a_nii < 3. or a_nii > 1000.:
            a_nii  = 0.6*oiii_profile['amp'][0]


    else:
        print("Unable to find H-alpha line profile information, using values based on [OIII]")
        a_habr = 0.5*oiii_profile['amp'][0]
        s_habr = 10.0*ha_wave*oiii_profile['widthfac'][0]
        a_ha   = 0.8*oiii_profile['amp'][0]
        a_nii  = 0.6*oiii_profile['amp'][0]

    # In order: broad_ha, narrow_ha, nii_6583, nii_6548 - with each of the last 3 having 3 profiles
    # each profile: [amplitude, wavelength, sigma]
    # assume broad Ha has half the amplitude of the primary [OIII] Gaussian and 10x its width

    # Halpha broad: [amplitude, wavelength, sigma]
    start = [a_habr, ha_wave, s_habr]

    # then the remaining free parameters are Narrow Halpha amplitude,
    #   Narrow Halpha wavelength, Narrow [NII]6583 amplitude
    # oiii_profile is sorted by descending amplitude so take the highest-amp profile here
    start.append(a_ha)
    start.append(ha_wave)
    start.append(a_nii)
    # defining the list and then appending 3 times isn't computationally efficient but we're only
    # calling this once per object so I'm not too concerned about it & this is more readable

    # start has the same pattern as the variable called theta later on:
    # broad ha a, broad ha mu, broad ha s, narrow ha a, narrow ha mu, narrow nii a

    return start



def sanity_check_parameters(theta, oiii_profile, thesource):
    start_default = get_initial_guesses(oiii_profile, thesource)

    # compare theta to the default starting parameters and replace anything pathological in theta
    a,   u,  s,  a1,  u1,  an1 = theta
    sa, su, ss, sa1, su1, san1 = start_default

    if (a < 1.) | (a > 1000.):
        a = sa

    if (abs(u - ha_wave) > 20.):
        u = su

    if (s < 0.1) | (s > 120.):
        s = ss

    if (a1 < 1.) | (a1 > 1000.):
        a1 = sa1

    if (abs(u1 - ha_wave) > 20.):
        u1 = su1

    if (an1 < 1.) | (an1 > 1000.):
        an1 = san1

    return [a, u, s, a1, u1, an1]




def plot_model_fit(flux, fluxerr, theta, results, ha_br, ha_nr, niia, niib, wave, outfile, thelabel=''):

    thewave     = theta[1]
    thefwhm     = calc_fwhm(wave, ha_br)
    #thefwhm_kms = c * (thefwhm/thewave)
    thefwhm_kms_sig = 2.355*c * N.log((theta[2]/thewave) + 1.)
    thefwhm_kms     =       c * N.log((thefwhm   /thewave) + 1.)



    P.figure(figsize=(8, 6))
    ax1 =P.subplot(111)


    # plot the uncertainties, then the spectrum over the top of them
    # generally the uncertainties are likely to be quite small, but this will plot the bounds just in case
    ax1.plot(wave, flux+fluxerr, c='k', alpha=0.3, linewidth=3)
    ax1.plot(wave, flux-fluxerr, c='k', alpha=0.3, linewidth=3)

    # spectrum
    ax1.plot(wave, flux, c='k', linewidth=2)

    # this next line is an approximation of a continuum-subtracted spectrum as it
    # assumes the continuum is constant at a value of the 1st pixel in the wavelength range specified
    #ax1.plot(wave, bf[0].data-bf[0].data[0], c='k', alpha=0.3, linewidth=2)

    # residual
    ax1.plot(wave, flux-results, c='k', alpha=0.3, linewidth=1)

    # best-fit
    ax1.plot(wave, results, c='r', linestyle='dashed', linewidth=2)

    # individual components
    ax1.plot(wave, ha_br, c='b')
    ax1.plot(wave, ha_nr, c='m')
    ax1.plot(wave, niia,  c='g')
    ax1.plot(wave, niib,  c='g')
    ax1.text(0.05, 0.9, r'FWHM = %3.2f km/s' % thefwhm_kms, transform=ax1.transAxes)
    ax1.text(0.95, 0.9, thelabel, horizontalalignment='right', transform=ax1.transAxes)
    P.savefig(outfile)

    P.close('all')




################################################################################
################################################################################
################################################################################
################################                ################################
############################          MAIN          ############################
################################                ################################
################################################################################
################################################################################
################################################################################






#ndim = 10
ndim = 6
nwalkers = 200
nsteps = 500
burnin = 800


#gfitlocdir = '/Users/becky/Projects/int_reduc/bdmass_fits_gandalf_bds/combined_sdss_spectra/'
#fitsavedir = '/Users/becky/Projects/int_reduc/bdmass_fits_gandalf_bds/combined_sdss_spectra/emcee_gauss_fits_4_components_Brooke_oiii_width/'
#profiledir = '/Users/becky/Projects/int_reduc/bdmass_fits_gandalf_bds/combined_sdss_spectra/oiii_line_components_profiles_tables/'

# testing and/or use on BDS' machine
#source = list(['spSpec-54241-2516-619_bluecutoff_fits.fits'])
sourcefile = '/Users/vrooje/Astro/bbcint_paper/data/spectra/combined_sdss_results/actually_fit_files_with_emlines_ebv_fe_nopath_oiiiemfile.txt'
thesources = ascii.read(sourcefile)
specfiles = [q+".fit" for q in thesources['specfilebase']]
source    = [q+"_fits.fits" for q in thesources['specfilebase']]

#
#thesource = 'spSpec-52368-0881-064_fits.fits' # it's crashing
#for n in range(len(source)):
#    if source[n] == thesource:
#        print(n,thesource)


#source = ['spSpec-52368-0881-064_fits.fits']
# if you crashed mid-run last time and want to start where you left off,
# set this to the correct index
#index_start = 0
index_start = 86

gfitlocdir   = 'combined_sdss_results/best_oiii_fits/'
fitsavedir   = 'combined_sdss_spectra/emcee_gauss_fits_4_components_Brooke_oiii_width/'
profiledir   = 'combined_sdss_results/best_oiii_fits/'
Haprofiledir = 'combined_sdss_results/best_ha_hb_fits/'

rerun_list=[]
rerun_list_lorentz=[]
for n in range(len(source)):
    # this adds the possibility of starting from the middle of the list
    if n >= index_start:
        bf = fits.open(gfitlocdir+source[n])
        hdr = bf[0].header
        #bf[0].data - spectrum
        #bf[1].data - fit, total (continuum + emission-line)
        #bf[2].data - emission-line-only fit
        #bf[3].data - Gandalf residual (?)
        #bf[4].data - good pixels, I think

        #flux = bf[2].data
        continuum = bf[1].data - bf[2].data
        spectrum_nocont = bf[0].data - continuum
        flux = spectrum_nocont


        # the original spectrum has an error axis - should probably use it instead
        fluxerr_old = N.sqrt(N.abs(flux))+N.random.random_sample(len(flux))*1E-30

        # Description from SDSS DR7 data model
        #
        # Line measurements and redshift determinations, as well as the spectrum, for a single object, summing over all of its exposures through a given mapped plate. Each file contains the following HDUs:
        # Primary HDU image: spectrum, continuum-subtracted spectrum, noise in spectrum, mask array.
        # HDU 1 and 2: Line detections and measurements. Under most circumstances, the line measurements in HDU 2 should be used.
        # HDU 3: Emission-line redshift measurements
        # HDU 4: Cross-correlation redshift measurement
        # HDU 5: Line index measurements
        # HDU 6: Mask and per-pixel resolution
        hdulist = fits.open(gfitlocdir+source[n].replace("_fits.fits", ".fit"))
        fluxerr = hdulist[0].data[2]
        hdulist.close()

        # define observed and rest-frame wavelength arrays
        lam = hdr['CRVAL1'] + hdr['CD1_1']*(N.arange(hdr['NAXIS1'] -  hdr['CRPIX1'] + 1))
        wave = (10**lam)/(1+hdr['Z'])
        # First fit to the oiii narrow component to get a limit on the width of the narrow component and a good guess for start point.
        # lim1 = N.searchsorted(wave, 4980)
        # lim2 = N.searchsorted(wave, 5050)
        # nll = lambda *args: -lnproboiii(*args)
        # result = minimize(nll, [1, 5004, 2], args=(wave[lim1:lim2], flux[lim1:lim2], fluxerr[lim1:lim2]))
        # oiii_width = result['x'][2]
        # oiii_amp = result['x'][0]
        # results = gauss(result['x'][0], result['x'][1], result['x'][2], wave)
        # P.figure()
        # ax1 =P.subplot(111)
        # ax1.plot(wave[lim1:lim2], flux[lim1:lim2], c='k', linewidth=2)
        # ax1.plot(wave[lim1:lim2], results[lim1:lim2], c='r', linestyle='dashed')
        # ax1.text(0.1, 0.9, r'[OIII] $\sigma$ = %3.2f' % oiii_width, transform=ax1.transAxes)
        # P.savefig(fitsavedir+source[n]+'_oiii_model_fit.png')

        cont_res = N.median(spectrum_nocont[(wave > 6350.) & (wave < 6425.)])

        if cont_res > 0.:
            print("\n\n####################### Continuum of %.2f for %s needs adjusting!\n\n\n" % (cont_res, source[n]))
            flux = spectrum_nocont - cont_res


        # get the details of the [OIII] profile (which has up to 3 Gaussians, but usually 1 or 2)
        oiii_profile, n_oiiip = get_oiii_profile(profiledir+source[n].split('_fits.fits')[0]+'_oiii_narrow_components.dat')

        # Now fit the Halpha area with the new limits on the narrow line widths from OIII
        # just fit the Ha + [NII] region
        wave_lo = 6400 # 6500
        wave_hi = 6700 # 6625
        lim1 = N.searchsorted(wave, wave_lo)
        lim2 = N.searchsorted(wave, wave_hi)




        # get some decent starting parameters using the [OIII] profile as initial guesses
        start = get_initial_guesses(oiii_profile, source[n])
        model_start, sha_br, sha_nr, sniia, sniib = model(start, oiii_profile, wave)

        # plot the initial guesses before any fits below have a chance to crash
        plot_model_fit(flux[lim1:lim2], fluxerr[lim1:lim2], start, model_start[lim1:lim2], sha_br[lim1:lim2], sha_nr[lim1:lim2], sniia[lim1:lim2], sniib[lim1:lim2], wave[lim1:lim2], fitsavedir+source[n]+'_initial_guess_fit.png', thelabel='Initial guess for the Gaussian fit')




        # do some basic optimization (refining initial guesses, and this doesn't take too long)
        # ~ 5 ms per call to curve_fit()
        xdata = wave[lim1:lim2]
        ydata = flux[lim1:lim2]
        ysig  = fluxerr[lim1:lim2]
        # some quite loose bounds on parameters (no absorption lines, sigma not 0 but not totally unbounded,
        # wavelength bounds to match those of emcee to avoid crashes)
        # in tuples of (min, max) per parameter a, u, s, a1, s1, an1
        # bounds = (0.0, N.inf), (6555, 6580), (0.01, 2000), (0.0, N.inf), (6555, 6580), (0.0, N.inf)
        # but curve_fit() wants a tuple of ([mins], [maxes])
        fitlim = ([0.0, 6555., 0.01, 0.0, 6555., 0.0], [N.inf, 6580., 2000., N.inf, 6580., N.inf])


        # at least one of the sources on the full list will crash in emcee because of shenanigans
        # with the continuum at the edges, so try to anticipate that and prevent it

        warnings.filterwarnings('error')
        try:
            popt,  pcov  = curve_fit(modelonly,            xdata, ydata, p0=start, bounds=fitlim, sigma=ysig)
            lpopt, lpcov = curve_fit(modelonly_lorentzian, xdata, ydata, p0=start, bounds=fitlim, sigma=ysig)
        except Warning:
            print("**********************************************************")
            print("**********************************************************")
            print("   Emcee will most likely crash, shrinking fit region     ")
            print("                and crossing fingers...                   ")
            print("**********************************************************")
            print("**********************************************************")

            # this source has a big region of craptacularity in the spectrum/errors itself just redward of Ha/[NII]
            # so do a little extra shrinking ONLY IN THIS CASE
            if source[n] == 'spSpec-54174-2494-045_fits.fits':
                lim1 = N.searchsorted(wave, 6500)
                lim2 = N.searchsorted(wave, 6595)
            else:
                lim1 = N.searchsorted(wave, wave_lo+75)
                lim2 = N.searchsorted(wave, wave_hi-75)
            xdata = wave[lim1:lim2]
            ydata = flux[lim1:lim2]
            ysig  = fluxerr[lim1:lim2]
            # there is probably a better way to iterate through this?
            try:
                popt,  pcov  = curve_fit(modelonly,            xdata, ydata, p0=start, bounds=fitlim, sigma=ysig)
                lpopt, lpcov = curve_fit(modelonly_lorentzian, xdata, ydata, p0=start, bounds=fitlim, sigma=ysig)
            except Warning:
                print("\n Fit still likely to crash after shrinking by 150 angstroms. ONCE MORE")
                print("   (after that you're on your own)")

                lim1 = N.searchsorted(wave, wave_lo+100)
                lim2 = N.searchsorted(wave, wave_hi-80)
                xdata = wave[lim1:lim2]
                ydata = flux[lim1:lim2]
                ysig  = fluxerr[lim1:lim2]

                try:
                    popt,  pcov  = curve_fit(modelonly,            xdata, ydata, p0=start, bounds=fitlim, sigma=ysig)
                    lpopt, lpcov = curve_fit(modelonly_lorentzian, xdata, ydata, p0=start, bounds=fitlim, sigma=ysig)
                except:
                    popt = get_initial_guesses(oiii_profile, source[n])
                    lpopt = get_initial_guesses(oiii_profile, source[n])


        # I mean, I kinda should use these fits as the start for emcee, yes?
        # The gandalf fits were performed *without* any of the narrow-line profile constraints
        # we're using, so the initial guesses are often quite wrong. These should be much better
        # or at least not worse
        # (do sanity checks so that we can still use the Gandalf values if things go astray)
        start_gauss   = sanity_check_parameters( popt, oiii_profile, source[n])
        start_lorentz = sanity_check_parameters(lpopt, oiii_profile, source[n])

        # also plot these, just because plot everything
        results,  ha_br,  ha_nr,  niia,  niib  = model(popt, oiii_profile, wave)
        lresults, lha_br, lha_nr, lniia, lniib = model_lorentzian(lpopt, oiii_profile, wave)

        plot_model_fit(flux[lim1:lim2], fluxerr[lim1:lim2],  popt,  results[lim1:lim2],  ha_br[lim1:lim2],  ha_nr[lim1:lim2],  niia[lim1:lim2],  niib[lim1:lim2],  wave[lim1:lim2], fitsavedir+source[n]+'_scipy_fit.png', thelabel='Scipy Gaussian best-fit')
        plot_model_fit(flux[lim1:lim2], fluxerr[lim1:lim2], lpopt, lresults[lim1:lim2], lha_br[lim1:lim2], lha_nr[lim1:lim2], lniia[lim1:lim2], lniib[lim1:lim2], wave[lim1:lim2], fitsavedir+source[n]+'_scipy_lorentz_fit.png', thelabel='Scipy Lorentzian best-fit')


        # I should do this with an error catcher, but, for now, testing
        #if source[n] == 'spSpec-52368-0881-064_fits.fits':
        #    start_gauss   = start
        #    start_lorentz = start

        #p0  = [start + 1e-4*N.random.randn(ndim) for i in range(nwalkers)]
        p0  = [start_gauss   + 1e-4*N.random.randn(ndim) for i in range(nwalkers)]
        lp0 = [start_lorentz + 1e-4*N.random.randn(ndim) for i in range(nwalkers)]


        #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(wave[lim1:lim2], flux[lim1:lim2], fluxerr[lim1:lim2], oiii_width/oiii_wave))
        sampler         = emcee.EnsembleSampler(nwalkers, ndim, lnprob,         args=(wave[lim1:lim2], flux[lim1:lim2], fluxerr[lim1:lim2], oiii_profile))
        sampler_lorentz = emcee.EnsembleSampler(nwalkers, ndim, lnprob_lorentz, args=(wave[lim1:lim2], flux[lim1:lim2], fluxerr[lim1:lim2], oiii_profile))


        ### burn in runs
        try:
            pos, prob, state = sampler.run_mcmc(p0, burnin)
        except Warning:
            print("\n\nEmcee seems to be having a problem with the initial Gaussian guesses - resetting and trying again...\n\n")
            p0  = [start   + 1e-4*N.random.randn(ndim) for i in range(nwalkers)]
            pos, prob, state = sampler.run_mcmc(p0, burnin)

        samples = sampler.chain[:,:,:].reshape((-1,ndim))
        N.save(fitsavedir+source[n]+'_samples_burn_in.npy', samples)
        ### Here you can look at the position of the walkers if you like as they step through the parameter space.
        ### Uncomment the next line if you want it to make this plot for the burn-in phase
        walker_plot(samples, nwalkers, burnin, ndim, source[n]+'_burn_in', fitsavedir)
        sampler.reset()

        try:
            lpos, lprob, lstate = sampler_lorentz.run_mcmc(lp0, burnin)
        except Warning:
            print("\n\nEmcee seems to be having a problem with the initial Lorentzian guesses - resetting and trying again...\n\n")
            lp0 = [start   + 1e-4*N.random.randn(ndim) for i in range(nwalkers)]
            lpos, lprob, lstate = sampler_lorentz.run_mcmc(lp0, burnin)

        samples_lorentz = sampler_lorentz.chain[:,:,:].reshape((-1,ndim))
        N.save(fitsavedir+source[n]+'_samples_lorentz_burn_in.npy', samples_lorentz)
        ### Here you can look at the position of the walkers if you like as they step through the parameter space.
        ### Uncomment the next line if you want it to make this plot for the burn-in phase
        walker_plot(samples_lorentz, nwalkers, burnin, ndim, source[n]+'_lorentz_burn_in', fitsavedir)
        sampler_lorentz.reset()


        ### main sampler runs
        sampler.run_mcmc(pos, nsteps)
        samples = sampler.chain[:,:,:].reshape((-1,ndim))
        N.save(fitsavedir+source[n]+'_samples.npy', samples)
        ### Same again - uncomment the next line if you want to see the walker plot for the main sampler run
        walker_plot(samples, nwalkers, nsteps, ndim, source[n], fitsavedir)

        sampler_lorentz.run_mcmc(lpos, nsteps)
        samples_lorentz = sampler_lorentz.chain[:,:,:].reshape((-1,ndim))
        N.save(fitsavedir+source[n]+'_samples_lorentz.npy', samples_lorentz)
        ### Same again - uncomment the next line if you want to see the walker plot for the main sampler run
        walker_plot(samples_lorentz, nwalkers, nsteps, ndim, source[n]+"_lorentz", fitsavedir)



        print("Gauss: Mean acceptance fraction: {0:.3f}".format(N.mean(sampler.acceptance_fraction)))
        if N.mean(sampler.acceptance_fraction) > 0.5 or N.mean(sampler.acceptance_fraction) < 0.25:
            rerun_list.append(source[n])
            print('Gauss: Acceptance fractions out of optimal range!')
        else:
            pass

        print("Lorentz: Mean acceptance fraction: {0:.3f}".format(N.mean(sampler_lorentz.acceptance_fraction)))
        if N.mean(sampler_lorentz.acceptance_fraction) > 0.5 or N.mean(sampler_lorentz.acceptance_fraction) < 0.25:
            rerun_list_lorentz.append(source[n])
            print('Lorentz: Acceptance fractions out of optimal range!')
        else:
            pass

        # BDS note: this next bit is going to break in Python 3 because you can't
        # pickle a lambda (lambdas don't count as named functions and you can only
        # pickle named functions starting in Python 3)
        best = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(samples, [16,50,84],axis=0)))
        N.save(fitsavedir+source[n]+'_best_fit.npy', best)
        print("Gauss:\n", best)

        # broad  = gaussbroad(best[0][0],            best[1][0],            best[2][0],            wave)
        # broadp = gaussbroad(best[0][0]+best[0][1], best[1][0]+best[4][1], best[2][0]+best[2][1], wave)
        # broadm = gaussbroad(best[0][0]-best[0][2], best[1][0]-best[4][2], best[2][0]-best[2][2], wave)


        best_lorentz = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(samples_lorentz, [16,50,84],axis=0)))
        N.save(fitsavedir+source[n]+'_lorentz_best_fit.npy', best_lorentz)
        print("Lorentz:\n", best_lorentz)

        bestest = [best[0][0], best[1][0], best[2][0], best[3][0], best[4][0], best[5][0]]
        results, ha_br, ha_nr, niia, niib = model(bestest, oiii_profile, wave)

        bestest_lorentz = [best_lorentz[0][0], best_lorentz[1][0], best_lorentz[2][0], best_lorentz[3][0], best_lorentz[4][0], best_lorentz[5][0]]
        lresults, lha_br, lha_nr, lniia, lniib = model_lorentzian(bestest_lorentz, oiii_profile, wave)

        # print both the best-fit plots
        #plot_model_fit(flux, fluxerr, theta, results, ha_br, ha_nr, niia, niib, wave, outfile)
        plot_model_fit(flux[lim1:lim2], fluxerr[lim1:lim2], bestest,         results[lim1:lim2],  ha_br[lim1:lim2],  ha_nr[lim1:lim2],  niia[lim1:lim2],  niib[lim1:lim2],  wave[lim1:lim2], fitsavedir+source[n]+'_model_fit.png', thelabel='Emcee Gaussian best-fit')
        plot_model_fit(flux[lim1:lim2], fluxerr[lim1:lim2], bestest_lorentz, lresults[lim1:lim2], lha_br[lim1:lim2], lha_nr[lim1:lim2], lniia[lim1:lim2], lniib[lim1:lim2], wave[lim1:lim2], fitsavedir+source[n]+'_lorentz_model_fit.png', thelabel='Emcee Lorentzian best-fit')

        thewave     = best[1][0]
        thefwhm     = calc_fwhm(wave, ha_br)
        #thefwhm_kms = c * (thefwhm/thewave) # I think this is an approximation?
        thefwhm_kms_sig = 2.355*c * N.log((best[2][0]/thewave) + 1.) # gaussian assumption - just a sanity check
        thefwhm_kms     =       c * N.log((thefwhm   /thewave) + 1.)

        print('FWHM calc %.2f (%.2f from sig as a sanity check) aka %.2f (%.2f) km/s at lambda = %.1f Angstroms' % (thefwhm, best[2][0]*2.355, thefwhm_kms, thefwhm_kms_sig, thewave))

    else:
        print("Skipping %s..." % source[n])

print("Here's the list of spectra that need running (Gauss) because the acceptance fractions were out of optimal range: ", rerun_list)
print("Here's the list of spectra that need running (Lorentz) because the acceptance fractions were out of optimal range: ", rerun_list_lorentz)
