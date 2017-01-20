"""File to use emcee to fit a double gaussian with a narrow and broad peak to emission data from GANDALF."""

import numpy as N
import pylab as P
import pyfits as F
import emcee 
import glob
import time
from scipy.optimize import minimize
from astropy.table import Table

def walker_plot(samples, nwalkers, limit, ndim, source, dir):
        s = samples.reshape(nwalkers, -1, ndim)
        s = s[:,:limit, :]
        fig = P.figure(figsize=(8,18))
        ax1 = P.subplot(ndim,1,1)
        ax2 = P.subplot(ndim,1,2)
        ax3 = P.subplot(ndim,1,3)
        ax4 = P.subplot(ndim,1,4)
        ax5 = P.subplot(ndim,1,5)
        ax6 = P.subplot(ndim,1,6)
        ax7 = P.subplot(ndim,1,7)
        ax8 = P.subplot(ndim,1,8)
        ax9 = P.subplot(ndim,1,9)
        ax10 = P.subplot(ndim,1,10)
        ax1.plot(s[:,:,0].T, 'k', rasterized=True)
        ax2.plot(s[:,:,1].T, 'k', rasterized=True)
        ax3.plot(s[:,:,2].T, 'k', rasterized=True)
        ax4.plot(s[:,:,3].T, 'k', rasterized=True)
        ax5.plot(s[:,:,4].T, 'k', rasterized=True)
        ax6.plot(s[:,:,5].T, 'k', rasterized=True)
        ax7.plot(s[:,:,6].T, 'k', rasterized=True)
        ax8.plot(s[:,:,7].T, 'k', rasterized=True)
        ax9.plot(s[:,:,8].T, 'k', rasterized=True)
        ax10.plot(s[:,:,9].T, 'k', rasterized=True)
        ax1.tick_params(axis='x', labelbottom='off')
        ax1.set_ylabel(r'amp N $H\alpha$')
        ax2.tick_params(axis='x', labelbottom='off')
        ax2.set_ylabel(r'wave $H\alpha$')
        ax3.tick_params(axis='x', labelbottom='off')
        ax3.set_ylabel(r'stdev N $H\alpha$')
        ax4.tick_params(axis='x', labelbottom='off')
        ax4.set_ylabel(r'amp B $H\alpha$')
        ax5.tick_params(axis='x', labelbottom='off')
        ax5.set_ylabel(r'wave B $H\alpha$')
        ax6.tick_params(axis='x', labelbottom='off')
        ax6.set_ylabel(r'stdev B $H\alpha$')
        ax7.tick_params(axis='x', labelbottom='off')
        ax7.set_ylabel(r'amp N [NII] 6547')
        ax8.tick_params(axis='x', labelbottom='off')
        ax8.set_ylabel(r'stdev N [NII] 6547')
        ax9.tick_params(axis='x', labelbottom='off')
        ax9.set_ylabel(r'amp N [NII] 6583')
        ax10.set_ylabel(r'stdev N [NII] 6583')
        P.subplots_adjust(hspace=0.05)
        #P.tight_layout()
        fig.savefig(dir+'./walkers_steps_'+source+'.png')

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

def model(theta, wave):
    # 12 theta values are 3 to describe narrow Halpha, 3 for broad Halpha, 3 for [NII] at 6547 and 3 for [NII] at 6583
    # a, u, s, a1, u1, s1, an1, un1, sn1, an2, un2, sn2 = theta
    # but some of these are set, so really we only have 7 variables
    a, u, s, a1, u1, s1, an1, sn1, an2, sn2  = theta
    un1 = u - (6562.8 - 6547.96)
    un2 = u + (6583.34 - 6562.8)
    #return a * (1/(s*N.sqrt(2*N.pi))) * N.exp(- (((wave - u)**2) /(s**2))) + a1 *(1/(s1*N.sqrt(2*N.pi))) * N.exp(- (((wave - u1)**2) /(s1**2))) + an1 * (1/(sn1*N.sqrt(2*N.pi))) * N.exp(- (((wave - un1)**2) /(sn1**2))) + an2 * (1/(sn2*N.sqrt(2*N.pi))) * N.exp(- (((wave - un2)**2) /(sn2**2))) 
    return a * N.exp(- (((wave - u)**2) /(s**2))) + a1 * N.exp(- (((wave - u1)**2) /(s1**2))) + an1 * N.exp(- (((wave - un1)**2) /(sn1**2))) + an2 * N.exp(- (((wave - un2)**2) /(sn2**2))) 

def lnlike(theta, wave, flux, fluxerr):
    pred_f = model(theta, wave)
    #print pred_f
    chi =  -0.5*N.log(2*N.pi*fluxerr**2)-0.5*((flux-pred_f)**2/fluxerr**2)
    #print N.sum(chi)
    return N.sum(chi)


def lnprior(theta, wave, flux, oiii_width):
    a, u, s, a1, u1, s1, an1, sn1, an2, sn2 = theta
    if a > 0 and a1 > 0 and an1 >= 0  and an2 >= 0 and an1 <= 0.34*an2 and an2 <= a and u > 6555 and u < 6580 and u1 > 6555 and u1 < 6580 and N.abs(u-u1)<3 and s > 0 and s < 3*6562.8*oiii_width and sn1 >= 0 and sn2 >= 0 and sn2 < 3*6583.34*oiii_width and sn1 < 3*6547.96*oiii_width and s1 > 0 and s1 < 2000 and s < s1:
    	return 0.0
    else:
    	return -N.inf

def lnprob(theta, wave, flux, fluxerr, oiii_width):
	lp = lnprior(theta, wave, flux, oiii_width)
        #print lp
    	if not N.isfinite(lp):
        	return -N.inf
        #print lp + lnlike(theta, wave, flux, fluxerr)
    	return lp + lnlike(theta, wave, flux, fluxerr)

def gauss(a, u, s, wave):
    #return a * (1/(s*N.sqrt(2*N.pi))) * N.exp(- (((wave - u)**2) /(s**2)))
    return a * N.exp(- (((wave - u)**2) /(s**2)))

def calc_fwhm(wave, emission):
    max = N.max(emission)
    i = N.argmax(emission)
    hm = max/2
    idx = N.abs(emission-hm)
    fidx = N.argmin(idx[:i])
    sidx = N.argmin(idx[i:]) + i
    fwhm = wave[sidx] - wave[fidx]
    return fwhm

ndim = 10
nwalkers = 200
nsteps = 500
burnin = 2000

source = list(['spSpec-54241-2516-619_bluecutoff_fits.fits',
                'spSpec-54184-2607-477_bluecutoff_fits.fits',
                'spSpec-53767-2246-066_bluecutoff_fits.fits',
                'spSpec-53762-1993-142_bluecutoff_fits.fits',
                'spSpec-53739-2365-149_bluecutoff_fits.fits',
                'spSpec-53523-1655-151_bluecutoff_fits.fits',
                'spSpec-53493-2088-266_bluecutoff_fits.fits',
                'spSpec-53474-2007-205_bluecutoff_fits.fits',
                'spSpec-53386-1943-466_bluecutoff_fits.fits',
                'spSpec-53172-1822-308_bluecutoff_fits.fits',
                'spSpec-53084-1619-245_bluecutoff_fits.fits',
                'spSpec-53053-1363-229_bluecutoff_fits.fits',
                'spSpec-52972-1589-277_bluecutoff_fits.fits',
                'spSpec-52642-0837-059_bluecutoff_fits.fits',
                'spSpec-52438-0965-203_bluecutoff_fits.fits',
                'spSpec-52435-0972-040_bluecutoff_fits.fits',
                'spSpec-52398-0957-169_bluecutoff_fits.fits',
                'spSpec-52368-0881-412_bluecutoff_fits.fits',
                'spSpec-52258-0725-580_fits.fits',
                'spSpec-51789-0398-142_fits.fits',
                'spSpec-51791-0374-142_fits.fits',
                'spSpec-51816-0410-523_fits.fits',
                'spSpec-51868-0441-578_fits.fits',
                'spSpec-51871-0420-314_fits.fits',
                'spSpec-51908-0450-575_fits.fits',
                'spSpec-51909-0276-038_fits.fits',
                'spSpec-51965-0475-008_fits.fits',
                'spSpec-51993-0542-364_fits.fits',
                'spSpec-52024-0517-360_fits.fits',
                'spSpec-52054-0622-210_fits.fits',
                'spSpec-52055-0589-111_fits.fits',
                'spSpec-52059-0597-195_fits.fits',
                'spSpec-52059-0597-246_fits.fits',
                'spSpec-52199-0641-349_fits.fits',
                'spSpec-52254-0765-058_fits.fits',
                'spSpec-52282-0770-213_fits.fits',
                'spSpec-52325-0783-556_fits.fits',
                'spSpec-52338-0850-245_fits.fits',
                'spSpec-52353-0877-180_fits.fits',
                'spSpec-52365-0606-515_fits.fits',
                'spSpec-52366-0508-074_fits.fits',
                'spSpec-52368-0881-064_fits.fits',
                'spSpec-52431-0978-524_fits.fits',
                'spSpec-52435-0972-118_fits.fits',
                'spSpec-52643-1000-593_fits.fits',
                'spSpec-52669-1159-526_fits.fits',
                'spSpec-52703-1005-104_fits.fits',
                'spSpec-52703-1165-424_fits.fits',
                'spSpec-52705-0936-564_fits.fits',
                'spSpec-52707-0991-603_fits.fits',
                'spSpec-52710-0993-075_fits.fits',
                'spSpec-52725-1231-580_fits.fits',
                'spSpec-52725-1286-598_fits.fits',
                'spSpec-52731-1288-236_fits.fits',
                'spSpec-52734-1233-185_fits.fits',
                'spSpec-52751-1221-437_fits.fits',
                'spSpec-52756-1170-162_fits.fits',
                'spSpec-52762-1237-077_fits.fits',
                'spSpec-52781-1318-242_fits.fits',
                'spSpec-52814-1345-367_fits.fits',
                'spSpec-52999-1597-227_fits.fits',
                'spSpec-53002-1430-485_fits.fits',
                'spSpec-53062-1445-246_fits.fits',
                'spSpec-53112-1396-329_fits.fits',
                'spSpec-53138-1460-481_fits.fits',
                'spSpec-53141-1417-078_fits.fits',
                'spSpec-53146-1398-373_fits.fits',
                'spSpec-53314-1866-403_fits.fits',
                'spSpec-53330-1936-059_fits.fits',
                'spSpec-53384-1871-066_fits.fits',
                'spSpec-53386-1941-632_fits.fits',
                'spSpec-53415-1762-496_fits.fits',
                'spSpec-53433-2027-603_fits.fits',
                'spSpec-53460-2098-582_fits.fits',
                'spSpec-53467-2110-043_fits.fits',
                'spSpec-53473-1692-398_fits.fits',
                'spSpec-53474-1628-465_fits.fits',
                'spSpec-53474-2017-180_fits.fits',
                'spSpec-53475-2021-027_fits.fits',
                'spSpec-53476-2033-057_fits.fits',
                'spSpec-53520-1657-144_fits.fits',
                'spSpec-53737-2004-425_fits.fits',
                'spSpec-53818-2093-618_fits.fits',
                'spSpec-53846-1849-220_fits.fits',
                'spSpec-53858-2101-496_fits.fits',
                'spSpec-53883-1778-119_fits.fits',
                'spSpec-54084-2501-358_fits.fits',
                'spSpec-54084-2501-384_fits.fits',
                'spSpec-54174-2494-045_fits.fits',
                'spSpec-54179-2504-639_fits.fits',
                'spSpec-54231-2654-275_fits.fits',
                'spSpec-54481-2614-279_fits.fits',
                'spSpec-54537-2785-319_fits.fits',
                'spSpec-54540-2779-556_fits.fits',
                'spSpec-54560-2953-258_fits.fits',
                'spSpec-54572-2531-250_fits.fits'])

dir1 = '/Users/becky/Projects/int_reduc/bdmass_fits_gandalf_bds/combined_sdss_spectra/'
dir2 = '/Users/becky/Projects/int_reduc/bdmass_fits_gandalf_bds/combined_sdss_spectra/emcee_gauss_fits_4_components_Brooke_oiii_width/'
dir3 = '/Users/becky/Projects/int_reduc/bdmass_fits_gandalf_bds/combined_sdss_spectra/oiii_line_components_profiles_tables/'

rerun_list=[]
for n in range(len(source)):
    bf = F.open(dir1+source[n])
    hdr = bf[0].header
    flux = bf[2].data
    fluxerr = N.sqrt(N.abs(flux))+N.random.random_sample(len(flux))*1E-30
    lam = hdr['CRVAL1'] + hdr['CD1_1']*(N.arange(hdr['NAXIS1'] -  hdr['CRPIX1'] + 1))
    wave = (10**lam)/(1+hdr['Z'])
    print source[n]
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
    # P.savefig(dir2+source[n]+'_oiii_model_fit.png') 

    # First load up the width of the narrow [OIII] component output by Brooke from GANDALF
    oiiip = Table.read(dir3+source[n].split('_fits.fits')[0]+'_oiii_narrow_components.dat', format='ascii')
    oiii_width = oiiip['sigma_Ang'][0]
    oiii_amp = oiiip['observed_A'][0]
    oiii_wave = oiiip['lambda_fit'][0]
    # Now fit the Halpha area with the new limits on the narrow line widths from OIII
    lim1 = N.searchsorted(wave, 6400)
    lim2 =N.searchsorted(wave, 6700)
    # Give emcee some appropriate starting points based on the OIII fits. 
    # guess some reasonable start values for a, u, s, a1, u1, s1, an1, sn1, an2, sn2, anw1, snw1, anw2, snw2
    start = [oiii_amp, 6562, (6562.8/oiii_wave)*oiii_width, 0.5*oiii_amp, 6562, 10*(6562.8/oiii_wave)*oiii_width, 0.34*oiii_amp, (6548/oiii_wave)*oiii_width, oiii_amp, (6583/oiii_wave)*oiii_width]
    p0 = [start + 1e-4*N.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(wave[lim1:lim2], flux[lim1:lim2], fluxerr[lim1:lim2], oiii_width/oiii_wave))
    ### burn in run 
    pos, prob, state = sampler.run_mcmc(p0, burnin)
    samples = sampler.chain[:,:,:].reshape((-1,ndim))
    N.save(dir2+source[n]+'_samples_burn_in.npy', samples)
    ### Here you can look at the position of the walkers if you like as they step through the parameter space.
    ### Uncomment the next line if you want it to make this plot for the burn-in phase
    walker_plot(samples, nwalkers, burnin, ndim, source[n]+'_burn_in', dir2)
    sampler.reset()
    ### main sampler run 
    sampler.run_mcmc(pos, nsteps)
    samples = sampler.chain[:,:,:].reshape((-1,ndim))
    N.save(dir2+source[n]+'_samples.npy', samples)
    ### Same again - uncomment the next line if you want to see the walker plot for the main sampler run
    walker_plot(samples, nwalkers, nsteps, ndim, source[n], dir2)
    print "Mean acceptance fraction: {0:.3f}".format(N.mean(sampler.acceptance_fraction))
    if N.mean(sampler.acceptance_fraction) > 0.5 or N.mean(sampler.acceptance_fraction) < 0.25:
        rerun_list.append(source[n])
        print 'Acceptance fractions out of optimal range!'
    else:
        pass
    best = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(samples, [16,50,84],axis=0)))
    N.save(dir2+source[n]+'_best_fit.npy', best)
    results = model([best[0][0], best[1][0], best[2][0], best[3][0], best[4][0], best[5][0], best[6][0], best[7][0], best[8][0], best[9][0]], wave)
    broad_only = gauss(best[3][0], best[4][0], best[5][0], wave)
    narrow_only = gauss(best[0][0], best[1][0], best[2][0], wave)
    narrow_NII_1 = gauss(best[6][0], best[1][0]-(6562.8 - 6547.96), best[7][0], wave)
    narrow_NII_2 = gauss(best[8][0], best[1][0]+(6583.34 - 6562.8), best[9][0], wave)
    print best
    print 'FWHM calc', calc_fwhm(wave, broad_only)
    P.figure()
    ax1 =P.subplot(111)
    ax1.plot(wave[lim1:lim2], flux[lim1:lim2], c='k', linewidth=2)
    ax1.plot(wave[lim1:lim2], bf[0].data[lim1:lim2]-bf[0].data[lim1:lim2][0], c='k', alpha=0.3, linewidth=2)
    ax1.plot(wave[lim1:lim2], results[lim1:lim2], c='r', linestyle='dashed')
    ax1.plot(wave[lim1:lim2], broad_only[lim1:lim2], c='b')
    ax1.plot(wave[lim1:lim2], narrow_only[lim1:lim2], c='m')
    ax1.plot(wave[lim1:lim2], narrow_NII_1[lim1:lim2], c='g')
    ax1.plot(wave[lim1:lim2], narrow_NII_2[lim1:lim2], c='g')
    ax1.text(0.1, 0.9, r'FWHM = %3.2f' % calc_fwhm(wave, broad_only), transform=ax1.transAxes)
    P.savefig(dir2+source[n]+'_model_fit.png')

print "Here's the list of spectra that need running because the acceptance fractions were out of optimal range: ", rerun_list
