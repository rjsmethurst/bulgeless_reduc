"""File to use emcee to fit a double gaussian with a narrow and broad peak to emission data from GANDALF."""

import numpy as N
import pylab as P
import pyfits as F
import emcee 
import glob
import time

def walker_plot(samples, nwalkers, limit):
        s = samples.reshape(nwalkers, -1, 5)
        s = s[:,:limit, :]
        fig = P.figure(figsize=(8,14))
        ax1 = P.subplot(5,1,1)
        ax2 = P.subplot(5,1,2)
        ax3 = P.subplot(5,1,3)
        ax4 = P.subplot(5,1,4)
        ax5 = P.subplot(5,1,5)
        for n in range(len(s)):
            ax1.plot(s[n,:,0], 'k')
            ax2.plot(s[n,:,1], 'k')
            ax3.plot(s[n,:,2], 'k')
            ax4.plot(s[n,:,3], 'k')
            ax5.plot(s[n,:,4], 'k')
        ax1.tick_params(axis='x', labelbottom='off')
        ax1.set_ylabel(r'amplitude narrow $H\alpha$')
        ax2.tick_params(axis='x', labelbottom='off')
        ax2.set_ylabel(r'stdev narrow $H\alpha$')
        ax3.tick_params(axis='x', labelbottom='off')
        ax3.set_ylabel(r'amplitude broad $H\alpha$')
        ax4.tick_params(axis='x', labelbottom='off')
        ax4.set_ylabel(r'stdev broad $H\alpha$')
        ax5.set_ylabel(r'amplitude narrow [NII] 6583')
        P.subplots_adjust(hspace=0.1)
        P.tight_layout()
        save_fig = './walkers_steps_'+str(time.strftime('%H_%M_%d_%m_%y'))+'.pdf'
        fig.savefig(save_fig)
        return fig


def model(theta, wave):
    # 12 theta values are 3 to describe narrow Halpha, 3 for broad Halpha, 3 for [NII] at 6547 and 3 for [NII] at 6583
    # a, u, s, a1, u1, s1, an1, un1, sn1, an2, un2, sn2 = theta
    # but some of these are set, so really we only have 7 variables
    a, s, a1, s1, an2  = theta
    u = u1 = 6562.8
    un1 = 6547.96
    un2 = 6583.34
    an1 = 0.34*an2
    sn1 = sn2 = s
    return a * N.exp(- (((wave - u)**2) /(s**2))) + a1 * N.exp(- (((wave - u1)**2) /(s1**2))) + an1 * N.exp(- (((wave - un1)**2) /(sn1**2))) + an2 * N.exp(- (((wave - un2)**2) /(sn2**2))) 



def lnlike(theta, wave, flux, fluxerr):
    pred_f = model(theta, wave)
    #print pred_f
    chi =  -0.5*N.log(2*N.pi*fluxerr**2)-0.5*((flux-pred_f)**2/fluxerr**2)
    #print N.sum(chi)
    return N.sum(chi)

def lnprior(theta, wave, flux):
    a, s, a1, s1, an2  = theta
    if a > 0 and a1 > 0 and an2 > 0 and 0 < s < 1000 and 0 < s1 < 1000 and s < s1:
    	return 0.0
    else:
    	return -N.inf

def lnprob(theta, wave, flux, fluxerr):
	lp = lnprior(theta, wave, flux)
        #print lp
    	if not N.isfinite(lp):
        	return -N.inf
        #print lp + lnlike(theta, wave, flux, fluxerr)
    	return lp + lnlike(theta, wave, flux, fluxerr)

def gauss(a, u, s, wave):
    return a * N.exp(- (((wave - u)**2) /(s**2)))

ndim = 5
nwalkers = 100
nsteps = 500
burnin = 1000
# guess some reasonable start values for a, s, a1, s1, an2
start = [2, 20, 1, 40, 2]

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
dir2 = '/Users/becky/Projects/int_reduc/bdmass_fits_gandalf_bds/combined_sdss_spectra/emcee_gauss_fits_4_components/'

for n in range(len(source)):
    bf = F.open(dir1+source[n])
    hdr = bf[0].header
    flux = bf[2].data
    fluxerr = N.sqrt(N.abs(flux))+N.random.random_sample(len(flux))*1E-30
    lam = hdr['CRVAL1'] + hdr['CD1_1']*(N.arange(hdr['NAXIS1'] -  hdr['CRPIX1'] + 1))
    wave = (10**lam)/(1+hdr['Z'])
    print source[n]
    p0 = [start + 1e-4*N.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(wave[2500:], flux[2500:], fluxerr[2500:]))
    ### burn in run 
    pos, prob, state = sampler.run_mcmc(p0, burnin)
    samples = sampler.chain[:,:,:].reshape((-1,ndim))
    N.save(dir2+source[n]+'_samples_burn_in.npy', samples)
    ### Here you can look at the position of the walkers if you like as they step through the parameter space.
    ### Uncomment the next line if you want it to make this plot for the burn-in phase
    #walker_plot(samples, nwalkers, burnin)
    sampler.reset()
    ### main sampler run 
    sampler.run_mcmc(pos, nsteps)
    samples = sampler.chain[:,:,:].reshape((-1,ndim))
    N.save(dir2+source[n]+'_samples.npy', samples)
    ### Same again - uncomment the next line if you want to see the walker plot for the main sampler run
    #walker_plot(samples, nwalkers, nsteps)
    best = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(samples, [16,50,84],axis=0)))
    N.save(dir2+source[n]+'_best_fit.npy', best)
    results = model([best[0][0], best[1][0], best[2][0], best[3][0], best[4][0]], wave)
    broad_only = gauss(best[2][0], 6562.8, best[3][0], wave)
    narrow_only = gauss(best[0][0], 6562.8, best[1][0], wave)
    narrow_NII_1 = gauss(0.34*best[4][0], 6547.96, best[1][0], wave)
    narrow_NII_2 = gauss(best[4][0], 6583.34, best[1][0], wave)
    print best
    lim1 = N.searchsorted(wave, 6400)
    lim2 =N.searchsorted(wave, 6700)
    P.figure()
    P.plot(wave[lim1:lim2], flux[lim1:lim2], c='k', linewidth=2)
    P.plot(wave[lim1:lim2], results[lim1:lim2], c='r', linestyle='dashed')
    P.plot(wave[lim1:lim2], broad_only[lim1:lim2], c='b')
    P.plot(wave[lim1:lim2], narrow_only[lim1:lim2], c='m')
    P.plot(wave[lim1:lim2], narrow_NII_1[lim1:lim2], c='g')
    P.plot(wave[lim1:lim2], narrow_NII_2[lim1:lim2], c='g')
    P.savefig(dir2+source[n]+'_model_fit.png')
