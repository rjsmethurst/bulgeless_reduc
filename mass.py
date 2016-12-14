""" File to calculate total mass of galaxy from (u-r) colours and black hole masses from FWHM of Halpha broad lines in galaxy spectra."""

import numpy as N
import matplotlib.pyplot as P
import pyfits as F
from prefig import Prefig

import os

from astropy.cosmology import FlatLambdaCDM

from astropy import units as un
from astropy.table import Table, vstack, Column

import math
import linmix

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

def mgal(mpet, psf):
    return -2.5 * N.log10((10**(-mpet/2.5))-(10**(-psf/2.5)))

def calc_ur(psf_u, psfErr_u, psf_r, psfErr_r, petro_u, petroErr_u, petro_r, petroErr_r):
        gal_u = mgal(petro_u, psf_u)
        gal_r = mgal(petro_r, psf_r)
        ur = gal_u - gal_r
        Err_u = 1/((10**(-petro_u/2.5))-(10**(-psf_u/2.5))) * ( ((10**(-petro_u/2.5))*petroErr_u)**2 + ((10**(-psf_u/2.5))*psfErr_u)**2 )**(0.5)
        Err_r = 1/((10**(-petro_r/2.5))-(10**(-psf_r/2.5))) * ( ((10**(-petro_r/2.5))*petroErr_r)**2 + ((10**(-psf_r/2.5))*psfErr_r)**2 )**(0.5)
        Err_ur = ( (Err_u)**2 + (Err_r) )**(0.5)
        return ur, Err_ur, gal_r, Err_r

def bhmass(ld, Err_ld, fwhm, Err_fwhm, wave, fit, fitp, fitm):
    fwhm_v = (fwhm/6562.8) * c  #fwhm in km/s
    Err_fwhm_v = (Err_fwhm/6562.8) * c
    print 'fwhm', fwhm_v
    #lum = flux*1E-17*6562.8*(4*N.pi*(d**2)) # luminosity in erg/s but flux in erg/cm2/s/A - so integrate over all wavelengths - i.e. multiply by the wavelength
    #total_flux = (-0.5 * N.sqrt(N.pi) * wave[-1] * a * s * math.erf((u-wave[-1])/s)) - (-0.5 * N.sqrt(N.pi) * wave[0] * a * s * math.erf((u-wave[0])/s))  #integrate over the entire guassian rather than taking the maximum of the Halpha flux which includes the narrow component
    total_flux = N.abs(N.trapz(fit, wave)) * (un.erg/(un.cm**2)/un.s)
    tfp = N.abs(N.trapz(fitp, wave)) * (un.erg/(un.cm**2)/un.s)
    tfm = N.abs(N.trapz(fitm, wave)) * (un.erg/(un.cm**2)/un.s)
    Err_total_flux =  N.mean([N.abs(total_flux-tfp).value, N.abs(total_flux-tfm).value])*(un.erg/un.s/un.cm**2)
    lum = total_flux*1E-17*(4*N.pi*(ld.to(un.cm)**2)) # luminosity in erg/s but flux in erg/cm2/s - so need d in cm2
    Err_lum = lum * ( (Err_total_flux/total_flux).value**2 + (2*Err_ld/ld.value)**2 )**(0.5)
    print 'luminosity', N.log10(lum.value), ' +- ', Err_lum/(lum*N.log(10))
    mbh = 2E6 * ((lum/(1E42*un.erg/un.s))**0.455) * ((fwhm_v/(1E3*un.km/un.s))**2.06)
    Err_mbh = mbh * ( (0.55/3)**2 + (0.45*Err_lum/lum).value**2 + (0.03*N.log(lum.value/1E42))**2 + (2.06*Err_fwhm_v/fwhm_v).value**2 + (0.06*N.log(fwhm_v.value/1E3))**2 )**(0.5)
    print N.log10(mbh.value), ' +- ', Err_mbh/(mbh*N.log(10))
    return (lum/lsol), (Err_lum/lsol), N.log10(mbh), Err_mbh/(mbh*N.log(10))

 
def calc_fwhm(wave, emission):
    max = N.max(emission)
    i = N.argmax(emission)
    hm = max/2
    idx = N.abs(emission-hm)
    fidx = N.argmin(idx[:i])
    sidx = N.argmin(idx[i:]) + i
    fwhm = wave[sidx] - wave[fidx]
    return fwhm

def gauss(a, u, s, a1, u1, s1, wave):
    return a * N.exp(- (((wave - u)**2) /(s**2))) + a1 * N.exp(- (((wave - u1)**2) /(s1**2)))

def gaussbroad(a, u, s, wave):
    return a * N.exp(- (((wave - u)**2) /(s**2)))

ints = Table.read('ints_colours_masses_integrate_WISE.fits', format='fits')

sourceList = ['Q078017', 'Q507953', 'Q464967', 'Q292635', 'Q186225']
sourceID = ['1237679457600078017', '1237666245208507953', '1237668631602464967', '1237662981045292635', '1237671688011186225']
target_flux = [11.34, 50.72, 36.55, 70.92, 672.08]
target_fwhm = [0.3, 2.92, 9.51, 6.61, 24.90]
emission_fwhm = [0.3, 2.92, 9.51, 6.61, 24.90]
target_I_v0 = N.zeros(len(sourceList))


z = N.zeros(len(sourceList))
int_fwhm = N.zeros(len(sourceList))
Err_int_fwhm = N.zeros(len(sourceList))
int_mbh = N.zeros(len(ints))
Err_int_mbh = N.zeros(len(ints))
int_mtot = N.zeros(len(ints))
Err_int_mtot = N.zeros(len(ints))
int_lum = N.zeros(len(ints))
Err_int_lum = N.zeros(len(ints))
int_flux_h_alpha = N.zeros(len(ints))
int_lbol = N.zeros(len(ints))
Err_int_lbol = N.zeros(len(ints))
ur = N.zeros(len(ints))
Err_ur = N.zeros(len(ints))
gal_r = N.zeros(len(ints))
Err_gal_r = N.zeros(len(ints))
gal_Mr = N.zeros(len(ints))
Err_gal_Mr = N.zeros(len(ints))

for n in range(len(z)):
    s = F.open('./int_reduced_spectra/'+sourceList[n]+'_extract_agn.fit')
    d = s[0].data
    h = s[0].header
    #l = s[0].header['crval1'] + s[0].header['cd1_1']*(N.arange(s[0].header['naxis1']))
    w = N.linspace(0-h['CRPIX2'], d.shape[0]-h['CRPIX2'], d.shape[0])
    l = w*h['cd2_2'] + h['crval2']
    if sourceList[n] == 'Q464967':
        ml = l[N.argmax(d[100:-100])+100]
        target_I_v0[n] = N.max(d[100:-100]) / 1E-17
    elif sourceList[n] == 'Q078017':
        ml = l[N.argmax(d[:100])]
        target_I_v0[n] = N.max(d[:100]) / 1E-17
    else:
        ml = l[N.argmax(d)]
        target_I_v0[n] = N.max(d) / 1E-17
    z[n] = (ml/6562.8) - 1
    #print 'lambda value', ml
    #print 'z = ',z[n]
    #ints[n,17] = z[n]   

for n in range(len(sourceList)):
    ur[n], Err_ur[n], gal_r, Err_gal_r = calc_ur(ints['col6'][n], ints['col7'][n], ints['col8'][n], ints['col9'][n], ints['col10'][n], ints['col11'][n], ints['col12'][n], ints['col13'][n])
    if ur[n] <= 2.1:
        log_m_l = -0.95 + 0.56 * ur[n]
        log_m_l_err = (0.56*Err_ur[n])**2
    else:
        log_m_l = -0.16 + 0.18 * ur[n]
        log_m_l_err = (0.18*Err_ur[n])**2
    ld = cosmo.luminosity_distance(z[n])
    Err_ld = N.mean([(cosmo.luminosity_distance(z[n]+0.001*z[n])-ld).value, (ld-cosmo.luminosity_distance(z[n]+0.001*z[n])).value])
    gal_Mr[n] = gal_r - 5 * (N.log10(ld.value * 1E6) - 1)
    Err_gal_Mr[n] = (Err_gal_r**2 + ((5*Err_ld)/(ld.value*N.log(10)))**2)**(0.5)
    int_mtot[n] = ((4.62 - gal_Mr[n])/2.5) + log_m_l
    Err_int_mtot[n] = ((Err_gal_Mr[n]/2.5)**2 + (log_m_l_err)**2)**(0.5)
    print int_mtot[n] 
    bf = F.open('./int_gandalf_results/'+sourceList[n]+'_extract_agn_deredshift_rebin_header_units_GANDALF_fits.fits')
    hdr = bf[0].header
    emission = bf[2].data
    lam = hdr['crval1'] + hdr['cd1_1']*(N.arange(hdr['naxis1'] - hdr['crpix1']))
    wave = 10**lam
    bestfit = N.load('./int_gandalf_results/best_fit_'+sourceList[n]+'.npy')
    broad = gaussbroad(bestfit[3][0], bestfit[4][0], bestfit[5][0], wave)
    broadp = gaussbroad(bestfit[3][0]+bestfit[3][1], bestfit[4][0]+bestfit[4][1], bestfit[5][0]+bestfit[5][1], wave)
    broadm = gaussbroad(bestfit[3][0]-bestfit[3][2], bestfit[4][0]-bestfit[4][2], bestfit[5][0]-bestfit[5][2], wave)
    fit = gauss(bestfit[0][0], bestfit[1][0], bestfit[2][0],bestfit[3][0], bestfit[4][0], bestfit[5][0], wave)
    fitp = gauss(bestfit[0][0]+bestfit[0][1], bestfit[1][0]+bestfit[1][1], bestfit[2][0]+bestfit[2][1],bestfit[3][0]+bestfit[3][1], bestfit[4][0]+bestfit[4][1], bestfit[5][0]+bestfit[5][1], wave)
    fitm = gauss(bestfit[0][0]-bestfit[0][2], bestfit[1][0]-bestfit[1][2], bestfit[2][0]-bestfit[2][2],bestfit[3][2]-bestfit[3][2], bestfit[4][0]-bestfit[4][2], bestfit[5][0]-bestfit[5][2], wave)
    int_fwhm[n] = calc_fwhm(wave, broad)
    fwhmp = calc_fwhm(wave, broadp)
    fwhmm = calc_fwhm(wave, broadm)
    Err_int_fwhm[n] = N.mean([N.abs(fwhmp-int_fwhm[n]), N.abs(int_fwhm[n]-fwhmm)]) 
    int_lum[n], Err_int_lum[n], int_mbh[n], Err_int_mbh[n] = bhmass(ld, Err_ld, int_fwhm[n], Err_int_fwhm[n], wave, fit, fitp, fitm)
    int_lbol[n] = 8 * (4*N.pi*(ld.to(un.cm).value)**2) * 1E-23 * (299792458/12E-6) * 31.674 * 10**(-ints['w3mag'][n]/2.5)
    Err_int_lbol[n] = int_lbol[n] * ( (8*4*N.pi*Err_ld/ld.value)**2 + ( ((1E-23 * (299792458/12E-6) * 31.674)/-2.5) * N.log(10) * ints['w3sigm'][n] )**2 )**(0.5)

    

# N.save('ints_colours_masses_integrate.npy', ints)
# N.savetxt('ints_colours_masses_integrate.csv', ints, delimiter=',')
# prihdu = F.PrimaryHDU(data=ints)
# hdulist=F.HDUList([prihdu])
# hdulist.writeto('/Users/becky/Projects/int_reduc/ints_colours_masses_int_results.fits')

qint = Table([sourceList, sourceID, ints['col4'], ints['col5'], z, 0.1*z, ur, Err_ur, int_mtot, Err_int_mtot, int_fwhm, Err_int_fwhm, int_lum, Err_int_lum, int_mbh, Err_int_mbh, int_lbol, Err_int_lbol], names=('name', 'ID', 'RA', 'Dec', 'z', 'Err z', 'ur gal', 'Err ur gal', 'stellar mass', 'Err stellar mass', 'FWHM', 'Err FWHM', 'Ha lum', 'Err Ha lum', 'MBH', 'Err MBH', 'L bol', 'Err L bol'))


import glob
sdss = Table.read('sdss_spectra_dr8_dr10_sources_magnitudes_MPA_JHU_mass_WISE.fits', format='fits')
sdss_mtot = N.zeros(len(sdss))
Err_sdss_mtot = N.zeros(len(sdss))
sdss_fwhm = N.zeros(len(sdss))
Err_sdss_fwhm = N.zeros(len(sdss))
sdss_mbh = N.zeros(len(sdss))
Err_sdss_mbh = N.zeros(len(sdss))
sdss_lum = N.zeros(len(sdss))
Err_sdss_lum = N.zeros(len(sdss))
sdss_flux_h_alpha = N.zeros(len(sdss))
sdss_lbol = N.zeros(len(sdss))
Err_sdss_lbol = N.zeros(len(sdss))
sdss_z = N.zeros(len(sdss))
Err_sdss_z = N.zeros(len(sdss))
sdss_ur = N.zeros(len(sdss))
Err_sdss_ur = N.zeros(len(sdss))
sdss_gal_Mr = N.zeros(len(sdss))
Err_sdss_gal_Mr = N.zeros(len(sdss))
for n in range(len(sdss)):
    if n !=73 and n!=35:
        a = glob.glob('./SDSS_RESULTS/spSpec-'+str(sdss['MJD'][n]).zfill(5)+'-'+str(sdss['PlateID'][n]).zfill(4)+'-'+str(sdss['Fiber'][n]).zfill(3)+'_fits.fits')
        if len(a) > 0:
            fit = F.open(a[0])
            bestfit = N.load('./SDSS_RESULTS/spSpec-'+str(sdss['MJD'][n]).zfill(5)+'-'+str(sdss['PlateID'][n]).zfill(4)+'-'+str(sdss['Fiber'][n]).zfill(3)+'_fits.fits_best_fit.npy')
            spec = fit[1].data
            hdr = fit[0].header
            lam = hdr['CRVAL1'] + hdr['CD1_1']*(N.arange(hdr['NAXIS1'] -  hdr['CRPIX1'] + 1))
            wave = (10**lam)/(1+hdr['Z'])
            sdss_ur[n], Err_sdss_ur[n], gal_r, Err_r = calc_ur(sdss['psfMag_u'][n], sdss['psfMagErr_u'][n], sdss['psfMag_r'][n], sdss['psfMagErr_r'][n], sdss['petroMag_u'][n], sdss['petroMagErr_u'][n], sdss['petroMag_r_2'][n], sdss['petroMagErr_r_2'][n])
            if sdss_ur[n] <= 2.1:
                log_m_l = -0.95 + 0.56 * sdss_ur[n]
                log_m_l_err = (0.56*Err_sdss_ur[n])**2
            else:
                log_m_l = -0.16 + 0.18 * sdss_ur[n]
                log_m_l_err = (0.18*Err_sdss_ur[n])**2
            sdss_z[n] = hdr['Z']
            Err_sdss_z[n] = hdr['Z_ERR']
            ld = cosmo.luminosity_distance(hdr['Z'])
            Err_ld = N.mean([(cosmo.luminosity_distance(hdr['Z']+hdr['Z_ERR'])-ld).value, (ld-cosmo.luminosity_distance(hdr['Z']-hdr['Z_ERR'])).value])
            sdss_gal_Mr[n] = gal_r - 5 * (N.log10(ld.value * 1E6) - 1)
            Err_sdss_gal_Mr[n] = (Err_r**2 + ((5*Err_ld)/(ld.value*N.log(10)))**2)**(0.5)
            sdss_mtot[n] = ((4.62 - sdss_gal_Mr[n])/2.5) + log_m_l
            Err_sdss_mtot[n] = ((Err_sdss_gal_Mr[n]/2.5)**2 + (log_m_l_err)**2)**(0.5)
            #sdss_mtot[n] = sdss['lgm_tot_p50'][n]
            sdss_flux_h_alpha[n] = N.max(spec[2500:])
            broad = gaussbroad(bestfit[3][0], bestfit[4][0], bestfit[5][0], wave)
            broadp = gaussbroad(bestfit[3][0]+bestfit[3][1], bestfit[4][0]+bestfit[4][1], bestfit[5][0]+bestfit[5][1], wave)
            broadm = gaussbroad(bestfit[3][0]-bestfit[3][2], bestfit[4][0]-bestfit[4][2], bestfit[5][0]-bestfit[5][2], wave)
            fit = gauss(bestfit[0][0], bestfit[1][0], bestfit[2][0],bestfit[3][0], bestfit[4][0], bestfit[5][0], wave)
            fitp = gauss(bestfit[0][0]+bestfit[0][1], bestfit[1][0]+bestfit[1][1], bestfit[2][0]+bestfit[2][1],bestfit[3][0]+bestfit[3][1], bestfit[4][0]+bestfit[4][1], bestfit[5][0]+bestfit[5][1], wave)
            fitm = gauss(bestfit[0][0]-bestfit[0][2], bestfit[1][0]-bestfit[1][2], bestfit[2][0]-bestfit[2][2],bestfit[3][2]-bestfit[3][2], bestfit[4][0]-bestfit[4][2], bestfit[5][0]-bestfit[5][2], wave)
            sdss_fwhm[n] = calc_fwhm(wave, broad)
            fwhmp = calc_fwhm(wave, broadp)
            fwhmm = calc_fwhm(wave, broadm)
            Err_sdss_fwhm[n] = N.mean([N.abs(fwhmp-sdss_fwhm[n]), N.abs(sdss_fwhm[n]-fwhmm)]) 
            sdss_lum[n], Err_sdss_lum[n], sdss_mbh[n], Err_sdss_mbh[n] = bhmass(ld, Err_ld, sdss_fwhm[n], Err_sdss_fwhm[n], wave, fit, fitp, fitm)
            sdss_lbol[n] = 8 * (4*N.pi*(ld.to(un.cm).value)**2) * 1E-23 * (299792458/12E-6) * 31.674 * 10**(-sdss['w3mag'][n]/2.5)
            Err_sdss_lbol[n] = sdss_lbol[n] * ( (8*4*N.pi*Err_ld/ld.value)**2 + ( ((1E-23 * (299792458/12E-6) * 31.674)/-2.5) * N.log(10) * sdss['w3sigm'][n] )**2 )**(0.5)
        else:
            pass
    else:
        pass

qsdss = Table([sdss['name'], sdss['objID_dr8_1'], sdss['RA_1'], sdss['Dec_1'], sdss_z, Err_sdss_z, sdss_ur, Err_sdss_ur, sdss_mtot, Err_sdss_mtot, sdss_fwhm, Err_sdss_fwhm, sdss_lum, Err_sdss_lum, sdss_mbh, Err_sdss_mbh, sdss_lbol, Err_sdss_lbol], names=('name', 'ID', 'RA', 'Dec', 'z', 'Err z', 'ur gal', 'Err ur gal', 'stellar mass', 'Err stellar mass', 'FWHM', 'Err FWHM', 'Ha lum', 'Err Ha lum', 'MBH', 'Err MBH', 'L bol', 'Err L bol'))


qso = Table.read('qsos_dontneedhst_comparebhmasses_spec_info_WISE.fits', format='fits')
qso_mtot = N.zeros(len(qso))
Err_qso_mtot = N.zeros(len(qso))
qso_fwhm = N.zeros(len(qso))
Err_qso_fwhm = N.zeros(len(qso))
qso_mbh = N.zeros(len(qso))
Err_qso_mbh = N.zeros(len(qso))
qso_lum = N.zeros(len(qso))
Err_qso_lum = N.zeros(len(qso))
qso_flux_h_alpha = N.zeros(len(qso))
qso_lbol = N.zeros(len(qso))
Err_qso_lbol = N.zeros(len(qso))
qso_z = N.zeros(len(qso))
Err_qso_z = N.zeros(len(qso))
qso_ur = N.zeros(len(qso))
Err_qso_ur = N.zeros(len(qso))
qso_gal_Mr = N.zeros(len(qso))
Err_qso_gal_Mr = N.zeros(len(qso))
qso_names = N.zeros(len(qso)).astype(str)


for n in range(len(qso)):
    qso_names[n] = 'qso'+str(n)
    a = glob.glob('./QSO_RESULTS/spSpec-'+str(qso['mjd'][n]).zfill(5)+'-'+str(qso['plate'][n]).zfill(4)+'-'+str(qso['fiberID'][n]).zfill(3)+'_fits.fits')
    if len(a) > 0:
        fit = F.open(a[0])
        bestfit = N.load('./QSO_RESULTS/spSpec-'+str(qso['mjd'][n]).zfill(5)+'-'+str(qso['plate'][n]).zfill(4)+'-'+str(qso['fiberID'][n]).zfill(3)+'_fits.fits_best_fit.npy')
        spec = fit[1].data
        hdr = fit[0].header
        lam = hdr['CRVAL1'] + hdr['CD1_1']*(N.arange(hdr['NAXIS1'] -  hdr['CRPIX1'] + 1))
        wave = (10**lam)/(1+hdr['Z'])
        qso_ur[n], Err_qso_ur[n], gal_r, Err_r = calc_ur(qso['psfMag_u'][n], qso['psfMagErr_u'][n], qso['psfMag_r'][n], qso['psfMagErr_r'][n], qso['petroMag_u'][n], qso['petroMagErr_u'][n], qso['petroMag_r'][n], qso['petroMagErr_r'][n])
        if qso_ur[n] <= 2.1:
            log_m_l = -0.95 + 0.56 * qso_ur[n]
            log_m_l_err = (0.56*Err_qso_ur[n])**2
        else:
            log_m_l = -0.16 + 0.18 * qso_ur[n]
            log_m_l_err = (0.18*Err_qso_ur[n])**2
        qso_z[n]= hdr['Z']
        Err_qso_z[n] = hdr['Z_ERR']
        ld = cosmo.luminosity_distance(hdr['Z'])
        Err_ld = N.mean([(cosmo.luminosity_distance(hdr['Z']+hdr['Z_ERR'])-ld).value, (ld-cosmo.luminosity_distance(hdr['Z']-hdr['Z_ERR'])).value])
        qso_gal_Mr[n] = gal_r - 5 * (N.log10(ld.value * 1E6) - 1)
        Err_qso_gal_Mr[n] = (Err_r**2 + ((5*Err_ld)/(ld.value*N.log(10)))**2)**(0.5)
        qso_mtot[n] = ((4.62 - qso_gal_Mr[n])/2.5) + log_m_l
        Err_qso_mtot[n] = ((Err_qso_gal_Mr[n]/2.5)**2 + (log_m_l_err)**2)**(0.5)
        #qso_mtot[n] = qso['lgm_tot_p50'][n]
        qso_flux_h_alpha[n] = N.max(spec[2500:])
        broad = gaussbroad(bestfit[3][0], bestfit[4][0], bestfit[5][0], wave)
        broadp = gaussbroad(bestfit[3][0]+bestfit[3][1], bestfit[4][0]+bestfit[4][1], bestfit[5][0]+bestfit[5][1], wave)
        broadm = gaussbroad(bestfit[3][0]-bestfit[3][2], bestfit[4][0]-bestfit[4][2], bestfit[5][0]-bestfit[5][2], wave)      
        fit = gauss(bestfit[0][0], bestfit[1][0], bestfit[2][0],bestfit[3][0], bestfit[4][0], bestfit[5][0], wave)
        fitp = gauss(bestfit[0][0]+bestfit[0][1], bestfit[1][0]+bestfit[1][1], bestfit[2][0]+bestfit[2][1],bestfit[3][0]+bestfit[3][1], bestfit[4][0]+bestfit[4][1], bestfit[5][0]+bestfit[5][1], wave)
        fitm = gauss(bestfit[0][0]-bestfit[0][2], bestfit[1][0]-bestfit[1][2], bestfit[2][0]-bestfit[2][2],bestfit[3][2]-bestfit[3][2], bestfit[4][0]-bestfit[4][2], bestfit[5][0]-bestfit[5][2], wave)    
        qso_fwhm[n] = calc_fwhm(wave, broad)
        fwhmp = calc_fwhm(wave, broadp)
        fwhmm = calc_fwhm(wave, broadm)
        Err_qso_fwhm[n] = N.mean([N.abs(fwhmp-qso_fwhm[n]), N.abs(qso_fwhm[n]-fwhmm)])
        qso_lum[n], Err_qso_lum[n], qso_mbh[n], Err_qso_mbh[n] = bhmass(ld, Err_ld, qso_fwhm[n], Err_qso_fwhm[n], wave, fit, fitp, fitm)
        qso_lbol[n] = 8 * (4*N.pi*(ld.to(un.cm).value)**2) * 1E-23 * (299792458/12E-6) * 31.674 * 10**(-qso['w3mag'][n]/2.5)
        Err_qso_lbol[n] = qso_lbol[n] * ( (8*4*N.pi*Err_ld/ld.value)**2 + ( ((1E-23 * (299792458/12E-6) * 31.674)/-2.5) * N.log(10) * qso['w3sigm'][n] )**2 )**(0.5)
    else:
        pass



qqso = Table([qso_names, qso['objID_1'], qso['RA_deg'], qso['Dec_deg'], qso_z, Err_qso_z, qso_ur, Err_qso_ur, qso_mtot, Err_qso_mtot, qso_fwhm, Err_qso_fwhm, qso_lum, Err_qso_lum, qso_mbh, Err_qso_mbh, qso_lbol, Err_qso_lbol], names=('name', 'ID', 'RA', 'Dec', 'z', 'Err z', 'ur gal', 'Err ur gal', 'stellar mass', 'Err stellar mass', 'FWHM', 'Err FWHM', 'Ha lum', 'Err Ha lum', 'MBH', 'Err MBH', 'L bol', 'Err L bol'))


results = vstack([qint, qsdss, qqso])
#results.write('all_galaxies_z_ur_mtot_mbh_lbol_lHa_some_zero_measurements.fits', format='fits')

results = results[N.where(results['MBH'] > 0)]
results = results[N.isnan(results['ur gal']) == False]
print len(results)
#results.write('all_galaxies_z_ur_mtot_mbh_lbol_lHa_with_measurements.fits', format='fits')

def calc_z(obs, rest):
    return round((obs/rest) -1, 2)

k=0
j = N.random.randint(9, len(sdss), 5)
fig = P.figure(figsize=(10,13), frameon=False, edgecolor='None')
for n in j:
    if n ==73 or n==35:
        n+=1
    else:
        pass
    a = glob.glob('./SDSS_RESULTS/spSpec-'+str(sdss['MJD'][n]).zfill(5)+'-'+str(sdss['PlateID'][n]).zfill(4)+'-'+str(sdss['Fiber'][n]).zfill(3)+'_fits.fits')
    if len(a) > 0:
        fit = F.open(a[0])
        bestfit = N.load('./SDSS_RESULTS/spSpec-'+str(sdss['MJD'][n]).zfill(5)+'-'+str(sdss['PlateID'][n]).zfill(4)+'-'+str(sdss['Fiber'][n]).zfill(3)+'_fits.fits_best_fit.npy')
        spec = fit[0].data
        hdr = fit[0].header
        lam = hdr['CRVAL1'] + hdr['CD1_1']*(N.arange(hdr['NAXIS1'] -  hdr['CRPIX1'] + 1))
        rest_wave = (10**lam)/(1+hdr['Z'])
        obs_wave = 10**lam
    
        ax = fig.add_subplot(5,1,k+1)
        ax.plot(rest_wave, spec, color='k')
        ax.minorticks_on()
        ax.tick_params(axis='y', which='minor', left='off', right='off')
        ax.set_xlim(5900, 7600)
        ax.text(0.925, 0.85, 'z = '+str(calc_z(obs_wave[0], rest_wave[0])), ha='right', transform=ax.transAxes)
        ax.text(0.025, 0.85, sdss['name'][n], ha='left', transform=ax.transAxes)
        if k == 4:
            ax.set_xlabel(r'$\rm{Rest}$-$\rm{frame }$ $\rm{Wavelength}$,  $\lambda_{e} [\AA]$')
        else:
            pass
        if k == 2:
            ax.set_ylabel(r'$\rm{Flux }$ $[10^{-17} \rm{erg}$ $\rm{cm}^{-2}$ $\rm{s}^{-1}$ $\AA^{-1}]$')
        else:
            pass
        ax12 = ax.twiny()
        ax12.set_xlim(5900, 7600)
        if k == 0:
            ax12.set_xlabel(r'$\rm{Observed }$ $\rm{Wavelength }$, $\lambda_{o} [\AA]$')
        else:
            pass
        rest_labels = ax.get_xticks()
        ax12.set_xticks(rest_labels[1:-1])
        ax12.set_xticklabels((N.interp(rest_labels[:-1], rest_wave, obs_wave).astype(int)))
        ax12.set_xlim(5900, 7600)
        ax12.minorticks_on()
        k+=1
fig.subplots_adjust(hspace=0.5)
#fig.tight_layout()
fig.savefig('sample_sdss_spectra.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)


k=0
j = N.random.randint(0, len(qso), 5)
fig = P.figure(figsize=(5,13), frameon=False, edgecolor='None')
for n in j:
    a = glob.glob('./QSO_RESULTS/spSpec-'+str(qso['mjd'][n]).zfill(5)+'-'+str(qso['plate'][n]).zfill(4)+'-'+str(qso['fiberID'][n]).zfill(3)+'_fits.fits')
    while len(a) <=0:
        n +=1
        a = glob.glob('./QSO_RESULTS/spSpec-'+str(qso['mjd'][n]).zfill(5)+'-'+str(qso['plate'][n]).zfill(4)+'-'+str(qso['fiberID'][n]).zfill(3)+'_fits.fits')
    fit = F.open(a[0])
    bestfit = N.load('./QSO_RESULTS/spSpec-'+str(qso['mjd'][n]).zfill(5)+'-'+str(qso['plate'][n]).zfill(4)+'-'+str(qso['fiberID'][n]).zfill(3)+'_fits.fits_best_fit.npy')
    spec = fit[0].data
    hdr = fit[0].header
    lam = hdr['CRVAL1'] + hdr['CD1_1']*(N.arange(hdr['NAXIS1'] -  hdr['CRPIX1'] + 1))
    rest_wave = (10**lam)/(1+hdr['Z'])
    obs_wave = 10**lam

    ax = fig.add_subplot(5,1,k+1)
    ax.plot(rest_wave, spec, color='k')
    ax.minorticks_on()
    ax.tick_params(axis='y', which='minor', left='off', right='off')
    ax.set_xlim(6350, 6850)
    ax.text(0.925, 0.9, 'z = '+str(calc_z(obs_wave[0], rest_wave[0])), ha='right', transform=ax.transAxes)
    if k == 4:
        ax.set_xlabel(r'$\rm{Rest}$-$\rm{frame }$ $\rm{Wavelength}$,  $\lambda_{e} [\AA]$')
    else:
        pass
    if k == 2:
        ax.set_ylabel(r'$\rm{Flux }$ $[10^{-17} \rm{erg}$ $\rm{cm}^{-2}$ $\rm{s}^{-1}$ $\AA^{-1}]$')
    else:
        pass
    ax12 = ax.twiny()
    ax12.set_xlim(6350, 6850)
    if k == 0:
        ax12.set_xlabel(r'$\rm{Observed }$ $\rm{Wavelength }$, $\lambda_{o} [\AA]$')
    else:
        pass
    rest_labels = ax.get_xticks()
    ax12.set_xticks(rest_labels[1:-1])
    ax12.set_xticklabels((N.interp(rest_labels[:-1], rest_wave, obs_wave).astype(int)))
    ax12.set_xlim(6350, 6850)
    ax12.minorticks_on()
    k+=1
fig.subplots_adjust(hspace=0.5)
#fig.tight_layout()
fig.savefig('sample_qso_spectra.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)

# fig = P.figure(figsize=(5,5))
# ax = fig.add_subplot(1,1,1)
# ax.scatter(sdss['published_BHmass'], sdss_mbh, marker='x', c='None', edgecolor='k', s=30, label=r'$\rm{SDSS }$ $\rm{spectra }$')
# ax.plot(N.linspace(5.5, 9.5, 100), N.linspace(5.5, 9.5, 100), linestyle='dashed', c='k')
# ax.set_xlabel(r'$\rm{published}$ $log_{10}(M_{BH}/M_{\odot})$')
# ax.set_ylabel(r'$log_{10}(M_{BH}/M_{\odot})$')
# ax.minorticks_on()
# ax.set_xlim(7.0, 9.5)
# ax.set_ylim(7.0, 9.5)
# ax.legend(frameon=False, loc=2, fontsize=12)
# #P.tight_layout()
# P.savefig('./published_vs_calculated_BH_mass_sdss_integrated_narrow_broad.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)

results = Table.read('all_galaxies_z_ur_mtot_mbh_lbol_lHa_with_measurements.fits', format='fits')

mtot = N.linspace(10**8, 10**13, 20)
mbh = 8.2 + 1.12 * N.log10(mtot/1E11)
mbhplus = 8.3 + 1.18 * N.log10(mtot/1E11)
mbhminus = 8.1 + 1.06 * N.log10(mtot/1E11)

yhr = Column(data=N.log10([3E9, 1.4E7, 1E8, 4.3E8, 5.2E8, 5.3E8, 3.3E8, 1.4E7, 3.7E7, 2.5E9, 4.5E7, 2.5E6, 4.4E7, 1.4E7, 1.0E9, 2.1E8, 1.0E8, 1.6E7, 1.9E8, 3.1E8, 3.0E8, 1.1E8, 5.6E7, 1.0E9, 2.0E9, 1.7E8, 2.4E8, 1.3E7, 3.5E6, 3.7E6]), name='Haring and Rix MBH', unit=un.MsolMass)
yhrerr = Column(data=([1E9, 1E7, 0.55E8, 2.45E8, 1.05E8, 3E8, 2.8E8, 0.45E7, 1.6E7, 0.45E9, 3.25E7, 0.5E6, 0.5E7, 1.2E7, 0.8E9, 0.8E8, 0.5E8, 0.11E7, 0.8E8, 1.2E8, 1.35E8, 0.6E8, 0.5E7, 0.85E9, 0.75E9, 0.11E8, 0.9E8, 0.55E7, 1.25E6, 1.5E6]/(10**yhr.data * N.log(10))), name='Haring and Rix Err MBH', unit =un.MsolMass)
xhr = Column(data=N.log10([6E11, 2.3E10, 6.8E10, 3.6E11, 3.6E11, 5.6E11, 2.9E11, 6.2E9, 1.3E11, 2.9E11, 3.7E10, 8E8, 6.9E10, 7.6E10, 1.2E11, 6.8E10, 1.6E10, 2E10, 9.7E10, 1.3E11, 1.2E10, 9.2E10, 4.4E10, 2.7E11, 4.9E11, 1.1E11, 3.7E10, 1.5E10, 7E9, 1.1E10]), name='Haring and Rix bulge mass', unit=un.MsolMass)
xhrerr = Column(data=0.18*N.log10(xhr.data), name='Haring and Rix Err bulge mass', unit =un.MsolMass)

y = results['MBH']
yerr = results['Err MBH']
x = results['stellar mass']
xerr = results['Err stellar mass']

 
#Load all SDSS galaxies, some of which have Simard bulge to total ratio fits. The [5:] is to disregard the INT galaxies at the top
bt = Table.read('all_101_galaxies_z_ur_mtot_mbh_lbol_etc_with_Btot_from_Simard_et_al.fits', format='fits')[5:]
# Set those which don't have Simard B/T to have a B/T=1 as an upper limit
bt['(B/T)r'][N.isnan(bt['(B/T)r'])] = 1
bt['e_(B/T)r'][N.isnan(bt['e_(B/T)r'])] = N.mean(bt['e_(B/T)r'][N.isfinite(bt['e_(B/T)r'])])
#Calculate bulgemass upper limits of SDSS galaxies
bulgemass = N.log10(bt['(B/T)r']*10**bt['stellar mass'])
err_bulgemass = N.sqrt((bt['e_(B/T)r']/bt['(B/T)r'])**2 + (bt['Err stellar mass']/bt['stellar mass'])**2)

int_bt = N.array([1, 1, 0.47, 1, 0.3])
int_bulgemass = N.log10(int_bt*10**int_mtot)
err_int_bulgemass = N.sqrt( (0.2)**2 + (Err_int_mtot/int_mtot)**2 )

brooke_mtot = N.array([10.05, 10.03])
err_brooke_mtot = N.array([0.1, 0.1])
brooke_mbh = N.array([7.1, 6.6])
err_brooke_mbh = N.array([0.13, 0.14])
brooke_lbol = N.array([44.1, 43.4])
err_brooke_lbol=N.array([0.3, 0.3])
brooke_BT = N.array([0.011, 0.022])
brooke_bulgemass = N.log10(brooke_BT*10**brooke_mtot)
err_brooke_bulgemass = N.sqrt( (N.array([0.15, 0.2]))**2 + (N.array([0.15,0.12]))**2 )


bt_mbhs = N.append(brooke_mbh, N.append(bt['MBH'], int_mbh[N.where(int_bt > 0)]))
err_bt_mbhs = N.append(err_brooke_mbh, N.append(bt['Err MBH'], Err_int_mbh[N.where(int_bt > 0)]))
bt_bulgemass = N.append(brooke_bulgemass, N.append(bulgemass, int_bulgemass[N.where(int_bt > 0)]))
err_bt_bulgemass = N.append(err_brooke_bulgemass, N.append(err_bulgemass, err_int_bulgemass[N.where(int_bt > 0)]))

#Set delta for linmix array which says what are upper limits. 1 is a measured value. 0 is an upper limit.
#Brooke's galaxies have measured B/T, SDSS are all upper limits and INT values have 2 measured, 3 upper limits (set to a B/T =1) 
delta = N.append(N.ones(len(brooke_mtot)), N.append(N.zeros(len(bt)), N.array([0,0,1,0,1])))

P.figure(figsize=(6,3))
P.hist(N.append(brooke_BT, N.append(bt['(B/T)r'], int_bt)), range=(0,1), bins=15, histtype='step', color='k')
P.xlabel(r'$[\rm{B}/\rm{T}]_r$')
P.ylabel(r'$\rm{number}$')
P.ylim(0,15)
P.minorticks_on()
P.tight_layout()
P.savefig('bulge_to_total_r_ratio_hist_with_INT_simmons13.pdf', frameon=False, transparent=True)


# print 'emceeing...'
# start = [1.12, -3, -0.7]
# ndim, nwalkers, burnin, nsteps = 3, 50, 500, 100
# p0 = [start + 1e-4*N.random.randn(ndim) for i in range(nwalkers)]
# import emcee
# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(bulgemass, bt['MBH'], err_bulgemass, bt['Err MBH']))
# pos, prob, state = sampler.run_mcmc(p0, burnin)
# samples = sampler.chain[:,:,:].reshape((-1,ndim))
# N.save('mbulge_mbh_fit_yerr_burn_in.npy', samples)
# #walker_plot(samples, nwalkers, burnin)
# sampler.reset()
# print 'RESET', pos
# # main sampler run 
# sampler.run_mcmc(pos, nsteps)
# samples = sampler.chain[:,:,:].reshape((-1,ndim))
# N.save('mbulge_mbh_fit_yerr_samples.npy', samples)
# #walker_plot(samples, nwalkers, nsteps)
# import triangle
# #fig = triangle.corner(samples, labels=[r'$m$', r'$b$'])
# #fig.savefig('mbh_mtot_fit_line_triangle.pdf')
# bf = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*N.percentile(samples, [34,50,66],axis=0)))
# N.save('mbulge_mbh_fit_yerr_best_fit.npy', bf)

# samplesbt = N.load('mbulge_mbh_fit_xerr_yerr_samples.npy')
# shbt = N.sort(samplesbt, axis=0)
# bestreshbt = shbt[0.5*len(shbt), 0] * N.log10(mtot) + shbt[0.5*len(shbt), 1]
# plusreshbt = shbt[0.84*len(shbt), 0] * N.log10(mtot) + shbt[0.16*len(shbt), 1]
# minusreshbt = shbt[0.16*len(shbt), 0] * N.log10(mtot) + shbt[0.84*len(shbt), 1]
xs = N.linspace(0, 15, 100)


#Use linmix to fit to haring and rix x and ys
lmhr  = linmix.LinMix(xhr, yhr, xhrerr, yhrerr, K=2)
lmhr.run_mcmc(silent=False, miniter=5000, maxiter=10000)
hrysb = lmhr.chain['alpha'].mean() + xs * lmhr.chain['beta'].mean()
hrysp = (lmhr.chain['alpha'].mean()+3*lmhr.chain['alpha'].std()) + xs * (lmhr.chain['beta'].mean()-3*lmhr.chain['beta'].std())  
hrysm = (lmhr.chain['alpha'].mean()-3*lmhr.chain['alpha'].std()) + xs * (lmhr.chain['beta'].mean()+3*lmhr.chain['beta'].std())

#Use linmix to fit to stellar mass and MBH relation from DISKDOM sample
lmr  = linmix.LinMix(results['stellar mass'], results['MBH'], results['Err stellar mass'], results['Err MBH'], K=2)
lmr.run_mcmc(silent=False, miniter=5000, maxiter=10000)
rysb = lmr.chain['alpha'].mean() + xs * lmr.chain['beta'].mean()
rysp = (lmr.chain['alpha'].mean()+3*lmr.chain['alpha'].std()) + xs * (lmr.chain['beta'].mean()-3*lmr.chain['beta'].std())  
rysm = (lmr.chain['alpha'].mean()-3*lmr.chain['alpha'].std()) + xs * (lmr.chain['beta'].mean()+3*lmr.chain['beta'].std())

#Use linmix to fit to upper limits on bulgemass from DISKDOM sample (inlcuding those with no measurement from Simard set to B/T = 1)
lmupl  = linmix.LinMix(bt_mbhs, bt_bulgemass, err_bt_mbhs, err_bt_bulgemass, delta=delta, K=2)
lmupl.run_mcmc(silent=False, miniter=5000, maxiter=10000) 
uplysb = lmupl.chain['alpha'].mean() + xs * lmupl.chain['beta'].mean()
uplysp = (lmupl.chain['alpha'].mean()+3*lmupl.chain['alpha'].std()) + xs * (lmupl.chain['beta'].mean()-3*lmupl.chain['beta'].std())
uplysm = (lmupl.chain['alpha'].mean()-3*lmupl.chain['alpha'].std()) + xs * (lmupl.chain['beta'].mean()+3*lmupl.chain['beta'].std())

#Use linmix to fit to bulgemass from DISKDOM sample assuming no measurement is an upper limit
lmnupl  = linmix.LinMix(bt_mbhs, bt_bulgemass, err_bt_mbhs, err_bt_bulgemass, K=2)
lmnupl.run_mcmc(silent=False, miniter=5000, maxiter=10000) 
nuplysb = lmnupl.chain['alpha'].mean() + xs * lmnupl.chain['beta'].mean()
nuplysp = (lmnupl.chain['alpha'].mean()+3*lmnupl.chain['alpha'].std()) + xs * (lmnupl.chain['beta'].mean()-3*lmnupl.chain['beta'].std())
nuplysm = (lmnupl.chain['alpha'].mean()-3*lmnupl.chain['alpha'].std()) + xs * (lmnupl.chain['beta'].mean()+3*lmnupl.chain['beta'].std())


# lmnupl  = linmix.LinMix(N.append(brooke_mbh, int_mbh[N.where(int_bt > 0)]), N.append(brooke_bulgemass, int_bulgemass[N.where(int_bt > 0)]), N.append(err_brooke_mbh, Err_int_mbh[N.where(int_bt > 0)]), N.append(err_brooke_bulgemass, err_int_bulgemass[N.where(int_bt > 0)]), delta=N.ones(6), K=2)
# lmnupl.run_mcmc(silent=False) 
# nuplysb = lmnupl.chain['alpha'].mean() + xs * lmnupl.chain['beta'].mean()
# nuplysp = (lmnupl.chain['alpha'].mean()+3*lmnupl.chain['alpha'].std()) + xs * (lmnupl.chain['beta'].mean()-3*lmnupl.chain['beta'].std())
# nuplysm = (lmnupl.chain['alpha'].mean()-3*lmnupl.chain['alpha'].std()) + xs * (lmnupl.chain['beta'].mean()+3*lmnupl.chain['beta'].std())

# lmjupl  = linmix.LinMix(bt['MBH'], bulgemass, bt['Err MBH'], err_bulgemass ,delta=N.zeros(len(bulgemass)), K=2)
# lmjupl.run_mcmc(silent=False) 
# juplysb = lmjupl.chain['alpha'].mean() + xs * lmjupl.chain['beta'].mean()
# juplysp = (lmjupl.chain['alpha'].mean()+3*lmjupl.chain['alpha'].std()) + xs * (lmjupl.chain['beta'].mean()-3*lmjupl.chain['beta'].std())
# juplysm = (lmjupl.chain['alpha'].mean()-3*lmjupl.chain['alpha'].std()) + xs * (lmjupl.chain['beta'].mean()+3*lmjupl.chain['beta'].std())

# lmhrupl  = linmix.LinMix(N.append(bt_mbhs, yhr), N.append(bt_bulgemass, xhr), N.append(err_bt_mbhs, yhrerr), N.append(N.ones(len(bt_mbhs)), N.ones(len(xhr))), K=10)
# lmhrupl.run_mcmc(silent=False) 
# uplhrysb = lmhrupl.chain['alpha'].mean() + xs * lmhrupl.chain['beta'].mean()
# uplhrysp = (lmhrupl.chain['alpha'].mean()+3*lmhrupl.chain['alpha'].std()) + xs * (lmhrupl.chain['beta'].mean()-3*lmhrupl.chain['beta'].std())
# uplhrysm = (lmhrupl.chain['alpha'].mean()-3*lmhrupl.chain['alpha'].std()) + xs * (lmhrupl.chain['beta'].mean()+3*lmhrupl.chain['beta'].std())




fig = P.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)
ax.plot(xs, rysb, color='k', linestyle='solid')
ax.fill_between(xs, y1=rysp, y2=rysm, color='k', alpha=0.1)
ax.scatter(brooke_mtot, brooke_mbh, marker='o', c='None', edgecolor='k', s=30, label=r'$\rm{Simmons }$ $\rm{et }$ $\rm{al. }$ $\rm{(2013)}$')
ax.errorbar(brooke_mtot, brooke_mbh, xerr = err_brooke_mtot, yerr=err_brooke_mbh, ecolor='k', capthick=1, fmt='None', fill_style='None', alpha=0.5)
ax.errorbar(int_mtot, int_mbh, xerr=Err_int_mtot, yerr=Err_int_mbh, marker='None', fmt='None', ecolor='k', alpha=0.4)
ax.scatter(int_mtot, int_mbh, marker='s', c='b', edgecolor ='b', s=30, label=r'$\rm{INT }$ $\rm{spectra}$')
ax.errorbar(sdss_mtot, sdss_mbh, xerr=Err_sdss_mtot, yerr=Err_sdss_mbh, marker='None', fmt='None', ecolor='k', alpha=0.4)
ax.scatter(sdss_mtot, sdss_mbh, marker='x', c='k', s=30, label=r'$\rm{SDSS}$ $\rm{spectra}$')
ax.errorbar(qso_mtot, qso_mbh, xerr=Err_qso_mtot, yerr=Err_qso_mbh, marker='None', fmt='None', ecolor='k', alpha=0.4)
ax.scatter(qso_mtot, qso_mbh, marker='x', c='k', s=30)
ax.plot(xs, hrysb, color='r', linestyle='dashed', label = r'$\rm{Haring }$ $\rm{\& }$  $\rm{Rix }$ $\rm{(2004)}$ $\rm{fit}$')
ax.fill_between(xs, y1=hrysp, y2=hrysm, color='r', alpha=0.1)
# ax.plot(N.log10(mtot), bestreshr, linestyle='dashed', c='k', label = r'$\rm{Haring }$ $\rm{\& }$  $\rm{Rix }$ $\rm{2004}$')
# ax.plot(N.log10(mtot), plusreshr, linestyle='-.', c='k')
# ax.plot(N.log10(mtot), minusreshr, linestyle='-.', c='k')
ax.set_xlabel(r'$\log_{10}(M_{*}/\rm{M}_{\odot})$')
ax.set_ylabel(r'$\log_{10}(M_{BH}/\rm{M}_{\odot})$')
ax.minorticks_on()
ax.set_xlim(9.4, 11.6)
ax.set_ylim(5.5, 10.5)
ax.legend(frameon=False, loc=2, fontsize=12)
P.tight_layout()
P.subplots_adjust(wspace=0.0)
P.savefig('mass_bh_total_mass_fit_linmix_fit.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)

fig = P.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)
ax.plot(uplysb, xs, color='k')
ax.fill_betweenx(xs, x1=uplysp, x2=uplysm, color='k', alpha=0.1)
ax.plot(nuplysb, xs, color='k', linestyle='dashed')
#ax.fill_betweenx(xs, x1=nuplysp, x2=nuplysm, color='k', alpha=0.1, hatch='-')
#ax.plot(juplysb, xs, color='k', linestyle='dotted', label='Fit to only upper limits')
#ax.fill_betweenx(xs, x1=juplysp, x2=juplysm, color='k', alpha=0.1, hatch='.')
#ax.plot(uplhrysb, xs, color='k', linestyle='dashdot', label='Fit to all our data, no limits + HR04 data')
#ax.fill_betweenx(xs, x1=uplhrysp, x2=uplhrysm, color='k', alpha=0.1, hatch='/')
ax.errorbar(bulgemass, bt['MBH'], xerr = 0.2, xuplims=True, ecolor='k', alpha=0.5, capthick=1, fmt='None', fill_style='None', label=r'$\rm{SDSS}$ $\rm{limits}$')
ax.errorbar(11.9, 10, xerr = N.mean(err_bt_bulgemass), yerr=N.mean(err_bt_mbhs), ecolor='k', fmt='None', alpha=0.5)
ax.errorbar(int_bulgemass, int_mbh, xerr = err_int_bulgemass, yerr=Err_int_mbh, ecolor='k', alpha=0.5, capthick=1, fmt='None', fill_style='None')
ax.scatter(int_bulgemass, int_mbh, marker='s', c='b', edgecolor='b', s=30, label=r'$\rm{INT }$ $\rm{spectra}$')
ax.errorbar(brooke_bulgemass, brooke_mbh, xerr = err_brooke_mbh, yerr=err_brooke_mbh, ecolor='k', capthick=1, fmt='None', fill_style='None', alpha=0.5)
ax.scatter(brooke_bulgemass, brooke_mbh, marker='o', c='None', edgecolor='k', s=30, label=r'$\rm{Simmons }$ $\rm{et }$ $\rm{al. }$ $\rm{(2013)}$')
ax.scatter(xhr, yhr, marker='o', c='None', edgecolor='r', label = r'$\rm{Haring }$ $\rm{\& }$  $\rm{Rix }$ $\rm{(2004)}$')
ax.errorbar(xhr, yhr, xerr=xhrerr, yerr=yhrerr, marker='o', color='None', ecolor='k', alpha=0.3)
ax.plot(xs, hrysb, color='r', linestyle='dashed', label = r'$\rm{Haring }$ $\rm{\& }$  $\rm{Rix }$ $\rm{(2004)}$ $\rm{fit}$')
ax.fill_between(xs, y1=hrysp, y2=hrysm, color='r', alpha=0.1)
ax.set_xlabel(r'$\log_{10}(M_{bulge}/\rm{M}_{\odot})$')
ax.minorticks_on()
ax.set_xlim(7.9, 12.1)
ax.set_ylim(5.5, 10.5)
ax.set_ylabel(r'$\log_{10}(M_{BH}/\rm{M}_{\odot})$')
ax.legend(frameon=False, loc=2, fontsize=12)
P.tight_layout()
P.subplots_adjust(wspace=0.0)
P.savefig('mass_bh_bulge_limits_INT_simmons13_measurements_linmix_fit_no_simard_bt_set_to_1.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)


# qsor = Table.read('shen_2011_matched_redshif.fits', format='fits')

# sdss_mdot*(Err_sdss_lbol/sdss_lbol)/(sdss_mdot*N.log(10))

# fig = P.figure(figsize=(5,5))
# ax = fig.add_subplot(1,1,1)
# ax.errorbar(qsor['LOGBH'], qsor['LOGLBOL'], xerr=qsor['LOGBH_ERR'], yerr=qsor['LOGLBOL_ERR'], marker='None', fmt='None', ecolor='k', alpha=0.1)
# ax.scatter(qsor['LOGBH'], qsor['LOGLBOL'], marker='^', c='None', edgecolor='g', s=20, label=r'$\rm{Shen}$ $\rm{et }$ $\rm{al. }$ $\rm{(2011)}$', alpha=0.8)
# ax.errorbar(int_mbh, N.log10(int_lbol), xerr=Err_int_mbh, yerr=Err_int_lbol/(int_lbol*N.log(10)), marker='None', fmt='None', ecolor='k', alpha=0.4)
# ax.scatter(int_mbh, N.log10(int_lbol), marker='s', c='b', edgecolor='b', s=30, label=r'$\rm{INT }$ $\rm{spectra}$')
# ax.errorbar(sdss_mbh, N.log10(sdss_lbol), xerr=Err_sdss_mbh, yerr=Err_sdss_lbol/(sdss_lbol*N.log(10)), marker='None', fmt='None', ecolor='k', alpha=0.4)
# ax.scatter(sdss_mbh, N.log10(sdss_lbol), marker='x', c='k', s=30, label=r'$\rm{SDSS}$ $\rm{spectra}$')
# ax.errorbar(qso_mbh, N.log10(qso_lbol), xerr=Err_qso_mbh, yerr=Err_qso_lbol/(qso_lbol*N.log(10)), marker='None', fmt='None', ecolor='k', alpha=0.4)
# ax.scatter(qso_mbh, N.log10(qso_lbol), marker='x', c='k', s=30)
# ax.scatter(brooke_mbh, brooke_lbol, marker='o', c='None', edgecolor='k', s=30, label=r'$\rm{Simmons }$ $\rm{et }$ $\rm{al. }$ $\rm{(2013)}$')
# ax.errorbar(brooke_mbh, brooke_lbol, xerr = err_brooke_mbh, yerr=err_brooke_lbol, ecolor='k', capthick=1, fmt='None', fill_style='None', alpha=0.5)
# line1 = ax.plot(mbh, N.log10(0.01*1.26E38*(10**mbh)), linestyle=':', c='0.3')
# line2 = ax.plot(mbh, N.log10(0.1*1.26E38*(10**mbh)), linestyle='-.', c='0.3')
# line3 = ax.plot(mbh, N.log10(1.26E38*(10**mbh)), linestyle='dashed', c='0.3')
# ax.text(9.25, 45.7, r'$\lambda_{Edd}=0.01$', rotation=0, color='0.3')
# ax.text(8.5, 46, r'$\lambda_{Edd}=0.1$', rotation=0, color='0.3')
# ax.text(7.75, 46.3, r'$\lambda_{Edd}=1$', rotation=0, color='0.3')
# ax.set_xlabel(r'$log_{10}(M_{BH}/M_{\odot})$')
# ax.set_ylabel(r'$log_{10}(L_{bol}$ $[\rm{erg}$ $\rm{s}^{-1}])$')
# ax.minorticks_on()
# ax.set_xlim(5.5, 10.5)
# ax.set_ylim(43, 46.5)
# ax.legend(frameon=False, loc=4, fontsize=12)
# #P.tight_layout()
# P.savefig('./mass_bh_bol_luminosity_with_all_errors_shen_11.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)



#Lsol = 3.846E33*un.g*un.cm**2/un.s**3
#results['L bol'].unit = un.erg/un.s
#mdot = (results['L bol']/(0.15*(c.to(un.cm/un.s))**2)).to(un.solMass/un.yr)

# brooke_mdot = (brooke_lbol*(un.erg/un.s)/(0.15*(c.to(un.cm/un.s))**2)).to(un.solMass/un.yr)
# qsor_mdot = ((10**qsor['LOGLBOL'].data.data)*(un.erg/un.s)/(0.15*(c.to(un.cm/un.s))**2)).to(un.solMass/un.yr)
# int_mdot = (int_lbol*(un.erg/un.s)/(0.15*(c.to(un.cm/un.s))**2)).to(un.solMass/un.yr)
# sdss_mdot = (sdss_lbol*(un.erg/un.s)/(0.15*(c.to(un.cm/un.s))**2)).to(un.solMass/un.yr)
# int_mdot = (int_lbol*(un.erg/un.s)/(0.15*(c.to(un.cm/un.s))**2)).to(un.solMass/un.yr)
# qso_mdot = (qso_lbol*(un.erg/un.s)/(0.15*(c.to(un.cm/un.s))**2)).to(un.solMass/un.yr)


# mbh_edd = results['MBH']/((results['L bol']/Lsol)/3E4)

# mseed=1E5
# t = 1E6
# time_seed = N.zeros(len(results))

# for n in range(len(results)):
#     current_mbh = results['MBH'][n]
#     tstep=0
#     while current_mbh > mbh_edd[n].value:
#         current_mbh -= mdot[n].value*t
#         tstep+=1
#         if current_mbh <= mseed:
#             time_seed[n] = t*tstep
#         else:
#             pass
#     while current_mbh > mseed:
#         print 'assuming eddingtonI like '
#         current_mdot = ((3E4*(current_mbh)*Lsol)/(0.15*(c.to(un.cm/un.s))**2)).to(un.solMass/un.yr)
#         current_mbh -= current_mdot*t
#         tstep+=1
#     time_seed[n] = t*tstep




# fig = P.figure(figsize=(5,5))
# ax = fig.add_subplot(1,1,1)
# ax.errorbar(qsor['LOGBH'], N.log10(qsor_mdot.value), xerr=qsor['LOGBH_ERR'], yerr=qsor_mdot*(qsor['LOGLBOL_ERR']/qsor['LOGLBOL'])/(qsor_mdot*N.log(10)), marker='None', fmt='None', ecolor='k', alpha=0.1)
# ax.scatter(qsor['LOGBH'], N.log10(qsor_mdot.value), marker='^', c='None', edgecolor='g', s=20, label=r'$\rm{Shen}$ $\rm{et }$ $\rm{al. }$ $\rm{(2011)}$', alpha=0.8)
# ax.errorbar(int_mbh, N.log10(int_mdot.value), xerr=Err_int_mbh, yerr=int_mdot*(Err_int_lbol/int_lbol)/(int_mdot*N.log(10)), marker='None', fmt='None', ecolor='k', alpha=0.4)
# ax.scatter(int_mbh, N.log10(int_mdot.value), marker='s', c='b', edgecolor='b', s=30, label=r'$\rm{INT }$ $\rm{spectra}$')
# ax.errorbar(sdss_mbh, N.log10(sdss_mdot.value), xerr=Err_sdss_mbh, yerr=sdss_mdot*(Err_sdss_lbol/sdss_lbol)/(sdss_mdot*N.log(10)), marker='None', fmt='None', ecolor='k', alpha=0.4)
# ax.scatter(sdss_mbh, N.log10(sdss_mdot.value), marker='x', c='k', s=30, label=r'$\rm{SDSS}$ $\rm{spectra}$')
# ax.errorbar(qso_mbh, N.log10(qso_mdot.value), xerr=Err_qso_mbh, yerr=qso_mdot*(Err_qso_lbol/qso_lbol)/(qso_mdot*N.log(10)), marker='None', fmt='None', ecolor='k', alpha=0.4)
# ax.scatter(qso_mbh, N.log10(qso_mdot.value), marker='x', c='k', s=30)
# ax.scatter(brooke_mbh, N.log10(brooke_mdot.value), marker='o', c='None', edgecolor='k', s=30, label=r'$\rm{Simmons }$ $\rm{et }$ $\rm{al. }$ $\rm{(2013)}$')
# ax.errorbar(brooke_mbh, N.log10(brooke_mdot.value), xerr = err_brooke_bulgemass, yerr=brooke_mdot*(err_brooke_lbol/brooke_lbol)/(brooke_mdot*N.log(10)), ecolor='k', capthick=1, fmt='None', fill_style='None', alpha=0.5)
# line1 = ax.plot(mbh, N.log10((0.01*1.26E38*(10**mbh)*un.erg/un.s/(0.15*(c.to(un.cm/un.s))**2)).to(un.solMass/un.yr).value), linestyle=':', c='0.3')
# line2 = ax.plot(mbh, N.log10((0.1*1.26E38*(10**mbh)*un.erg/un.s/(0.15*(c.to(un.cm/un.s))**2)).to(un.solMass/un.yr).value), linestyle='-.', c='0.3')
# line3 = ax.plot(mbh, N.log10((1*1.26E38*(10**mbh)*un.erg/un.s/(0.15*(c.to(un.cm/un.s))**2)).to(un.solMass/un.yr).value), linestyle='dashed', c='0.3')
# ax.text(9.25, -0.22, r'$\lambda_{Edd}=0.01$', rotation=0, color='0.3')
# ax.text(8.5, 0.07, r'$\lambda_{Edd}=0.1$', rotation=0, color='0.3')
# ax.text(7.75, 0.32, r'$\lambda_{Edd}=1$', rotation=0, color='0.3')
# ax.set_xlabel(r'$\log_{10}(M_{BH}/M_{\odot})$')
# ax.set_ylabel(r'$\log_{10}(\dot{m}/\rm{M}_{\odot}~\rm{yr}^{-1})$')
# ax.minorticks_on()
# ax.set_xlim(5.5, 10.5)
# ax.set_ylim(-3, 0.5)
# ax.legend(frameon=False, loc=4, fontsize=12)
# #P.tight_layout()
# P.savefig('./mass_bh_mdot_with_all_errors_shen_11.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)


# # P.figure(figsize=(8,8))
# # P.scatter(ints[:,-8], target_fwhm, marker='s', c='k', s=20, label='INT observations')
# # #P.scatter(brooke_mtot, brooke_mbh, marker='x', c='k', s=16, label='Simmons et al. 2013')
# # #P.plot(N.log10(mtot), mbh, linestyle='dashed', c='k', label = 'Haring & Rix 2004')
# # P.xlabel(r'$u-r$ colour')
# # P.ylabel(r'FWHM $[\AA]$')
# # #P.xlim(9.5, 11.5)
# # #P.ylim(6.0, 9.0)
# # #P.legend(frameon=False, loc=4)
# # P.savefig('colour_fwhm.pdf')

# # P.figure(figsize=(8,8))
# # P.scatter(ints[:,-8], (ints[:,-6] - 5 * (N.log10(cosmo.luminosity_distance(ints[n,17]) * 1E6) - 1)), marker='s', c='k', s=20, label='INT observations')
# # #P.scatter(brooke_mtot, brooke_mbh, marker='x', c='k', s=16, label='Simmons et al. 2013')
# # #P.plot(N.log10(mtot), mbh, linestyle='dashed', c='k', label = 'Haring & Rix 2004')
# # P.xlabel(r'$u-r$ colour')
# # P.ylabel(r'$M_{r}$')
# # #P.xlim(9.5, 11.5)
# # #P.ylim(6.0, 9.0)
# # #P.legend(frameon=False, loc=4)
# # P.savefig('colour_r_magnitude.pdf')


# b = F.open('/Users/becky/Projects/int_reduc/bulgeless_Simmons13')
# bd = b[1].data
# bd_mbh_lim = bd["mbh_lim"]
# bd_mbh_lims = bd["mbh_lim"][N.where(bd_mbh_lim!=-1)]
# bd_mbh = bd['Mbh_BL'][N.where(bd_mbh_lim==-1)]
# bd_mtot = bd['LOGMSTAR_BALDRY06'][N.where(bd_mbh_lim==-1)]
# bd_mtot_lim = bd['LOGMSTAR_BALDRY06'][N.where(bd_mbh_lim!=-1)]
# bd_bollum = (bd['L_bol_use'])[N.where(bd_mbh_lim!=-1)]

# iedd = int_lbol/(10**int_mbh*1.26E38)
# Err_iedd = iedd * ( (Err_int_lbol/int_lbol)**2 + (N.log(10)*Err_int_mbh)**2)**0.5
# sedd = sdss_lbol/(10**sdss_mbh*1.26E38)
# Err_sedd = sedd * ( (Err_sdss_lbol/sdss_lbol)**2 + (N.log(10)*Err_sdss_mbh)**2)**0.5
# qedd = qso_lbol/(10**qso_mbh*1.26E38)
# Err_qedd = qedd * ( (Err_qso_lbol/qso_lbol)**2 + (N.log(10)*Err_qso_mbh)**2)**0.5


# fig = P.figure(figsize=(5,5))
# ax = fig.add_subplot(1,1,1)
# ax.errorbar(ints['col18'], iedd, yerr=Err_iedd, marker='None', fmt='None', ecolor='k', alpha=0.4)
# ax.scatter(ints['col18'], iedd, marker='s', c='k', s=30, label=r'$\rm{INT }$ $\rm{spectra}$')
# ax.errorbar(sdss['Redshift'], sedd, yerr=Err_sedd, marker='None', fmt='None', ecolor='k', alpha=0.4)
# ax.scatter(sdss['Redshift'], sedd, marker='x', c='k', s=30, label=r'$\rm{SDSS}$ $\rm{spectra}$')
# ax.errorbar(qso['z'], qedd, yerr=Err_qedd, marker='None', fmt='None', ecolor='k', alpha=0.4)
# ax.scatter(qso['z'], qedd, marker='x', c='k', s=30)
# ax.set_ylim(-0.25, 3)
# ax.set_xlabel(r'$z$')
# ax.set_ylabel(r'$\lambda_{Edd}$')
# ax.minorticks_on()
# ax.legend(frameon=False, loc=3  , fontsize=12)
# #P.tight_layout()
# P.savefig('./redshift_eddington_ratio_with_all_errors.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)



# isedd = N.append(iedd, sedd)
# avedd = N.mean(N.append(isedd, qedd))

# rs_bd_mbh_lim = N.log10(bd_bollum/(1.26E38*avedd))

# fig = P.figure(figsize=(6, 6))
# ax = fig.add_subplot(1,1,1)
# #P.figure(figsize=(8,8))
# ax.scatter(bd_mtot, bd_mbh, marker='o', c='None', edgecolor='k', s=50, label=r'$\rm{Simmons}$ $\rm{et}$ $\rm{al.}$ $\rm{(2013)}$')
# ax.scatter(bd_mtot_lim, rs_bd_mbh_lim, marker='^', c='None', edgecolor='k', s=50, label=r'$\rm{Simmons}$ $\rm{et}$ $\rm{al.}$ $\rm{(2013)}$ $\rm{limits}$')
# #P.scatter(bd_mtot_lim, bd_mbh_lims, marker='x', c='cyan', s=60, label=r'$\rm{Simmons}$ $\rm{et}$ $\rm{al.}$ $\rm{(2013)}$ $\rm{limits}$')
# ax.errorbar(int_mtot, int_mbh, xerr=Err_int_mtot, yerr=Err_int_mbh, marker='None', fmt='None', ecolor='k', alpha=0.4)
# ax.scatter(int_mtot, int_mbh, marker='s', c='k', s=50, label=r'$\rm{This}$ $\rm{work}$')
# ax.errorbar(sdss_mtot, sdss_mbh, xerr=Err_sdss_mtot, yerr=Err_sdss_mbh, marker='None', fmt='None', ecolor='k', alpha=0.4)
# ax.scatter(sdss_mtot, sdss_mbh, marker='x', c='k', s=30, label=r'$\rm{SDSS}$ $\rm{spectra}$')
# ax.errorbar(qso_mtot, qso_mbh, xerr=Err_qso_mtot, yerr=Err_qso_mbh, marker='None', fmt='None', ecolor='k', alpha=0.4)
# ax.scatter(qso_mtot, qso_mbh, marker='x', c='k', s=30)
# ax.plot(N.log10(mtot), mbh, linestyle='dashed', linewidth=2, c='k', label = r'$\rm{Haring}$ $\rm{\&}$ $\rm{Rix}$ $\rm{(2004)}$')
# ax.set_xlabel(r'$log_{10}$ $\rm{(M_{*}/M_{\odot})}$')
# ax.set_ylabel(r'$log_{10}$ $\rm{(M_{BH}/M_{\odot})}$')
# ax.minorticks_on()
# ax.tick_params('both', which='major', length=10, width=1)
# ax.tick_params('both', which='minor', length=5, width=0.5)
# ax.set_xlim(9.5, 11.5)
# ax.set_ylim(5.5, 10.5)
# P.legend(frameon=False, loc=2)
# #P.legend(frameon=False, loc=4)
# P.savefig('mass_bh_total_mass_sdss_qso_with_simmons_limits_all_errors.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)

# Mrs = N.linspace(-24.5, -17, 200)
# C_dash = 2.06 - 0.244*N.tanh((Mrs + 20.07)/1.09)
# upper = C_dash + 0.128
# lower = C_dash - 0.128

# Z = N.genfromtxt('/Users/becky/Projects/Green-Valley-Project/bayesian/find_t_tau/colour_mag_diag/baldry_hist_data.txt')
# X = N.genfromtxt('/Users/becky/Projects/Green-Valley-Project/bayesian/find_t_tau/colour_mag_diag/baldry_x_hist.txt')
# Y = N.genfromtxt('/Users/becky/Projects/Green-Valley-Project/bayesian/find_t_tau/colour_mag_diag/baldry_y_hist.txt')

# P.figure(figsize=(6,5))
# ax1 = P.subplot(111)
# plothist2d(Z.T, X, Y, ax=ax1, plot_contours=True, alpha=0.5, plot_datapoints=False, levels=[80, 160, 320, 640, 1280, 1920, 2560, 5120])
# ax1.plot(Mrs, C_dash, color='k', linewidth=2)
# ax1.plot(Mrs, upper, color='k', linestyle='dashed', linewidth=2)
# ax1.plot(Mrs, lower, color='k', linestyle='dashed', linewidth=2)
# im =ax1.scatter(gal_Mr, ur, c=int_mdot, marker='o', cmap=P.cm.Reds, vmin = 0.0107, vmax=0.59)
# im =ax1.scatter(sdss_gal_Mr, sdss_ur, c=sdss_mdot, marker='o', cmap=P.cm.Reds, vmin = 0.0107, vmax=0.59)
# im =ax1.scatter(qso_gal_Mr, qso_ur, c=qso_mdot, marker='o', cmap=P.cm.Reds, vmin = 0.0107, vmax=0.59)
# ax1.set_ylabel(r'$u-r$ $\rm{colour}$', fontsize=14)
# ax1.set_xlabel(r'$M_r$', fontsize=14)
# ax1.set_ylim(0, 3.5)
# ax1.set_xlim(-18.5, -23.5)
# cb = P.colorbar(im)
# cb.set_label(r'$\dot{m}$ $[\rm{M}_{\odot}\rm{yr}^{-1}]$', labelpad=10, fontsize=15)
# P.savefig('CMD_DISCDOM_coloured_accretion_rate.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
