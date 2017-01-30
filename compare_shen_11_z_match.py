""" File to plot all the comparison plots to the Shen + 2011 catalogue """

import numpy as N
import matplotlib.pyplot as P
import pyfits as F
from prefig import Prefig
from scipy.stats import ks_2samp
import os

from astropy.cosmology import FlatLambdaCDM

from astropy import units as un
from astropy.table import Table, vstack

import math

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

params =   {'font.size' : 20,
            'xtick.major.size': 8,
            'ytick.major.size': 8,
            'xtick.minor.size': 3,
            'ytick.minor.size': 3,
            }
P.rcParams.update(params) 


results =  Table.read('/Users/becky/Projects/int_reduc/desktop/all_brooke_101_galaxies_z_ur_mtot_mbh_lbol_lHa_some_zero_measurements.fits', format='fits')
qso = Table.read('/Users/becky/Projects/int_reduc/desktop/dr7_bh_May_2011.fits', format='fits')
qsor = Table.read('shen_2011_matched_redshift_Simard_Btot.fits', format='fits')
gal = Table.read('/Users/becky/Projects/followup_gv/bpt/MPA_JHU_MASS_SFR.fit', format='fits')

Lsol = 3.846E33*un.g*un.cm**2/un.s**3
Ledd = 3E4 * 10**results['MBH'] * Lsol
edd = results['L bol']/Ledd
edd = (edd/edd.unit)
# P.figure(figsize=(15,4))
# ax1 = P.subplot(131)
# ax1.hist(qso['LOGBH'], bins=15,  range=(5.5,10.5), histtype='step', color='k', alpha=0.3, linestyle='dashdot', normed=True)
# #ax1.hist(qsor['LOGBH'], bins=15, range=(5.5,10.5), histtype='step', color='k', linestyle='dashed', normed=True)
# ax1.hist(results['MBH'], bins=15, range=(5.5,10.5), histtype='step', color='k', normed=True)
# ax1.set_ylabel(r'$\rm{normalised}$ $\rm{density}$')
# ax1.set_xlabel(r'$log_{10}(M_{BH}/M_{\odot})$')
# ax1.set_xlim(5.5, 10.5)
# ax1.minorticks_on()
# ax2 = P.subplot(132)
# ax2.hist(qso['(B/T)r'], range=(5.5,10.5), histtype='step', color='k', alpha=0.3, linestyle='dashdot', normed=True)
# #ax2.hist(qsor['(B/T)r'], bins=15, range=(0,1), histtype='step', color='k', linestyle='dashed', normed=True)
# ax2.hist(results['(B/T)r'], bins=15, range=(0,1), histtype='step', color='k', normed=True)
# ax2.set_xlabel(r'$\rm{upper}$ $\rm{limit}$ $(B/T)_r$')
# #ax2.tick_params('y', labelleft='off')
# ax2.minorticks_on()
# ax3 = P.subplot(133)
# #ax3.hist(qso['LOGBH'], range=(5.5,10.5), histtype='step', color='k', alpha=0.3, linestyle='dashdot', normed=True)
# ax3.hist(gal['AVG_MASS'], bins=15, range=(7.5,12.5), histtype='step', color='k', linestyle='dashed', normed=True)
# ax3.hist(results['stellar mass'], bins=15, range=(7.5,12.5), histtype='step', color='k', normed=True)
# #ax3.tick_params('y', labelleft='off')
# ax3.set_xlabel(r'$\log_{10}[M_{*}/M_{\odot}]$')
# ax3.minorticks_on()
# ax3.set_xlim(7.5,12.5)
# P.tight_layout()
# #P.subplots_adjust(wspace=0.05)
# P.savefig('diskdom_mbh_btot_stellar_mass_distributions_compare_all_shen.pdf')


P.figure(figsize=(15,4))
ax1 = P.subplot(131)
ax1.hist(qso['LOGBH'], bins=15,  range=(5.5,10.5), histtype='step', color='k', linestyle='dashed', normed=True)
#ax1.hist(qsor['LOGBH'], bins=15, range=(5.5,10.5), histtype='step', color='k', linestyle='dashed', normed=True)
ax1.hist(results['MBH'], bins=15, range=(5.5,10.5), histtype='step', color='k', normed=True)
ax1.set_ylabel(r'$\rm{normalised}$ $\rm{density}$')
ax1.set_xlabel(r'$log_{10}(M_{BH}/M_{\odot})$')
ax1.set_xlim(5.5, 10.5)
ax1.minorticks_on()
ax2 = P.subplot(132)
ax2.hist(qso['LOGLBOL'], range=(42,48), histtype='step', color='k', linestyle='dashed', normed=True)
#ax2.hist(qsor['(B/T)r'], bins=15, range=(0,1), histtype='step', color='k', linestyle='dashed', normed=True)
ax2.hist(N.log10(results['L bol']), bins=15, range=(42,48), histtype='step', color='k', normed=True)
ax2.set_xlabel(r'$log_{10}(L_{bol} [\rm{erg}$ $\rm{s}^{-1}])$')
#ax2.tick_params('y', labelleft='off')
ax2.minorticks_on()
ax3 = P.subplot(133)
#ax3.hist(gal['LOGBH'], range=(5.5,10.5), histtype='step', color='k', alpha=0.3, linestyle='dashdot', normed=True)
ax3.hist(qso['LOGEDD_RATIO'], bins=15, range=(-3, 2), histtype='step', color='k', linestyle='dashed', normed=True)
ax3.hist(N.log10(edd), bins=15, range=(-3, 2), histtype='step', color='k', normed=True)
#ax3.tick_params('y', labelleft='off')
ax3.set_xlabel(r'$\log_{10}(\lambda_{Edd})$')
ax3.minorticks_on()
ax3.set_xlim(-3, 2)
P.tight_layout()
#P.subplots_adjust(wspace=0.05)
P.savefig('diskdom_mbh_lbol_edd_ratio_distributions_compare_all_117_extra_qso_shen.pdf')

DBH, pBH = ks_2samp(results['MBH'], qsor['LOGBH'])
Dbol, pbol = ks_2samp(N.log10(results['L bol']), qsor['LOGLBOL'])
Dedd, pedd = ks_2samp(qsor['LOGEDD_RATIO'], N.log10(edd.value))

DBH, pBHr = ks_2samp(results['MBH'], qso['LOGBH'])
Dbol, pbolr = ks_2samp(N.log10(results['L bol']), qso['LOGLBOL'])
Dedd, peddr = ks_2samp(qso['LOGEDD_RATIO'], N.log10(edd.value))



P.figure(figsize=(15,4))
ax1 = P.subplot(132)
ax1.hist(results['MBH'], bins=15, range=(5.5,10.5), histtype='step', color='k', normed=True)
ax1.hist(qsor['LOGBH'], bins=15,  range=(5.5,10.5), histtype='step', color='k', linestyle='dashed', normed=True)
#ax1.hist(qso['LOGBH'], bins=15, range=(5.5,10.5), histtype='step', color='k', linestyle='dashdot', normed=True)
ax1.set_xlabel(r'$\rm{log}($$M_{\rm{BH}}/M_{\odot})$')
ax1.set_xlim(5.5, 10.5)
ax1.set_ylim(0, 0.95)
#ax1.plot([0.05, 0.075], [0.85, 0.85], color='k', linestyle='dashed', transform=ax1.transAxes)
#ax1.plot([0.05, 0.075], [0.765, 0.765], color='k', alpha=0.6, linestyle='dashdot', transform=ax1.transAxes)
#ax1.text(0.05, 0.85, r'$p = $'+"%.2g" % pBH, color='k', transform=ax1.transAxes)
#ax1.text(0.1, 0.765, r'$p = $'+"%.2g" % pBHr, color='k', alpha=0.6, transform=ax1.transAxes)
ax1.minorticks_on()
ax1.tick_params('y', labelleft='off')
ax2 = P.subplot(131)
ax2.hist(N.log10(results['L bol']), bins=15, range=(43.5,46), histtype='step', color='k', normed=True)
ax2.hist(qsor['LOGLBOL'], bins=15, range=(43.5,46), histtype='step', color='k', linestyle='dashed', normed=True)
#ax2.hist(qso['LOGLBOL'], bins=15, range=(42,48), histtype='step', color='k', linestyle='dashdot', normed=True)
ax2.set_xlabel(r'$\rm{log}($$L_{\rm{bol}} [\rm{erg}$ $\rm{s}^{-1}])$')
#ax2.plot([0.05, 0.075], [0.85, 0.85], color='k', linestyle='dashed', transform=ax2.transAxes)
#ax2.plot([0.05, 0.075], [0.765, 0.765], color='k', alpha=0.6, linestyle='dashdot', transform=ax2.transAxes)
#ax2.text(0.05, 0.85, r'$p = $'+"%.2g" % pbol, color='k', transform=ax2.transAxes)
#ax2.text(0.1, 0.765, r'$p = $'+"%.2g" % pbolr, color='k', alpha=0.6, transform=ax2.transAxes)
ax2.minorticks_on()
ax2.tick_params('y', labelleft='off')
ax2.set_ylabel(r'$\rm{normalised}$ $\rm{density}$')
ax3 = P.subplot(133)
#ax3.hist(qso['LOGEDD_RATIO'], bins=15, range=(-3, 2), histtype='step', color='k', alpha=0.3, linestyle='dashdot', normed=True)
ax3.hist(qsor['LOGEDD_RATIO'], bins=15, range=(-3, 2), histtype='step', color='k', linestyle='dashed', normed=True)
ax3.hist(N.log10(edd), bins=15, range=(-3, 2), histtype='step', color='k', normed=True)
ax3.tick_params('y', labelleft='off')
ax3.set_xlabel(r'$\rm{log}(\lambda_{\rm{Edd}})$')
#ax3.plot([0.05, 0.075], [0.85, 0.85], color='k', linestyle='dashed', transform=ax3.transAxes)
#ax3.plot([0.05, 0.075], [0.765, 0.765], color='k', alpha=0.6, linestyle='dashdot', transform=ax3.transAxes)
#ax3.text(0.05, 0.85, r'$p = $'+"%.2g" % pedd, color='k', transform=ax3.transAxes)
#ax3.text(0.1, 0.765, r'$p = $'+"%.2g" % peddr, color='k', alpha=0.6, transform=ax3.transAxes)
ax3.minorticks_on()
ax3.set_xlim(-3, 2)
P.tight_layout()
#P.subplots_adjust(wspace=0.05)
P.savefig('diskdom_117_mbh_lbol_edd_ratio_distributions_compare_redshift_matched_shen.pdf')




P.figure(figsize=(6, 3))
P.hist(gal['Z'], range=(0,0.3), histtype='step', color='k', alpha=0.3, linestyle='dashdot', normed=True)
P.hist(qso['REDSHIFT'], range=(0,0.3), histtype='step', color='k', linestyle='dashed', normed=True)
P.hist(results['z'],range=(0,0.3), histtype='step', color='k', normed=True)
P.xlabel(r'$z$')
P.ylabel(r'$\rm{normalised}$ $\rm{density}$')
P.tick_params('y', labelleft='off')
P.tight_layout()
P.savefig('z_all_shen_2011_compare.pdf')

# P.figure(figsize=(6, 3))
# P.hist(qso['LOGBH'], range=(5.5,10.5), histtype='step', color='k', alpha=0.3, linestyle='dashdot', normed=True)
# P.hist(qsor['LOGBH'], range=(5.5,10.5), histtype='step', color='k', linestyle='dashed', normed=True)
# P.hist(results['MBH'], range=(5.5,10.5), histtype='step', color='k', normed=True)
# P.xlabel(r'$log_{10}(M_{BH}/M_{\odot})$')
# P.tight_layout()
# P.savefig('mbh_z_matched_shen_2011_compare.pdf')

# P.figure(figsize=(6, 3))
# P.hist(qso['LOGLBOL'], range=(43.5, 48.5), histtype='step', color='k', alpha=0.3, linestyle='dashdot', normed=True)
# P.hist(qsor['LOGLBOL'],range=(43.5, 48.5), histtype='step', color='k', linestyle='dashed', normed=True)
# P.hist(N.log10(results['L bol']),range=(43.5, 48.5), histtype='step', color='k', normed=True)
# P.xlabel(r'$log_{10}(L_{bol}$ $[\rm{erg}$ $\rm{s}^{-1}])$')
# P.xlim(43, 46.5)
# P.tight_layout()
# P.savefig('lbol_z_matched_shen_2011_compare.pdf')

# Dr, pr = ks_2samp(N.log10(results['L bol']/(10**results['MBH']*1.26E38)), qsor['LOGEDD_RATIO'])
# D, p = ks_2samp(N.log10(results['L bol']/(10**results['MBH']*1.26E38)), qso['LOGEDD_RATIO'])

# P.figure(figsize=(5, 2.5))
# P.hist(qso['LOGEDD_RATIO'], range=(-3, 1), histtype='step', color='k', alpha=0.6, linestyle='dashdot', normed=True)
# P.hist(qsor['LOGEDD_RATIO'], range=(-3, 1), histtype='step', color='k', linestyle='dashed', normed=True)
# P.hist(N.log10(results['L bol']/(10**results['MBH']*1.26E38)), range=(-3, 1), histtype='step', color='k', normed=True)
# P.plot([-2.8, -2.55], [0.765, 0.765], color='k', alpha=0.6, linestyle='dashdot')
# P.plot([-2.8, -2.55], [0.85, 0.85], color='k', linestyle='dashed')
# P.text(-2.5, 0.85, r'$p = $'+"%.2g" % pr, color='k')
# P.text(-2.5, 0.765, r'$p = $'+"%.2g" % p, color='k', alpha=0.6)
# P.xlabel(r'$log_{10} \lambda_{Edd}$')
# P.ylabel(r'$\rm{normalised}$ $\rm{density}$')
# P.tick_params('y', labelleft='off')
# P.tight_layout()
# P.savefig('edd_ratio_z_matched_shen_2011_compare.pdf')

# H, X, Y = N.histogram2d(qso['LOGBH'], qso['LOGLBOL'], bins=25, range=((5.5, 10.5),(43.5, 48.5)))

# P.figure(figsize=(6,6))
# # P.scatter(qso['LOGBH'], qso['LOGLBOL'], color='0.1', marker='x', alpha=0.1)
# P.pcolor(X, Y, H.T, cmap=P.cm.binary, alpha=0.2)
# P.contour(X[:-1]+N.diff(X)[-1], Y[:-1]+N.diff(Y)[-1], H.T, colors='k', alpha=0.2)
# P.scatter(results['MBH'], N.log10(results['L bol']), color='r', marker='x')
# P.errorbar(results['MBH'], N.log10(results['L bol']), xerr=results['Err MBH'], yerr=results['Err L bol']/(results['L bol']*N.log(10)), marker='None', fmt='None', ecolor='k', alpha=0.1)
# P.scatter(qsor['LOGBH'], qsor['LOGLBOL'], color='k', marker='x')
# P.xlabel(r'$log_{10}(M_{BH}/M_{\odot})$')
# P.ylabel(r'$log_{10}(L_{bol}$ $[\rm{erg}$ $\rm{s}^{-1}])$')
# P.xlim(5.5, 10.5)
# P.ylim(43.5, 48.5)
# P.minorticks_on()
# P.tight_layout()
# P.savefig('mbh_lbol_z_matched_shen_2011_compare.pdf')




