import numpy as N
import matplotlib.pyplot as P
#import pyfits as F
from astropy.table import Table
#from prefig import Prefig

import sys, os

from astropy.cosmology import FlatLambdaCDM

from astropy import units as un
from astropy.table import Table, vstack, Column

import math
import linmix

try:
    linmixfilebase = sys.argv[1]
    read_from_file = True
    print("Reading linmix parameters from a previous fit, specified as being stored in in %s_*.npy" % linmixfilebase)
except:
    linmixfilebase = ''
    read_from_file = False
    print("New linmix parameters to be computed, will be stored in linmix_results_*.npy")



datatable = '/Users/vrooje/Documents/Astro/bbcint_paper/data/spectra/sdss_int_results_table_101_sources_new_bhmasses.fits'
# this is quite an all-inclusive table and it includes results from stuff we've done in other scripts,
# like computing BH masses, galaxy masses, and also external stuff like Simard et al.
# bulge-to-total ratios.
# The relevant columns here are:
# name, ID, RA, Dec, spectrum_source, z, Err z, stellar mass, Err stellar mass, L bol, Err L bol, (B/T)r, e_(B/T)r, MBH_best, dMBH_hi_best, dMBH_lo_best
cols_keep = ['name', 'ID', 'RA', 'Dec', 'spectrum_source', 'z', 'Err z', 'stellar mass', 'Err stellar mass', 'L bol', 'Err L bol', '(B/T)r', 'e_(B/T)r', 'MBH_best', 'dMBH_hi_best', 'dMBH_lo_best']

mbhcol  = 'MBH_best'
dmbhcol = 'dMBH_best'

intcolor = '#228dcc'




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
##################################          ####################################
###############################    PLOTTING    #################################
##################################          ####################################
################################################################################




####################################################
# Just the bulge mass plot
####################################################

def plot_mbh_mbulge_only(xs, uplysb, nuplysb, uplysp, uplysm, xhr, yhr, xhrerr, yhrerr, brooke_bulgemass, brooke_mbh, err_brooke_mtot, err_brooke_mbh, int_bulgemass, int_mbh, err_int_bulgemass, Err_int_mbh, bulgemass, sdss_mbh, hrysb, hrysp, hrysm, int_is_meas):
    fig = P.figure(figsize=(5,5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(uplysb, xs, color='k', label='Best Fit')
    ax.fill_betweenx(xs, x1=uplysp, x2=uplysm, color='k', alpha=0.1)
    ax.plot(nuplysb, xs, color='k', linestyle='dashed')
    #ax.fill_betweenx(xs, x1=nuplysp, x2=nuplysm, color='k', alpha=0.1, hatch='-')
    #ax.plot(juplysb, xs, color='k', linestyle='dotted', label='Fit to only upper limits')
    #ax.fill_betweenx(xs, x1=juplysp, x2=juplysm, color='k', alpha=0.1, hatch='.')
    #ax.plot(uplhrysb, xs, color='k', linestyle='dashdot', label='Fit to all our data, no limits + HR04 data')
    #ax.fill_betweenx(xs, x1=uplhrysp, x2=uplhrysm, color='k', alpha=0.1, hatch='/')

    ax.scatter( xhr, yhr, marker='o', c='None', edgecolor='r', label = r'$\rm{Haring }$ $\rm{\& }$  $\rm{Rix }$ $\rm{(2004)}$')
    ax.errorbar(xhr, yhr, xerr=xhrerr, yerr=yhrerr, marker='o', color='None', ecolor='k', alpha=0.3, label='_nolegend_')

    ax.errorbar(brooke_bulgemass, brooke_mbh, xerr = err_brooke_mtot, yerr=err_brooke_mbh, ecolor='k', capthick=1, fmt='None', fill_style='None', alpha=0.5, label='_nolegend_')
    ax.scatter( brooke_bulgemass, brooke_mbh, marker='o', c='None', edgecolor='k', s=30, label=r'$\rm{Simmons }$ $\rm{et }$ $\rm{al. }$ $\rm{(2013)}$')

    ax.errorbar(int_bulgemass,  int_mbh, xerr = err_int_bulgemass, yerr=Err_int_mbh, ecolor='k', alpha=0.5, capthick=1, fmt='None', fill_style='None', label='_nolegend_')
    ax.scatter( int_bulgemass,  int_mbh, marker='s', c=intcolor, edgecolor=intcolor, s=30, label=r'$\rm{INT }$ $\rm{spectra}$')
    ax.errorbar(    bulgemass, sdss_mbh, xerr = 0.2, xuplims=True, ecolor='k', alpha=0.5, capthick=1, fmt='None', fill_style='None', label=r'$\rm{SDSS}$ $\rm{limits}$')
    #ax.errorbar(11.9, 10, xerr = N.mean(err_bt_bulgemass), yerr=N.mean(err_bt_mbhs), ecolor='k', fmt='None', alpha=0.5, label='_nolegend_')
    ax.plot(xs, hrysb, color='r', linestyle='dashed', label = r'$\rm{Haring }$ $\rm{\& }$  $\rm{Rix }$ $\rm{(2004)}$ $\rm{fit}$')
    ax.fill_between(xs, y1=hrysp, y2=hrysm, color='r', alpha=0.1)
    ax.set_xlabel(r'$\log (M_{\rm{bulge}}/\rm{M}_{\odot})$')
    ax.minorticks_on()
    ax.set_xlim(7.9, 12.1)
    ax.set_ylim(5.5, 10.5)
    ax.set_ylabel(r'$\log (M_{\rm{BH}}/\rm{M}_{\odot})$')
    ax.legend(frameon=False, loc=2, fontsize=11)
    P.tight_layout()
    P.subplots_adjust(wspace=0.0)
    P.savefig('mass_bh_bulge_limits_INT_simmons13_measurements_linmix_fit_no_simard_bt_set_to_1.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
    P.close()



####################################################
# Just the total mass plot
####################################################

def plot_mbh_mgal_only(xs, rysb, rysp, rysm, brooke_mtot, brooke_mbh, err_brooke_mtot, err_brooke_mbh, int_mtot, int_mbh, Err_int_mtot, Err_int_mbh, sdss_mtot, sdss_mbh, Err_sdss_mtot, Err_sdss_mbh, hrysb, hrysp, hrysm):
    fig = P.figure(figsize=(5,5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(xs, rysb, color='k', linestyle='solid', label='Best Fit')
    ax.fill_between(xs, y1=rysp, y2=rysm, color='k', alpha=0.1)

    ax.scatter( brooke_mtot, brooke_mbh, marker='o', c='None', edgecolor='k', s=30, label=r'$\rm{Simmons }$ $\rm{et }$ $\rm{al. }$ $\rm{(2013)}$')
    ax.errorbar(brooke_mtot, brooke_mbh, xerr = err_brooke_mtot, yerr=err_brooke_mbh, ecolor='k', capthick=1, fmt='None', fill_style='None', alpha=0.5, label='_nolegend_')

    ax.errorbar(   int_mtot,    int_mbh, xerr=Err_int_mtot, yerr=Err_int_mbh, marker='None', fmt='None', ecolor='blue', alpha=0.4, label='_nolegend_')
    ax.scatter(    int_mtot,    int_mbh, marker='s', c=intcolor, edgecolor =intcolor, s=30, label=r'$\rm{INT }$ $\rm{spectra}$')

    ax.errorbar(  sdss_mtot, sdss_mbh, xerr=Err_sdss_mtot, yerr=Err_sdss_mbh, marker='None', fmt='None', ecolor='k', alpha=0.4, label='_nolegend_')
    ax.scatter(   sdss_mtot, sdss_mbh, marker='x', c='k', s=30, label=r'$\rm{SDSS}$ $\rm{spectra}$')
    #ax.errorbar(qso_mtot, qso_mbh, xerr=Err_qso_mtot, yerr=Err_qso_mbh, marker='None', fmt='None', ecolor='k', alpha=0.4)
    #ax.scatter(qso_mtot, qso_mbh, marker='x', c='k', s=30)
    ax.plot(xs, hrysb, color='r', linestyle='dashed', label = r'$\rm{Haring }$ $\rm{\& }$  $\rm{Rix }$ $\rm{(2004)}$ $\rm{fit}$')
    ax.fill_between(xs, y1=hrysp, y2=hrysm, color='r', alpha=0.1)
    # ax.plot(N.log10(mtot), bestreshr, linestyle='dashed', c='k', label = r'$\rm{Haring }$ $\rm{\& }$  $\rm{Rix }$ $\rm{2004}$')
    # ax.plot(N.log10(mtot), plusreshr, linestyle='-.', c='k')
    # ax.plot(N.log10(mtot), minusreshr, linestyle='-.', c='k')
    ax.set_xlabel(r'$\log (M_{*}/\rm{M}_{\odot})$')
    ax.set_ylabel(r'$\log (M_{\rm{BH}}/\rm{M}_{\odot})$')
    ax.minorticks_on()
    ax.set_xlim(9.4, 11.6)
    ax.set_ylim(5.5, 10.5)
    ax.legend(frameon=False, loc=2, fontsize=12)
    P.tight_layout()
    P.subplots_adjust(wspace=0.0)
    P.savefig('mass_bh_total_mass_fit_linmix_fit_101_brooke.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
    P.close()




####################################################
# Both plots together
####################################################
def plot_mbh_mbulge_mgal(xs, uplysb, uplysp, uplysm, nuplysb, nuplysp, nuplysm, xhr, yhr, xhrerr, yhrerr, hrysb, hrysp, hrysm, rysb, rysp, rysm, brooke_bulgemass, brooke_mtot, brooke_mbh, err_brooke_mtot, err_brooke_mbh, int_bulgemass, int_mtot, int_mbh, err_int_bulgemass, Err_int_mbh, bulgemass, sdss_mtot, sdss_mbh, Err_sdss_mtot, Err_sdss_mbh, int_is_meas):
    fig = P.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)



    # BULGE MASS
    # the plot if we assumed none of the bulge masses were limits
    # (which they are, but this is just to check)
    #ax1.plot(nuplysb, xs, color='green', linestyle='dotted', alpha=0.8)
    #ax1.fill_betweenx(xs, x1=nuplysp, x2=nuplysm, color='green', alpha=0.06)

    # fit performed with upper limits
    ax1.plot(uplysb, xs, color='k', label='Best Fit')
    ax1.fill_betweenx(xs, x1=uplysp, x2=uplysm, color='k', alpha=0.1)

    # Haring & Rix points
    ax1.scatter(xhr, yhr, marker='o', c='None', edgecolor='r', label = r'$\rm{Haring }$ $\rm{\& }$  $\rm{Rix }$ $\rm{(2004)}$')
    ax1.errorbar(xhr, yhr, xerr=xhrerr, yerr=yhrerr, marker='o', color='None', ecolor='k', alpha=0.3, label='_nolegend_')

    # Simmons et al. 2013 points
    ax1.errorbar(brooke_bulgemass, brooke_mbh, xerr = err_brooke_mtot, yerr=err_brooke_mbh, ecolor='k', capthick=1, fmt='None', fill_style='None', alpha=0.5, label='_nolegend_')
    ax1.scatter( brooke_bulgemass, brooke_mbh, marker='o', c='None', edgecolor='k', s=30, label=r'$\rm{Simmons }$ $\rm{et }$ $\rm{al. }$ $\rm{(2013)}$')


    # INT points
    ax1.errorbar(int_bulgemass[int_is_meas],  int_mbh[int_is_meas], xerr = err_int_bulgemass[int_is_meas], yerr=Err_int_mbh[int_is_meas], ecolor=intcolor, alpha=0.5, capthick=1, fmt='None', fill_style='None', label='_nolegend_')
    ax1.scatter( int_bulgemass[int_is_meas],  int_mbh[int_is_meas], marker='s', c=intcolor, edgecolor=intcolor, s=30, label=r'$\rm{INT }$ $\rm{spectra}$')
    ax1.errorbar(int_bulgemass[N.invert(int_is_meas)],  int_mbh[N.invert(int_is_meas)], xerr = 0.2, xuplims=True, ecolor=intcolor, alpha=0.5, capthick=1, fmt='None', fill_style='None', label='_nolegend_')

    # SDSS points (actually limits)
    # would love to know how to get this to plot correctly in the legend
    ax1.errorbar(    bulgemass, sdss_mbh, xerr = 0.2, xuplims=True, ecolor='k', alpha=0.5, capthick=1, fmt='None', fill_style='None', label=r'$\rm{SDSS}$ $\rm{spectra}$')
    #ax1.errorbar(11.9, 10, xerr = N.mean(err_bt_bulgemass), yerr=N.mean(err_bt_mbhs), ecolor='k', fmt='None', alpha=0.5, label='_nolegend_')

    # Haring & Rix fit
    ax1.plot(xs, hrysb, color='r', linestyle='dashed', label = r'$\rm{Haring }$ $\rm{\& }$  $\rm{Rix }$ $\rm{(2004)}$ $\rm{fit}$')
    ax1.fill_between(xs, y1=hrysp, y2=hrysm, color='r', alpha=0.1)

    ax1.set_xlabel(r'$\log (M_{\rm{bulge}}/\rm{M}_{\odot})$')
    ax1.minorticks_on()
    ax1.set_xlim(7.9, 12.1)
    ax1.set_ylim(5.5, 10.5)
    ax1.set_ylabel(r'$\log (M_{\rm{BH}}/\rm{M}_{\odot})$')
    ax1.legend(frameon=False, loc=2, fontsize=11)


    ########################
    # TOTAL MASS

    # linmix best fit
    ax2.plot(xs, rysb, color='k', linestyle='solid', label='Best Fit')
    ax2.fill_between(xs, y1=rysp, y2=rysm, color='k', alpha=0.1)

    # Simmons et al.
    ax2.scatter( brooke_mtot, brooke_mbh, marker='o', c='None', edgecolor='k', s=30, label=r'$\rm{Simmons }$ $\rm{et }$ $\rm{al. }$ $\rm{(2013)}$')
    ax2.errorbar(brooke_mtot, brooke_mbh, xerr = err_brooke_mtot, yerr=err_brooke_mbh, ecolor='k', capthick=1, fmt='None', fill_style='None', alpha=0.5, label='_nolegend_')

    # INT
    ax2.errorbar(   int_mtot,    int_mbh, xerr=Err_int_mtot, yerr=Err_int_mbh, marker='None', fmt='None', ecolor=intcolor, alpha=0.4, label='_nolegend_')
    ax2.scatter(    int_mtot, int_mbh, marker='s', c=intcolor, edgecolor =intcolor, s=30, label=r'$\rm{INT }$ $\rm{spectra}$')

    # SDSS
    ax2.errorbar(  sdss_mtot, sdss_mbh, xerr=Err_sdss_mtot, yerr=Err_sdss_mbh, marker='None', fmt='None', ecolor='k', alpha=0.4, label='_nolegend_')
    ax2.scatter(   sdss_mtot, sdss_mbh, marker='x', c='k', s=30, label=r'$\rm{SDSS}$ $\rm{spectra}$')

    # Haring & Rix
    ax2.plot(xs, hrysb, color='r', linestyle='dashed', label = r'$\rm{Haring }$ $\rm{\& }$  $\rm{Rix }$ $\rm{(2004)}$ $\rm{fit}$')
    ax2.fill_between(xs, y1=hrysp, y2=hrysm, color='r', alpha=0.1)
    #ax2.fill_between(xs, y1=qhrysp, y2=qhrysm, color='g', alpha=0.1)

    ax2.set_xlabel(r'$\log (M_{*}/\rm{M}_{\odot})$')

    ax2.minorticks_on()
    ax2.set_xlim(9.4, 11.6)
    ax2.set_ylim(5.5, 10.5)
    ax2.legend(frameon=False, loc=2, fontsize=12)

    ax1.text(0.95, 0.05, '(a)', verticalalignment='center', horizontalalignment='center', transform=ax1.transAxes)
    ax2.text(0.95, 0.05, '(b)', verticalalignment='center', horizontalalignment='center', transform=ax2.transAxes)


    P.tight_layout()
    P.subplots_adjust(wspace=0.0)
    P.savefig('mass_bh_bulge_limits_total_mass_INT_simmons13_measurements_linmix_fit_no_simard_bt_set_to_1.pdf', frameon=False, bbox_inches='tight', pad_inches=0.1, transparent=True)
    P.close()






################################################################################
##################################          ####################################
##############################   END PLOTTING    ###############################
##################################          ####################################
################################################################################







thedata = Table.read(datatable)
results = thedata[cols_keep]

results[dmbhcol] = 0.5*(results['dMBH_hi_best'] + results['dMBH_lo_best'])


mtot = N.linspace(10**8, 10**13, 20)
mbh = 8.2 + 1.12 * N.log10(mtot/1E11)
mbhplus = 8.3 + 1.18 * N.log10(mtot/1E11)
mbhminus = 8.1 + 1.06 * N.log10(mtot/1E11)

yhr = Column(data=N.log10([3E9, 1.4E7, 1E8, 4.3E8, 5.2E8, 5.3E8, 3.3E8, 1.4E7, 3.7E7, 2.5E9, 4.5E7, 2.5E6, 4.4E7, 1.4E7, 1.0E9, 2.1E8, 1.0E8, 1.6E7, 1.9E8, 3.1E8, 3.0E8, 1.1E8, 5.6E7, 1.0E9, 2.0E9, 1.7E8, 2.4E8, 1.3E7, 3.5E6, 3.7E6]), name='Haring and Rix MBH', unit=un.MsolMass)
yhrerr = Column(data=([1E9, 1E7, 0.55E8, 2.45E8, 1.05E8, 3E8, 2.8E8, 0.45E7, 1.6E7, 0.45E9, 3.25E7, 0.5E6, 0.5E7, 1.2E7, 0.8E9, 0.8E8, 0.5E8, 0.11E7, 0.8E8, 1.2E8, 1.35E8, 0.6E8, 0.5E7, 0.85E9, 0.75E9, 0.11E8, 0.9E8, 0.55E7, 1.25E6, 1.5E6]/(10**yhr.data * N.log(10))), name='Haring and Rix Err MBH', unit =un.MsolMass)
xhr = Column(data=N.log10([6E11, 2.3E10, 6.8E10, 3.6E11, 3.6E11, 5.6E11, 2.9E11, 6.2E9, 1.3E11, 2.9E11, 3.7E10, 8E8, 6.9E10, 7.6E10, 1.2E11, 6.8E10, 1.6E10, 2E10, 9.7E10, 1.3E11, 1.2E10, 9.2E10, 4.4E10, 2.7E11, 4.9E11, 1.1E11, 3.7E10, 1.5E10, 7E9, 1.1E10]), name='Haring and Rix bulge mass', unit=un.MsolMass)
xhrerr = Column(data=0.18*N.log10(xhr.data), name='Haring and Rix Err bulge mass', unit =un.MsolMass)

y = results[mbhcol]
yerr = results[dmbhcol]
x = results['stellar mass']
xerr = results['Err stellar mass']

# I think this is a legacy step that makes copy-pasting easier
bt = results

is_sdss = bt['spectrum_source'] == 'SDSS'
# it seems to be reading these in as "INT ", so remove trailing spaces
is_int  = N.array([q.rstrip() == 'INT' for q in bt['spectrum_source']])

# Set those which don't have Simard B/T to have a B/T=1 as an upper limit
bt['(B/T)r'][N.isnan(bt['(B/T)r'])] = 1
bt['e_(B/T)r'][N.isnan(bt['e_(B/T)r'])] = N.mean(bt['e_(B/T)r'][N.isfinite(bt['e_(B/T)r'])])


#Calculate bulgemass upper limits of SDSS galaxies
bulgemass = N.log10(bt['(B/T)r'][is_sdss]) + bt['stellar mass'][is_sdss]
# in the second term of this I don't think you need the #/bt['stellar mass'][is_sdss]
# because the error in log space already is a ratio
err_bulgemass = N.sqrt((bt['e_(B/T)r'][is_sdss]/bt['(B/T)r'][is_sdss])**2 + (bt['Err stellar mass'][is_sdss])**2)



int_bt = bt['(B/T)r'][is_int]
int_bulgemass = N.log10(bt['(B/T)r'][is_int]) + bt['stellar mass'][is_int]
# in the second term of this I don't think you need the #/bt['stellar mass'][is_int]
# because the error in log space already is a ratio
err_int_bulgemass = N.sqrt( (0.2)**2 + (bt['Err stellar mass'][is_int])**2 )
#B/T == 1 means it's an upper limit, not actually measured
int_is_meas = int_bt < 1.
int_delta = N.zeros(len(int_bt))
int_delta[int_is_meas] += 1

# going to define some other things for plotting purposes
int_mtot      = bt['stellar mass'][is_int]
Err_int_mtot  = bt['Err stellar mass'][is_int]
int_mbh       = bt[mbhcol][is_int]
Err_int_mbh   = bt[dmbhcol][is_int]

sdss_mtot     = bt['stellar mass'][is_sdss]
Err_sdss_mtot = bt['Err stellar mass'][is_sdss]
sdss_mbh      = bt[mbhcol][is_sdss]
Err_sdss_mbh  = bt[dmbhcol][is_sdss]


# from Simmons et al. (2013)
brooke_mtot = N.array([10.05, 10.03])
err_brooke_mtot = N.array([0.1, 0.1])
brooke_mbh = N.array([7.1, 6.6])
err_brooke_mbh = N.array([0.13, 0.14])
brooke_lbol = N.array([44.1, 43.4])
err_brooke_lbol=N.array([0.3, 0.3])
brooke_BT = N.array([0.011, 0.022])
brooke_bulgemass = N.log10(brooke_BT*10**brooke_mtot)
err_brooke_bulgemass = N.sqrt( (N.array([0.15, 0.2]))**2 + (N.array([0.15,0.12]))**2 )



bt_mbhs     = N.append(    brooke_mbh, N.append(N.array(bt[mbhcol][is_sdss]),  N.array(bt[mbhcol][is_int])))
err_bt_mbhs = N.append(err_brooke_mbh, N.append(N.array(bt[dmbhcol][is_sdss]), N.array(bt[dmbhcol][is_int])))
bt_bulgemass     = N.append(    brooke_bulgemass, N.append(N.array(bulgemass),     N.array(int_bulgemass)))
err_bt_bulgemass = N.append(err_brooke_bulgemass, N.append(N.array(err_bulgemass), N.array(err_int_bulgemass)))

#Set delta for linmix array which says what are upper limits. 1 is a measured value. 0 is an upper limit.
#Brooke's galaxies have measured B/T, SDSS are all upper limits and INT values have 2 measured, 3 upper limits (set to a B/T =1)
delta = N.append(N.ones(len(brooke_mtot)), N.append(N.zeros(len(bt[is_sdss])), int_delta))

# plot a histogram of B/T ratios, just a check (as almost all, but not all, are upper limits)
P.figure(figsize=(6,3))
P.hist(N.append(brooke_BT, N.array(bt['(B/T)r'])), range=(-0.05,1.05), bins=15, histtype='step', color='k')
P.xlabel(r'$[\rm{B}/\rm{T}]_r$')
P.ylabel(r'$\rm{number}$')
P.ylim(0,20)
P.minorticks_on()
P.tight_layout()
P.savefig('bulge_to_total_r_ratio_hist_with_INT_simmons13.pdf', frameon=False, transparent=True)





xs = N.linspace(0, 15, 100)


# Now either load the linmix parameters from a file or re-fit the data
# depending on what was specified when the program started
if read_from_file:
    lmhr_chain    = N.load('%s_haringrixfit.npy' % linmixfilebase)
    lmhr_alpha    = lmhr_chain['alpha']
    lmhr_beta     = lmhr_chain['beta']

    lmr_chain     = N.load('%s_MBH_stellarmass.npy' % linmixfilebase)
    lmr_alpha     = lmr_chain['alpha']
    lmr_beta      = lmr_chain['beta']

    lmupl_chain   = N.load('%s_MBH_bulge_withlimits.npy' % linmixfilebase)
    lmupl_alpha   = lmupl_chain['alpha']
    lmupl_beta    = lmupl_chain['beta']

    lmnupl_chain  = N.load('%s_MBH_bulge_withoutlimits.npy' % linmixfilebase)
    lmnupl_alpha  = lmnupl_chain['alpha']
    lmnupl_beta   = lmnupl_chain['beta']

else:


    #Use linmix to fit to haring and rix x and ys
    lmhr  = linmix.LinMix(xhr, yhr, xhrerr, yhrerr, K=2)
    lmhr.run_mcmc(silent=False, maxiter=5000)
    lmhr_alpha = lmhr.chain['alpha']
    lmhr_beta  = lmhr.chain['beta']

    N.save('linmix_results_haringrixfit.npy', lmhr.chain)


    #Use linmix to fit to stellar mass and MBH relation from DISKDOM sample
    lmr  = linmix.LinMix(results['stellar mass'], results[mbhcol], results['Err stellar mass'], results[dmbhcol], K=2)
    lmr.run_mcmc(silent=False, maxiter=5000)
    lmr_alpha = lmr.chain['alpha']
    lmr_beta  = lmr.chain['beta']

    N.save('linmix_results_MBH_stellarmass.npy', lmr.chain)



    #Use linmix to fit to upper limits on bulgemass from DISKDOM sample (inlcuding those with no measurement from Simard set to B/T = 1)
    # this one is pretty much all upper limits so sometimes it requires some extra iterations to converge
    lmupl  = linmix.LinMix(bt_mbhs, bt_bulgemass, err_bt_mbhs, err_bt_bulgemass, delta=delta, K=2)
    lmupl.run_mcmc(silent=False, maxiter=40000)
    lmupl_alpha = lmupl.chain['alpha']
    lmupl_beta  = lmupl.chain['beta']

    N.save('linmix_results_MBH_bulge_withlimits.npy', lmupl.chain)



    #Use linmix to fit to bulgemass from DISKDOM sample assuming no measurement is an upper limit
    lmnupl  = linmix.LinMix(bt_mbhs, bt_bulgemass, err_bt_mbhs, err_bt_bulgemass, K=2)
    lmnupl.run_mcmc(silent=False, maxiter=5000)
    lmnupl_alpha = lmnupl.chain['alpha']
    lmnupl_beta  = lmnupl.chain['beta']

    N.save('linmix_results_MBH_bulge_withoutlimits.npy', lmnupl.chain)



# Now whether we've read in or re-fit, compute the shaded regions

# we're going to be using these a lot so don't keep recalculating them
lmhr_amed   = N.median(lmhr_alpha)
lmhr_bmed   = N.median(lmhr_beta)
lmhr_astd   = lmhr_alpha.std()
lmhr_bstd   = lmhr_beta.std()

lmr_amed    = N.median(lmr_alpha)
lmr_bmed    = N.median(lmr_beta)
lmr_astd    = lmr_alpha.std()
lmr_bstd    = lmr_beta.std()

lmupl_amed  = N.median(lmupl_alpha)
lmupl_bmed  = N.median(lmupl_beta)
lmupl_astd  = lmupl_alpha.std()
lmupl_bstd  = lmupl_beta.std()

lmnupl_amed = N.median(lmnupl_alpha)
lmnupl_bmed = N.median(lmnupl_beta)
lmnupl_astd = lmnupl_alpha.std()
lmnupl_bstd = lmnupl_beta.std()

#hrysb = lmhr_alpha.mean() + xs * lmhr_beta.mean()
#hrysp = (lmhr_alpha.mean()+3*lmhr_alpha.std()) + xs * (lmhr_beta.mean()-3*lmhr_beta.std())
#hrysm = (lmhr_alpha.mean()-3*lmhr_alpha.std()) + xs * (lmhr_beta.mean()+3*lmhr_beta.std())
hrysb = lmhr_amed + (xs * lmhr_bmed)
hrysp = (lmhr_amed+3*lmhr_astd) + xs * (lmhr_bmed-3*lmhr_bstd)
hrysm = (lmhr_amed-3*lmhr_astd) + xs * (lmhr_bmed+3*lmhr_bstd)

#rysb = lmr_alpha.mean() + xs * lmr_beta.mean()
rysb = lmr_amed + (xs * lmr_bmed)
rysp = (lmr_amed+3*lmr_astd) + xs * (lmr_bmed-3*lmr_bstd)
rysm = (lmr_amed-3*lmr_astd) + xs * (lmr_bmed+3*lmr_bstd)

#uplysb = lmupl_alpha.mean() + xs * lmupl_beta.mean()
uplysb = lmupl_amed + (xs * lmupl_bmed)
uplysp = (lmupl_amed+3*lmupl_astd) + xs * (lmupl_bmed-3*lmupl_bstd)
uplysm = (lmupl_amed-3*lmupl_astd) + xs * (lmupl_bmed+3*lmupl_bstd)

#nuplysb = lmnupl_alpha.mean() + xs * lmnupl_beta.mean()
nuplysb = lmnupl_amed + (xs * lmnupl_bmed)
nuplysp = (lmnupl_amed+3*lmnupl_astd) + xs * (lmnupl_bmed-3*lmnupl_bstd)
nuplysm = (lmnupl_amed-3*lmnupl_astd) + xs * (lmnupl_bmed+3*lmnupl_bstd)



nsamp = 10000

# only one delta array for each fit type because alpha, beta for each come as a pair
d_hra    = N.random.randn(nsamp)
d_ra     = N.random.randn(nsamp)
d_upla   = N.random.randn(nsamp)
d_nupla  = N.random.randn(nsamp)

ll_hr    = N.zeros([nsamp, len(xs)])
ll_r     = N.zeros([nsamp, len(xs)])
ll_upl   = N.zeros([nsamp, len(xs)])
ll_nupl  = N.zeros([nsamp, len(xs)])

qhrysp   = N.zeros(len(xs))
qhrysm   = N.zeros(len(xs))
qrysp    = N.zeros(len(xs))
qrysm    = N.zeros(len(xs))
quplysp  = N.zeros(len(xs))
quplysm  = N.zeros(len(xs))
qnuplysp = N.zeros(len(xs))
qnuplysm = N.zeros(len(xs))

for i in range(nsamp):
    ll_hr[i]   = (lmhr_amed   + d_hra[i]*lmhr_astd)     + xs * (lmhr_bmed   - d_hra[i]  *lmhr_bstd)
    ll_r[i]    = (lmr_amed    + d_ra[i]*lmr_astd)       + xs * (lmr_bmed    - d_ra[i]   *lmr_bstd)
    ll_upl[i]  = (lmupl_amed  + d_upla[i]*lmupl_astd)   + xs * (lmupl_bmed  - d_upla[i] *lmupl_bstd)
    ll_nupl[i] = (lmnupl_amed + d_nupla[i]*lmnupl_astd) + xs * (lmnupl_bmed - d_nupla[i]*lmnupl_bstd)

# the above made nsamp separate arrays of A lines. Now we want A arrays of nsamp yvalues
# for each x value
hrsamp   = ll_hr.T
rsamp    = ll_r.T
uplsamp  = ll_upl.T
nuplsamp = ll_nupl.T

# 3 sigma
pctiles = [1, 99]
# 2 sigma
#pctiles = [5, 95]
# 1 sigma
#pctiles = [16, 84]

for i_x, thex in enumerate(xs):
    qhrysm[i_x]   = N.percentile(hrsamp[i_x], pctiles[0])
    qrysm[i_x]    = N.percentile(rsamp[i_x], pctiles[0])
    quplysm[i_x]  = N.percentile(uplsamp[i_x], pctiles[0])
    qnuplysm[i_x] = N.percentile(nuplsamp[i_x], pctiles[0])
    qhrysp[i_x]   = N.percentile(hrsamp[i_x], pctiles[1])
    qrysp[i_x]    = N.percentile(rsamp[i_x], pctiles[1])
    quplysp[i_x]  = N.percentile(uplsamp[i_x], pctiles[1])
    qnuplysp[i_x] = N.percentile(nuplsamp[i_x], pctiles[1])







####################################################
# Just the total mass plot
####################################################
#plot_mbh_mgal_only(xs, rysb, qrysp, rysm, brooke_mtot, brooke_mbh, err_brooke_mtot, err_brooke_mbh, int_mtot, int_mbh, Err_int_mtot, Err_int_mbh, sdss_mtot, sdss_mbh, Err_sdss_mtot, Err_sdss_mbh, hrysb, hrysp, hrysm)



####################################################
# Just the bulge mass plot
####################################################
#plot_mbh_mbulge_only(xs, uplysb, nuplysb, uplysp, uplysm, xhr, yhr, xhrerr, yhrerr, brooke_bulgemass, brooke_mtot, err_brooke_mbh, err_brooke_mbh, int_bulgemass, int_mbh, err_int_bulgemass, Err_int_mbh, bulgemass, sdss_mbh, hrysb, hrysp, hrysm, int_is_meas)





####################################################
# Both plots together
####################################################
plot_mbh_mbulge_mgal(xs, uplysb, uplysp, uplysm, nuplysb, qnuplysp, qnuplysm, xhr, yhr, xhrerr, yhrerr, hrysb, qhrysp, qhrysm, rysb, qrysp, qrysm, brooke_bulgemass, brooke_mtot, brooke_mbh, err_brooke_mtot, err_brooke_mbh, int_bulgemass, int_mtot, int_mbh, err_int_bulgemass, Err_int_mbh, bulgemass, sdss_mtot, sdss_mbh, Err_sdss_mtot, Err_sdss_mbh, int_is_meas )



#
