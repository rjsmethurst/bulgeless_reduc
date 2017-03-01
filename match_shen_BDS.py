import sys, os, math
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy import special
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy import units as un
#from astropy.utils.exceptions import AstropyWarning

plt.rc('figure', facecolor='none', edgecolor='none', autolayout=True)
plt.rc('path', simplify=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('axes', labelsize='large', facecolor='none', linewidth=0.7, color_cycle = ['k', 'r', 'g', 'b', 'c', 'm', 'y'])
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')
plt.rc('lines', markersize=4, linewidth=1, markeredgewidth=0.2)
plt.rc('legend', numpoints=1, frameon=False, handletextpad=0.3, scatterpoints=1, handlelength=2, handleheight=0.1)
plt.rc('savefig', facecolor='none', edgecolor='none', frameon='False')

params =   {'font.size' : 11,
            'xtick.major.size': 8,
            'ytick.major.size': 8,
            'xtick.minor.size': 3,
            'ytick.minor.size': 3,
            }
plt.rcParams.update(params)


Lsol = 3.846E33*un.g*un.cm**2/un.s**3

# from Elvis et al. (2002)
epsilon = 0.15


#gal = Table.read('/Users/becky/Projects/followup_gv/bpt/MPA_JHU_MASS_SFR.fit', format='fits')


# read in an abbreviated version of the Shen et al. quasar sample that already
# has the 38 matches between it and the BBCINT sample removed
shen_table = '/Users/vrooje/Documents/Astro/catalogs_general/Shen_etal_SDSS_quasars/Shen_etal_11/Shen_etal_2011_SDSS_quasars_catalog_lowz_not_in_bbcint.fits'


shentable_out = '/Users/vrooje/Documents/Astro/bbcint_paper/data/spectra/shen_matched_to_sdss_int_results.fits'


datatable = '/Users/vrooje/Documents/Astro/bbcint_paper/data/spectra/sdss_int_results_table_101_sources_new_bhmasses_newcolnames.fits'
# this is quite an all-inclusive table and it includes results from stuff we've done in other scripts,
# like computing BH masses, galaxy masses, and also external stuff like Simard et al.
# bulge-to-total ratios.
#
# The relevant columns here are:
# name, ID, RA, Dec, spectrum_source, z, Err z, stellar mass, Err stellar mass, L bol, Err L bol, (B/T)r, e_(B/T)r, MBH_best, dMBH_hi_best, dMBH_lo_best
cols_keep = ['name', 'ID', 'RA', 'Dec', 'spectrum_source', 'z', 'Err_z', 'stellar_mass', 'Err_stellar_mass', 'L_bol', 'Err_L_bol', 'BT_r', 'e_BT_r', 'MBH_best', 'dMBH_hi_best', 'dMBH_lo_best']

# if you want to re-select candidates, use this. If not, =False and it will read from shentable_out
reselect = True






#note: you can derive this from the definition of L = dE/dt
# and dE = ec^2 dm where e is the radiative efficiency
# Mdot = 2.22e-9*(MBH/Msun)*(Edd/e)  [solar masses/year]
# (Edd is the Eddington ratio & has the BH mass in it)
# Mdot doesn't actually depend on BH mass at all, but on luminosity
# so re-write:
# with bolometric luminosity in cgs (erg/s):
# Mdot = 1.762e-47 * L_bol / e [solar masses/year]
# or log(Mdot) = log(1.762) - 47. + log(L_bol) - log(e)
# or log(e Mdot) = log(1.762) - 47. + log(L_bol)
# define epsilon 0.15
# set lMdot  = lg(1.762) - 47. + lLbol - lg($epsilon)
# set leMdot = lg(1.762) - 47. + lLbol
# set Mdot   = 10**lMdot
# set eMdot  = 10**leMdot

# L = dE/dT and E = mc^2 * e
# L = dM c^2/dt * e
# dM/dT = L / c^2 / e
# c = 29979245800 cm/s, e = 0.15 (below)
# and L in cgs, so that is going to come out in grams per second
# which isn't super useful
# 1 Msun = 1.989e33 g
# 1 year = 3.1536e7 s
# so with the answer in g/s --> Msun/yr:
# (g/s) * (3.1536e7 s/yr) / (1.989e33 g/Msun) --> 1.5855e-26
# so mdot = L / e * 1.5855e-26/(29979245800^2) = 1.764e-47 * L / e [Msun/yr]

def get_mdot(Lbol, epsilon):
    return 1.762e-47 * Lbol / epsilon


def somestats(x):
    if (np.median(x) > 1000.):
        print("values from %e to %e, with mean %e and median %e" % (min(x), max(x), np.mean(x), np.median(x)))
    else:
        print("values from %f to %f, with mean %f and median %f" % (min(x), max(x), np.mean(x), np.median(x)))




# columns of note for these purposes: "MBH" "L bol"
mbhcol = 'MBH_best'
Lcol   = 'L_bol'


thedata  = Table.read(datatable)
shen_all = Table.read(shen_table)


thedata['L_edd']  = [1.26e38 * 10**q for q in thedata[mbhcol]]
shen_all['L_edd'] = [1.26e38 * 10**q for q in shen_all['logBH']]

thedata['L_edd'].unit  = (un.cm**2 * un.g/un.s**2)
shen_all['L_edd'].unit = (un.cm**2 * un.g/un.s**2)

thedata['eddrat']  = thedata['L_bol'] / thedata['L_edd']
shen_all['eddrat'] = 10**(shen_all['logLbol'] - np.log10(shen_all['L_edd']))

thedata['eddrat'].unit  = ''
shen_all['eddrat'].unit = ''

thedata['Mdot']  = np.array([get_mdot(q, epsilon)     for q in thedata['L_bol']])
shen_all['Mdot'] = np.array([get_mdot(10**q, epsilon) for q in shen_all['logLbol']])

thedata['Mdot'].unit  = un.solMass/un.yr
shen_all['Mdot'].unit = un.solMass/un.yr


z_nmatch = 0.118

if reselect:

    # I manually determined that z = 0.118 is the redshift at which there are
    # at least as many Shen galaxies as our galaxies below that z
    # so we just have to take all of them below that z
    zbins_max    = np.linspace(z_nmatch, max(thedata['z'])+0.02, 15)
    dzbins       = zbins_max[2]-zbins_max[1]
    zbins_min    = zbins_max - dzbins
    zbins_min[0] = 0.00

    i_shen_select = []
    i_shen        = np.arange(0, len(shen_all['z']), dtype='int')

    for i, zmin in enumerate(zbins_min):
        zmax = zbins_max[i]

        # how many of our sources are in this bin?
        n_bbc = sum((thedata['z'] >= zmin) & (thedata['z'] < zmax))

        if n_bbc > 0:
            # isolate the Shen sources that qualify for random selection
            in_zrange = (shen_all['z'] >= zmin) & (shen_all['z'] < zmax)
            # operating on the basis that I have to sample indices
            i_candidates = i_shen[in_zrange]

            # sample randomly without replacement
            i_selected = np.random.choice(i_candidates, size=n_bbc, replace=False)

            i_shen_select.extend(i_selected)

    # now find the selected rows
    i_shen_selected = np.in1d(i_shen, i_shen_select)

    shen_selected = shen_all[i_shen_selected]

    # write the file so we don't have to do all this again
    print("Printing to %s...\n" % shentable_out)
    shen_selected.write(shentable_out)
    if shentable_out.endswith('.fits'):
        shen_selected.write(shentable_out.replace('.fits', '.csv'))
    if shentable_out.endswith('.csv'):
        shen_selected.write(shentable_out.replace('.csv', '.fits'))

else:
    print("Reading from %s...\n" % shentable_out)
    shen_selected = Table.read(shentable_out)



####################################################
# Compute and print some basic stats
#

# recall z_match is the redshift at which we can select enough Shen galaxies
# to match the z distribution in our sample (Shen lowlowz is really light on numbers)
# so this subselection is designed to check whether any differences we do see are
# due to the fact we've had to select a slightly higher-z subset of matching
# galaxies from the Shen et al. sample, at these low redshifts.
# have a look at the redshift histogram below and you'll see what I mean.
higherz_data =       thedata['z'] > z_nmatch
higherz_shen = shen_selected['z'] > z_nmatch

DBH,  pBH  = ks_2samp(thedata[mbhcol], shen_selected['logBH'])
Dbol, pbol = ks_2samp(np.log10(thedata['L_bol']), shen_selected['logLbol'])
Dedd, pedd = ks_2samp(np.log10(shen_selected['eddrat']), np.log10(thedata['eddrat']))
Dz,   pz   = ks_2samp(thedata['z'], shen_selected['z'])

sigBH  = special.erfcinv(pBH)*np.sqrt(2)
sigbol = special.erfcinv(pbol)*np.sqrt(2)
sigedd = special.erfcinv(pedd)*np.sqrt(2)
sigz   = special.erfcinv(pz)*np.sqrt(2)

qDBH,  qpBH  = ks_2samp(thedata[mbhcol][higherz_data], shen_selected['logBH'][higherz_shen])
qDbol, qpbol = ks_2samp(np.log10(thedata['L_bol'][higherz_data]), shen_selected['logLbol'][higherz_shen])
qDedd, qpedd = ks_2samp(np.log10(thedata['eddrat'][higherz_data]), np.log10(shen_selected['eddrat'][higherz_shen]))
qDz,   qpz   = ks_2samp(thedata['z'][higherz_data], shen_selected['z'][higherz_shen])

qsigBH  = special.erfcinv(qpBH)*np.sqrt(2)
qsigbol = special.erfcinv(qpbol)*np.sqrt(2)
qsigedd = special.erfcinv(qpedd)*np.sqrt(2)
qsigz   = special.erfcinv(qpz)*np.sqrt(2)

print("\nRedshifts KS p = %.2e (%.1f sigma)" % (pz, sigz))
print(" Higher-z-only Redshifts KS p = %.2e (%.1f sigma)" % (qpz, qsigz))

print("\nOur BH masses:")
somestats(thedata[mbhcol])

print("\nMatched Shen et al. BH masses:")
somestats(shen_selected['logBH'])

print("\nKS between BH mass samples: %.2e (%.1f sigma)" % (pBH, sigBH))
print(" Higher-z-only KS between BH mass samples: %.2e (%.1f sigma)\n" % (qpBH, qsigBH))

print("\nOur L_bol:")
somestats(np.log10(thedata['L_bol']))

print("\nMatched Shen et al. L_bol:")
somestats(shen_selected['logLbol'])

print("\nKS between L_bol samples: %.2e (%.1f sigma)" % (pbol, sigbol))
print(" Higher-z-only KS between L_bol samples: %.2e (%.1f sigma)\n" % (qpbol, qsigbol))

print("\nOur L/Ledd:")
somestats(thedata['eddrat'])

print("\nMatched Shen et al. L/Ledd:")
somestats(shen_selected['eddrat'])

print("\nKS between Eddington ratio samples: %.2e (%.1f sigma)" % (pedd, sigedd))
print(" Higher-z-only KS between Eddington ratio samples: %.2e (%.1f sigma)\n" % (qpedd, qsigedd))



################################################################################
################################################################################
# Lbol, MBH, eddrat histograms
#
plt.figure(figsize=(10,3.5))

ax1 = plt.subplot(131)
h1 = ax1.hist(shen_selected['logLbol'], bins=12, range=(43,47), histtype='step', color='k', linestyle='dashed', normed=True)
#ax1.hist(qsor['(B/T)r'], bins=15, range=(0,1), histtype='step', color='k', linestyle='dashed', normed=True)
h2 = ax1.hist(np.log10(thedata['L_bol']), bins=12, range=(43,47), histtype='step', color='k', normed=True)
ymax = max(np.append(h1[0], h2[0]))
ax1.set_xlabel(r'$\log (L_{bol} / [\rm{erg}$ $\rm{s}^{-1}])$')
ax1.set_ylabel(r'$\rm{normalised}$ $\rm{density}$')
ax1.tick_params('y', labelleft='off')
ax1.set_xlim((43.1, 46.9))
ax1.set_ylim(0., ymax*1.1)
ax1.minorticks_on()

ax2 = plt.subplot(132)
h1 = ax2.hist(shen_selected['logBH'], bins=12,  range=(5.5,10.5), histtype='step', color='k', linestyle='dashed', normed=True)
#ax2.hist(qsor['LOGBH'], bins=15, range=(5.5,10.5), histtype='step', color='k', linestyle='dashed', normed=True)
h2 = ax2.hist(thedata[mbhcol], bins=12, range=(5.5,10.5), histtype='step', color='k', normed=True)
ymax = max(np.append(h1[0], h2[0]))
ax2.set_xlabel(r'$\log (M_{BH}/M_{\odot})$')
ax2.set_xlim(5.5, 10.5)
ax2.set_ylim(0., ymax*1.1)
ax2.tick_params('y', labelleft='off')
ax2.minorticks_on()


ax3 = plt.subplot(133)
#ax3.hist(gal['LOGBH'], range=(5.5,10.5), histtype='step', color='k', alpha=0.3, linestyle='dashdot', normed=True)
h2 = ax3.hist(np.log10(thedata['eddrat']), bins=12, range=(-3, 2), histtype='step', color='k', normed=True, label=r'$\textsc{diskdom}$')
h1 = ax3.hist(np.log10(shen_selected['eddrat']), bins=12, range=(-3, 2), histtype='step', color='k', linestyle='dashed', normed=True, label=r'$\rm{S11~matched}$')
ymax = max(np.append(h1[0], h2[0]))
ax3.tick_params('y', labelleft='off')
ax3.set_xlabel(r'$\log (\lambda_{Edd})$')
ax3.minorticks_on()
ax3.set_xlim(-3.95, 3.95)
ax3.set_ylim(0., ymax*1.1)
ax3.legend(loc='upper right')


plt.tight_layout()
plt.subplots_adjust(wspace=0.05)
plt.savefig('diskdom_mbh_lbol_edd_ratio_distributions_compare_with_shen.pdf')

plt.close()
plt.cla()





################################################################################
################################################################################
# redshift histograms
#
plt.figure(figsize=(5.5, 4))
#plt.hist(gal['Z'], range=(0,0.3), histtype='step', color='k', alpha=0.3, linestyle='dashdot', normed=True)
h1 = plt.hist(thedata['z'],range=(0,0.3), histtype='step', color='k', normed=True, label=r'$\textsc{diskdom}$ $\rm{sample}$')
h2 = plt.hist(shen_selected['z'], range=(0,0.3), histtype='step', color='k', linestyle='dashed', normed=True, label=r'$\rm{Shen~et~al.~(2011)~matched}$')

ymax = max(np.append(h1[0], h2[0]))
plt.xlabel(r'$z$')
plt.ylabel(r'$\rm{normalised}$ $\rm{density}$')
plt.tick_params('y', labelleft='off')
plt.ylim(0., ymax*1.1)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('z_all_shen_2011_compare.pdf')

plt.close()





################################################################################
################################################################################
# Lbol vs MBH plot
#
intcolor  = "#228dcc"
shencolor = "#1e8c15"

is_sdss = np.array([q == 'SDSS' for q in thedata['spectrum_source']])
is_int  = np.invert(is_sdss)

# from Simmons et al. (2013)
brooke_mtot          = np.array([10.05, 10.03])
err_brooke_mtot      = np.array([0.1, 0.1])
brooke_mbh           = np.array([7.1, 6.6])
err_brooke_mbh       = np.array([0.13, 0.14])
brooke_lbol          = np.array([44.1, 43.4])
err_brooke_lbol      = np.array([0.3, 0.3])
brooke_BT            = np.array([0.011, 0.022])
brooke_bulgemass     = np.log10(brooke_BT*10**brooke_mtot)
err_brooke_bulgemass = np.sqrt( (np.array([0.15, 0.2]))**2 + (np.array([0.15,0.12]))**2 )



plt.figure(figsize=(5.7, 5))
ax1 = plt.subplot(111)
ax2 = ax1.twinx()

xlimits = (5.5, 10.5)
ylimits = (42.92, 46.43)



# get ylimits for the right side of the plot (though we may do this differently later)
ylimits_mdot  = [get_mdot(10**q, epsilon) for q in ylimits]
ylimits_lmdot = (np.log10(ylimits_mdot[0]), np.log10(ylimits_mdot[1]))

# create and plot lines of constant eddington ratio
lmbh_edd  = np.array([4., 12.])
ledd_1p0  = np.log10(1.26) + 38. + lmbh_edd
ledd_0p1  = ledd_1p0 - 1.
ledd_0p01 = ledd_1p0 - 2.

ax1.plot(lmbh_edd, ledd_1p0,  linestyle='dashed', color='k')
ax1.plot(lmbh_edd, ledd_0p1,  linestyle='dashdot', color='k')
ax1.plot(lmbh_edd, ledd_0p01, linestyle='dotted', color='k')


# plot Shen matched sample
ax1.scatter(shen_selected['logBH'],  shen_selected['logLbol'], marker='^', color='None', edgecolor=shencolor, label='Shen et al. (2011)', s=36)
ax1.errorbar(shen_selected['logBH'], shen_selected['logLbol'], xerr=shen_selected['e_logBH'], yerr=shen_selected['e_logLbol'], ecolor=shencolor, capthick=1, fmt='None', fill_style='None', alpha=0.3, label='_nolegend_')

# plot the Simmons et al. sample
ax1.scatter(brooke_mbh, brooke_lbol, marker='o', color='None', edgecolor='#333333', label='Simmons et al. (2013)', s=49)
ax1.errorbar(brooke_mbh, brooke_lbol, xerr = err_brooke_mbh, yerr=err_brooke_lbol, ecolor='#333333', capthick=1, fmt='None', fill_style='None', alpha=0.3, label='_nolegend_')

e_lLbol = np.log10(thedata['Err_L_bol']+thedata['L_bol']) - np.log10(thedata['L_bol'])
# plot our points
ax1.scatter(thedata[mbhcol][is_sdss],  np.log10(thedata['L_bol'][is_sdss]), marker='x', color='k', label='SDSS spectra', s=36.)
ax1.errorbar(thedata[mbhcol][is_sdss], np.log10(thedata['L_bol'][is_sdss]), xerr = thedata['dMBH_best'][is_sdss], yerr=e_lLbol[is_sdss], ecolor='#333333', capthick=1, fmt='None', fill_style='None', alpha=0.3, label='_nolegend_')


# plot INT sample
ax1.scatter(thedata[mbhcol][is_int],  np.log10(thedata['L_bol'][is_int]), marker='s', color=intcolor, label='INT spectra', s=49.)
ax1.errorbar(thedata[mbhcol][is_int], np.log10(thedata['L_bol'][is_int]), xerr = thedata['dMBH_best'][is_int], yerr=e_lLbol[is_int], ecolor=intcolor, capthick=1, fmt='None', fill_style='None', alpha=0.3, label='_nolegend_')

xt = 7.5
yt = 45.75
angt = 53.25

ax1.text(xt,   yt, r'$\lambda_{Edd} = 1$',    verticalalignment='bottom', horizontalalignment='left', rotation=angt) #, transform=ax1.transAxes)
ax1.text(xt+1, yt, r'$\lambda_{Edd} = 0.1$',  verticalalignment='bottom', horizontalalignment='left', rotation=angt) #, transform=ax1.transAxes)
ax1.text(xt+2, yt, r'$\lambda_{Edd} = 0.01$', verticalalignment='bottom', horizontalalignment='left', rotation=angt) #, transform=ax1.transAxes)

ax1.legend(loc='lower right')


ax1.set_xlim(xlimits)
ax1.set_ylim(ylimits)
ax2.set_ylim(ylimits_lmdot)

ax1.minorticks_on()
ax2.minorticks_on()

ax1.set_xlabel(r'$\log (M_{BH}/M_{\odot})$')
ax1.set_ylabel(r'$\log (L_{bol} / [\rm{erg}$ $\rm{s}^{-1}])$')
ax2.set_ylabel(r'$\log (\dot m/ [M_{\odot}$ $\rm{yr}^{-1}])$')

plt.tight_layout()
plt.savefig('lbol_mbh_sdss_int_shen_matched.pdf')
plt.close()

#
