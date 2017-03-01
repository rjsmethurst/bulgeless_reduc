import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy
from astropy.table import Table
from scipy import stats, interpolate, special
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

#datafile = 'all_101_galaxies_z_ur_mtot_mbh_lbol_etc_with_Btot_from_Simard_et_al.fits'
datafile = '/Users/vrooje/Documents/Astro/bbcint_paper/data/spectra/sdss_int_results_table_101_sources_new_bhmasses_newcolnames.fits'

thedata = Table.read(datafile)

# columns of note for these purposes: "MBH" "L bol"
mbhcol = 'MBH_best'
Lcol   = 'L_bol'

# from Elvis et al. (2002)
epsilon = 0.15

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


def get_t_growth(thedata, m_seed):
    # thedata needs to contain Mdot (Msun/yr) and log MBH (Msun)
    # m_seed is the seed mass you want to start with (Msun)
    # I'm going to use a timestep of 10k years
    # output time will be in Gyr
    dt = 10000.
    MBH = pow(10.,thedata[mbhcol])
    # mdot per timestep
    mdot = thedata['Mdot']*dt

    # start at the seed mass and t = 0
    themass = m_seed
    t = 0.

    while themass < MBH:
        # mdot for this timestep if the source were at its current eddington rate
        mdot_edd = get_mdot(1.26e38*themass, epsilon) * dt

        # don't allow super-Eddington growth
        # so if the observed mdot is too high, grow at Eddington until you reach that mdot
        # note this means that a source observed to be super-Eddington will never
        # actually accrete at its observed mdot.
        # If you have sources that are way above Eddington this may be an issue.
        if mdot > mdot_edd:
            themass += mdot_edd
        else:
            themass += mdot
        t += 1.

    #print("source reaches %.1e Msun from %.1e Msun in %.1f Gyr at max mdot %.3f Msun/yr\n" % (MBH, m_seed, t/1.0e5, mdot/dt))

    # t is in units of timesteps, i.e. 1e4 years, but we want to return
    # t in Gyr, so convert
    return t/(1.0e9/dt)


def somestats(x):
    if (np.median(x) > 1000.):
        print("values from %e to %e, with mean %e and median %e" % (min(x), max(x), np.mean(x), np.median(x)))
    else:
        print("values from %f to %f, with mean %f and median %f" % (min(x), max(x), np.mean(x), np.median(x)))




thedata['Mdot'] = [get_mdot(q, epsilon) for q in thedata[Lcol]]


# Here's the thing. Eddington-limited accretion is *really* efficient.
# The doubling time is very short, so within our model where we assume
# Eddington-limited accretion until the point where that matches the observed
# accretion rate, the seed mass is not that important. A seed mass difference
# of orders of magnitude will change the growth time by < 1 Gyr.
# But still, calculate that each time and verify.
thedata['t_seed_1e2'] = [get_t_growth(q, 100.) for q in thedata]
thedata['t_seed_1e3'] = [get_t_growth(q, 1000.) for q in thedata]
thedata['t_seed_1e4'] = [get_t_growth(q, 10000.) for q in thedata]
thedata['t_seed_1e5'] = [get_t_growth(q, 100000.) for q in thedata]
thedata['t_seed_1e6'] = [get_t_growth(q, 1000000.) for q in thedata]
