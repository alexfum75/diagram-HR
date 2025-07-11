#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate a skymap with equatorial grid"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from skyfield.api import Star, load
from skyfield.data import hipparcos, stellarium
from skyfield.projections import build_stereographic_projection
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes

# Design
plt.style.use("dark_background")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# Query object from Simbad
OBJECT = "Alioth"
FOV = 30.0
MAG = 6.5

TABLE = Simbad.query_object(OBJECT)
RA = TABLE['RA'][0]
DEC = TABLE['DEC'][0]
COORD = SkyCoord(f"{RA} {DEC}", unit=(u.hourangle, u.deg), frame='fk5')

print("RA is", RA)
print("DEC is", DEC)

ts = load.timescale()
t = ts.now()

# An ephemeris from the JPL provides Sun and Earth positions.
eph = load('de421.bsp')
earth = eph['earth']

# Load constellation outlines from Stellarium
url = 'constellationship.fab'

with load.open(url) as f:
    constellations = stellarium.parse_constellations(f)

edges = [edge for name, edges in constellations for edge in edges]
edges_star1 = [star1 for star1, star2 in edges]
edges_star2 = [star2 for star1, star2 in edges]

# The Hipparcos mission provides our star catalog.
with load.open(hipparcos.URL) as f:
    stars = hipparcos.load_dataframe(f)

# Center the chart on the specified object's position.
center = earth.at(t).observe(Star(ra_hours=COORD.ra.hour, dec_degrees=COORD.dec.degree))
projection = build_stereographic_projection(center)

# Compute the x and y coordinates that each star will have on the plot.
star_positions = earth.at(t).observe(Star.from_dataframe(stars))
stars['x'], stars['y'] = projection(star_positions)

# Create a True/False mask marking the stars bright enough to be included in our plot.
bright_stars = (stars.magnitude <= MAG)
magnitude = stars['magnitude'][bright_stars]
marker_size = (0.5 + MAG - magnitude) ** 2.0

# The constellation lines will each begin at the x,y of one star and end at the x,y of another.
xy1 = stars[['x', 'y']].loc[edges_star1].values
xy2 = stars[['x', 'y']].loc[edges_star2].values
lines_xy = np.rollaxis(np.array([xy1, xy2]), 1)

# Define the limit for the plotting area
angle = np.deg2rad(FOV / 2.0)
limit = np.tan(angle)  # Calculate limit based on the field of view

# Build the figure with WCS axes
fig = plt.figure(figsize=[6, 6])
wcs = WCS(naxis=2)
wcs.wcs.crpix = [1, 1]
wcs.wcs.cdelt = np.array([-FOV / 360, FOV / 360])
wcs.wcs.crval = [COORD.ra.deg, COORD.dec.deg]
wcs.wcs.ctype = ["RA---STG", "DEC--STG"]

ax = fig.add_subplot(111, projection=wcs)

# Draw the constellation lines
ax.add_collection(LineCollection(lines_xy, colors='#ff7f2a', linewidths=1, linestyle='-'))

# Draw the stars
ax.scatter(stars['x'][bright_stars], stars['y'][bright_stars],
           s=marker_size, color='white', zorder=2)

ax.scatter(RA, DEC, marker='*', color='red', zorder=3)

angle = np.pi - FOV / 360.0 * np.pi
limit = np.sin(angle) / (1.0 - np.cos(angle))

# Set plot limits
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_aspect('equal')

# Add RA/Dec grid lines
ax.coords.grid(True, color='white', linestyle='dotted')

# Set the coordinate grid
ax.coords[0].set_axislabel('RA (hours)')
ax.coords[1].set_axislabel('Dec (degrees)')
ax.coords[0].set_major_formatter('hh:mm:ss')
ax.coords[1].set_major_formatter('dd:mm:ss')

# Title
ax.set_title(f'Sky map centered on {OBJECT}', color='white', y=1.04)

# Save the image
FILE = "chart.png"
plt.savefig(FILE, dpi=100, facecolor='#1a1a1a')