import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import matplotlib.pyplot as plt4
import pandas as pd
import seaborn as sns
from skyfield.api import load
from skyfield.data import hipparcos
from matplotlib import colors
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier


# https://vizier.cds.unistra.fr/viz-bin/VizieR-4
#https://colab.research.google.com/github/guitar79/OA-2018/blob/master/07_2_Drawing_H_R_Diagram.ipynb#scrollTo=EY-BQN5YMPrf
#https://vizier.cds.unistra.fr/cgi-bin/VizieR?-source=I/239/hip_main

filename = 'asu.tsv'

if __name__ == "__main__":
    # https://wwwhip.obspm.fr/heritage/hipparcos/SandT/hip-SandT.html
    cols = ['_Glon', '_Glat', 'HIP', 'B-V', 'SpType']
    df_spectral = pd.read_csv(filename, usecols=cols, sep=';', low_memory=True)

    vizier = Vizier()
    catalog_name = 'I/239/hip_main'
    catalog_list = vizier.find_catalogs(catalog_name)

    if len(catalog_list) != 1:
        print(f"Error: No catalogs {catalog_name} found")
        sys.exit(1)

    Vizier.ROW_LIMIT = 999999999
    catalogs = Vizier.get_catalogs(catalog_name)
    print (catalogs)

    hipparcos_table = catalogs[0] # The main Hipparcos catalog is usually the first table in the list

    print (f'Catalog table columns {hipparcos_table.colnames}')
    df = pd.DataFrame.from_records(hipparcos_table.as_array())
    print(f"Catalog: {catalog_name} Total number of stars: {df.shape}")

    df.dropna(inplace = True)
    df['_DE.icrs'] = pd.to_numeric(df['_DE.icrs'], errors='coerce')
    df['_RA.icrs'] = pd.to_numeric(df['_RA.icrs'], errors='coerce')
    df['RAICRS'] = pd.to_numeric(df['RAICRS'], errors='coerce')
    df['DEICRS'] = pd.to_numeric(df['DEICRS'], errors='coerce')
    df['pmRA'] = pd.to_numeric(df['pmRA'], errors='coerce')
    df['pmDE'] = pd.to_numeric(df['pmDE'], errors='coerce')
    df['B-V'] = pd.to_numeric(df['B-V'], errors='coerce')
    df['e_Plx'] = pd.to_numeric(df['e_Plx'], errors='coerce')
    df['Vmag'] = pd.to_numeric(df['Vmag'], errors='coerce')
    df['Plx'] = pd.to_numeric(df['Plx'], errors='coerce')
    df['Notes'] = df['Notes'].apply(str)
    df['DEdms'] = df['DEdms'].astype(str)
    df['RAhms'] = df['RAhms'].astype(str)
    df['Plx'] = df['Plx'].astype(np.float32)
    df['Plx'] = df['Plx'].apply(lambda x: x if x > 0.0 else None).dropna()

    # https://docs.astropy.org/en/stable/coordinates/skycoord.html
    ra = df['_RA.icrs'].values
    dec = df['_DE.icrs'].values
    c = SkyCoord(ra=ra, dec=dec, unit= u.deg, frame='icrs')
    l_plot, b_plot = c.ra.wrap_at(180 * u.deg).radian, c.dec.radian

    plt.figure(figsize=(18, 10))
    plt.subplot(111, projection='aitoff')
    plt.scatter(l_plot, b_plot, s=2, alpha=0.3)
    plt.grid(True)
    plt.axhline(0, color='red', linestyle='--', linewidth=0.8, label='Galactic Equator')
    plt.xlabel('Galactic Longitude', fontsize=16)
    plt.ylabel('Galactic Latitude', fontsize=16)
    plt.title('Distribution of Hipparcos stars over the celestial sphere', fontsize=16)
    plt.tight_layout()
    plt.show()

    df_merge = pd.merge(df, df_spectral, on='HIP', how='inner')
    df['SpType2'] = df_merge['SpType'].str[:2]
    df['SpType2'].dropna(inplace=True)

    blue = '#0000CD'
    bright_red = '#EE4B2B'
    yellow = '#FFFF00'
    orange = "#FFA500"
    light_blue = "#ADD8E6"
    light_yellow = '#FFFFE0'
    silver = '#C0C0C0'
    color_dict = {
        "O": blue,
        "B": light_blue,
        "A": silver,
        "F": light_yellow,
        "G": yellow,
        "K": orange,
        "M": bright_red
    }
    col_sp =[blue, light_blue, silver, light_yellow, yellow, orange, bright_red]

    df['tmp_SpType'] = df['SpType2'].str[:1].astype(str).apply(lambda x: x.upper()).dropna()
    df = df[df['tmp_SpType'].isin(color_dict.keys())]
    df['SpectralTypeColor'] = df['tmp_SpType'].apply(lambda x: color_dict[x[0].upper()] if x[0].upper() in color_dict else None).dropna()
    df['Spectral'] = df['tmp_SpType'].apply(lambda x: x.upper() if x[0].upper() in color_dict else None).dropna()

    plt2.figure(figsize=(18, 10))
    fig, ax = plt2.subplots()
    ax = sns.countplot(x="Spectral", data=df, palette = col_sp, legend=False, hue="Spectral")
    ax.grid(True)
    plt2.xlabel('Spectral class', fontsize=16)
    plt2.ylabel('Star count', fontsize=16)
    plt2.title('Distribution of spectral  in Hipparcos catalog', fontsize=16)
    plt2.tight_layout()
    plt2.show()

    df['M_V'] = df['Vmag'] + 5 * np.log10(df['Plx'] / 100)
    # Ballesteros' formula
    # https://itu.physics.uiowa.edu/labs/advanced/photometry-globular-cluster/part-2-finding-temperature-and-spectral-type
    fig, ax = plt.subplots(figsize=(8, 10))
    df['T'] = 4600 * (1 / (0.92 * df['B-V'] + 1.7)) + 1700 * (1 / (0.92 * df['B-V'] + 0.62))
    plt3.scatter(np.log10(df['T']), df['M_V'], s=2, edgecolors='none', alpha=0.3)
    ax.set_xlim(max(np.log10(df['T']) - 0.1), min(np.log10(df['T'])) + 0.25)
    ax.set_ylim(max(df['M_V']), min(df['M_V'])-1)
    plt3.title("H-R Diagram", fontsize=14)
    plt3.ylabel("Absolute Magnitude", fontsize=14)
    plt3.xlabel("Temperature (log(K))", fontsize=14)
    ax.grid()
    plt3.show()

    #https: // wwwhip.obspm.fr / heritage / hipparcos / SandT / images / hipphr.jpg
    fig, ax = plt.subplots(figsize=(8, 10))
    plt4.scatter(df['B-V'], df['M_V'], s=2, edgecolors='none', alpha=0.3)
    ax.set_xlim(min((df['B-V'])), max(df['B-V']) - 3.5)
    ax.set_ylim(max(df['M_V']), min(df['M_V'])-1)
    plt4.title("H-R Diagram", fontsize=14)
    #plt4.ylabel("Absolute Magnitude", fontsize=14)
    #plt4.xlabel("Temperature (log(K))", fontsize=14)
    ax.grid()
    plt4.show()

