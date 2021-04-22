from astropy.table import Table
import os
import pandas as pd
import numpy as np
import psycopg2

# read in the file containing coordinates
# for the catalog query
filename = "ps1_query_coordinates.txt"
gt = Table.read("ps1_query_coordinates.txt", format="ascii")


# rewrite as dict for easier access of coordinates
# of a specific field by name
gt_dict = {}
for i in range(len(gt)):
    gt_dict[gt["name"][i]] = {"ra": gt["ra_deg"][i], "dec": gt["dec_deg"][i]}


#  define which fields and instruments
all_fields = list(gt_dict.keys())
fields =  all_fields
instruments = ['comCam', 'lsstCam']#  ["comCam"]  # or ['comCam', 'lsstCam']


#  connect to a database
dbname = "lsstdevaosdb1"
host = "lsst-pgsql03.ncsa.illinois.edu"
dbport = 5432

connection = psycopg2.connect(
    dbname=dbname,
    host=host,
    port=dbport,
)


def get_star_catalog(
    connection,
    ra,
    dec,
    field_name,
    instrument="comCam",
    passband="rmeanpsfmag",
    save_file=True,
    output_dir="/project/scichris/aos/ps1_phosim/",
):
    """
    Query the SQL  PS1 database for a particular pointining
    (ra,dec) and field of view (single raft or full array mode).

    Parameters:
    ------------
    connection : active psycopg2 connection to the database
    instrument : lsstCam or comCam, defines the size of the query
                 (0.4 or 2.5 degrees). Case-insensitive (can be
                 'comcam', 'comCam', 'ComCam', etc.).
    passband : magnitudes to return in the query (rmeanpsfmag by
               default).

    Returns:
    ----------
    pandaCat : a pandas table with star catalog


    """
    if instrument.lower() == "lsstcam":
        radius = 2.5  # degrees : for famCam test ...
    elif instrument.lower() == "comcam":
        radius = 0.4  # degrees - R22 only

    querystring = 'SELECT "objid", "ra", "dec", "{}" \
    from "aos_catalog_ps1" where "ra" between {} and {} \
    and "dec" between {} and {} and "rmeanpsfmag" > -900.;'.format(
        passband,
        ra - radius / np.cos(np.radians(dec)),
        ra + radius / np.cos(np.radians(dec)),
        dec - radius,
        dec + radius,
    )
    print("\nRunning query:")
    print(querystring)
    panda_cat = pd.read_sql_query(querystring, connection)

    if save_file:
        table_cat = Table.from_pandas(panda_cat)
        file_name = f"{field_name}_PS1_{instrument.lower()}.txt"
        file_path = os.path.join(output_dir, file_name)
        table_cat.write(file_path, format="ascii", overwrite=True)
        print(f"Saved as {file_path}")

    return panda_cat


# make sure that the root dir exists
root_dir = "/project/scichris/aos/ps1_phosim/"
if not os.path.exists(root_dir):
    os.makedirs(root_dir)


# begin loop
catalogs = {}
for instrument in instruments:
    catalogs[instrument] = {}

    for field_name in fields:
        ra = gt_dict[field_name]["ra"]
        dec = gt_dict[field_name]["dec"]
        print('\n Obtaining star catalog for ', field_name)

        # query database for sources within a given radius,
        # store as ASCII file and return as a pandas dataframe
        panda_cat = get_star_catalog(
            connection, ra, dec, field_name, instrument, output_dir=root_dir
        )

        # add to the dictionary for fast access
        catalogs[instrument][field_name] = panda_cat


# NB:  here we only do 0.25 sec exposure to quickly
# test if the image is properly made
# in generating large catalogs, change to 15 sec.
def write_phosim_header(
    output,
    ra,
    dec,
    camconfig=3,
    opsim_filter=3,
    mjd=59580.0,
    exposure=0.25,
    obsid=9006002,
    defocal=False,
):
    """Write to file handle the phosim header"""
    output.write("Opsim_obshistid {}\n".format(obsid))
    output.write("Opsim_filter {}\n".format(opsim_filter))
    output.write("mjd {}\n".format(mjd))
    output.write("SIM_SEED {}\n".format(1000))
    output.write("rightascension {}\n".format(ra))
    output.write("declination {}\n".format(dec))
    output.write("rotskypos {}\n".format(0.000000))
    output.write("rottelpos {}\n".format(0))
    output.write("SIM_VISTIME {}\n".format(exposure))
    output.write("SIM_NSNAP {}\n".format(1))
    output.write("moonphase {}\n".format(0.0))
    output.write("moonalt {}\n".format(-90))
    output.write("sunalt {}\n".format(-90))
    output.write("Opsim_rawseeing {}\n".format(-1))
    if defocal:
        output.write("move 10 -1500.0000\n")  # write the defocal movement
    output.write("camconfig {}\n".format(camconfig))


def write_phosim_inst_file(
    panda_cat,
    ra,
    dec,
    phosim_file="stars.cat",
    passband="r",
    out_dir="./",
    camconfig=3,
    opd=False,
    exposure=15,
    obsid=9006002,
    position="defocal",
):
    """Generate a phosim instance catalog from a Pandas dataframe"""

    passbands = {
        "u": "umeanpsfmag",
        "g": "gmeanpsfmag",
        "r": "rmeanpsfmag",
        "i": "imeanpsfmag",
        "z": "zmeanpsfmag",
        "y": "ymeanpsfmag",
    }

    out_file = os.path.join(out_dir, phosim_file)

    filterid = list(passbands.keys()).index(passband)

    if position == "defocal":
        defocal = True
    else:
        defocal = False

    with open(out_file, "w") as output:
        write_phosim_header(
            output,
            ra,
            dec,
            camconfig=camconfig,
            opsim_filter=filterid,
            mjd=59580.0,
            exposure=exposure,
            obsid=obsid,
            defocal=defocal,
        )
        if opd:
            output.write("opd 0 {} {} 500".format(ra, dec))
        else:
            for index, row in panda_cat.iterrows():
                output.write(
                    "object {} {} {} {} ../sky/sed_flat.txt \
0.0 0.0 0.0 0.0 0.0 0.0 star 12.0 none none\n".format(
                        int(row["objid"]),
                        row["ra"],
                        row["dec"],
                        row[passbands[passband]],
                    )
                )
    print("Saved as ", out_file)
    return out_file


def write_phosim_cmd_file(root_dir, file_name="phosim.cmd",
                          no_background=True):
    """
    Write a phosim physics commands file
    that is used to generate simulated images.
    Full reference https://bitbucket.org/phosim/\
    phosim_release/wiki/Physics%20Commands
    and https://bitbucket.org/phosim/phosim_release/\
    wiki/Background%20Approximations
    """
    out_file = os.path.join(root_dir, file_name)
    with open(out_file, "w") as output:

        # choose whether to simulate background or not
        if no_background:
            output.write("backgroundmode 0\n")
        else:
            output.write("quickbackground\n")

        # write other options that are usually unchanged
        output.write("raydensity 0\n")  # X is the density of cosmic rays
        output.write("perturbationmode 1\n")
        # X=0: ideal design;
        # X=1: ideal design+deliberate control
        # (move commands);
        # X=2: ideal design + non-controlled environmental/fabrication;
        # X=3: all on (ideal design + deliberate control +
        # non-controlled); Default is X=3
        output.write(
            "trackingmode 0\n"
        )  # X=1: tracking on; X=0: tracking turned off; Default is X=1
        output.write("cleartracking\n")  # resets all tracking to 0
        output.write("clearclouds\n")  # resets all clouds to 0
        output.write(
            "lascatprob 0.0\n"
        )  # (X is the large angle scattered light fraction)
        output.write(
            "contaminationmode 0\n"
        )  # X=1 dust/condensation ; X=0 perfect surfaces
        output.write("diffractionmode 1\n")
        # X=1: monte carlo diffraction;
        # X=0 diffraction off;
        # X=2 full FFT approach (very slow); Default is X=1
        output.write("straylight 0\n")
        output.write(
            "detectormode 0\n"
        )  # X=1: charge diffusion in detector on; X=0 charge diffusion off
        output.write("centroidfile 1\n")  # X=1: output centroid files

    print(f"Saved as {out_file}")
    return out_file


# write the phosim .cmd file
# just once, since we use
# the same physics commands
# file for all simulations
cmd_file = write_phosim_cmd_file(root_dir,
                                 file_name="noBkgnd.cmd", no_background=True)


# write the star catalogs as .inst files:
# {field_name} X {field_size} X {defocal}
# {med,high,low,Baade} X {comCam, lsstCam} X {focal, defocal}


for instrument in instruments:
    for field_name in fields:
        for position in ["focal", "defocal"]:
            ra = gt_dict[field_name]["ra"]
            dec = gt_dict[field_name]["dec"]

            # access the pandas catalog via dict
            panda_catalog = catalogs[instrument][field_name]

            inst_file = write_phosim_inst_file(
                panda_catalog,
                ra,
                dec,
                phosim_file=f"stars_{instrument}_PS1_{field_name}_{position}.inst",
                passband="r",
                out_dir=root_dir,
                exposure=0.25,
                obsid=9006002,
                position=position,
            )
