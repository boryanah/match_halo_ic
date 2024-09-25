import os, gc, glob
from pathlib import Path

import numpy as np
import argparse
import asdf
from numba import njit

from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from abacusnbody.data.bitpacked import unpack_pids

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph006"
DEFAULTS['sim_dir'] = "/global/cfs/cdirs/desi/cosmosim/Abacus/"
DEFAULTS['save_dir'] = "/global/cfs/cdirs/desi/users/boryanah/halo_IC/"
DEFAULTS['subsample_AB'] = "A" 
DEFAULTS['redshift'] = 0.8

"""
Description: 

This code generates proto halo positions for each halo in a an AbacusSummit box at a given redshift. By default, it uses only the
particle A subsample (3%), but to use particle B (7%), one needs to fetch that from tape (and slightly modify the code below).

Usage:

- Example with base-resolution simulation
`python match_halos.py --sim_dir /global/cfs/cdirs/desi/cosmosim/Abacus/  --sim_name AbacusSummit_small_c000_ph006 --redshift 0.8`
- Example with small-resolution simulation
`python match_halos.py --sim_dir /global/cfs/cdirs/desi/cosmosim/Abacus/small/  --sim_name AbacusSummit_small_c000_ph3000 --redshift 0.5`

"""

@njit
def dist(pos1, pos2, L=None):
    '''
    Calculate L2 norm distance between a set of points
    and either a reference point or another set of points.
    Optionally includes periodicity.
    
    Parameters
    ----------
    pos1: ndarray of shape (N,m)
        A set of points
    pos2: ndarray of shape (N,m) or (m,) or (1,m)
        A single point or set of points
    L: float, optional
        The box size. Will do a periodic wrap if given.
    
    Returns
    -------
    dist: ndarray of shape (N,)
        The distances between pos1 and pos2
    '''
    
    # read dimension of data
    N, nd = pos1.shape
    
    # allow pos2 to be a single point
    pos2 = np.atleast_2d(pos2)
    assert pos2.shape[-1] == nd
    broadcast = len(pos2) == 1
    dist = np.empty(N, dtype=pos1.dtype)

    # loop over all objects
    i2 = 0
    for i in range(N):
        delta = 0.
        for j in range(nd):
            dx = pos1[i][j] - pos2[i2][j]
            if L is not None:
                if dx >= L/2:
                    dx -= L
                elif dx < -L/2:
                    dx += L
            delta += dx*dx
        dist[i] = np.sqrt(delta)
        if not broadcast:
            i2 += 1
    return dist

@njit
def calc_avg_pos(pos, Lbox, dtype=np.float32):
    """
    Compute average position of a halo and account for
    periodic wrapping of the box.
    """
    # read off parameter
    N = pos.shape[0]
    D = pos.shape[1]
    half_box = dtype(Lbox/2.)
    Lbox = dtype(Lbox)

    # initialize avg position array
    avg_pos = np.zeros(D, dtype=dtype)

    # loop over each dimension
    for j in range(D):
        
        # loop over each particle
        for i in range(N):
            
            # compute difference vector in this dimension
            dx = pos[i, j] - pos[0, j]

            # wrap around if on opposite side of the box
            if dx >= half_box:
                avg_pos[j] += (pos[i, j] - Lbox)
            elif dx < -half_box:
                avg_pos[j] += (pos[i, j] + Lbox)
            else:
                avg_pos[j] += pos[i, j]
                
        # take the average over all particles in halo
        avg_pos[j] /= N
        
        # wrap around since pos go from (-L/2, L/2)
        if avg_pos[j] < -half_box:
            avg_pos[j] += Lbox
        elif avg_pos[j] >= half_box:
            avg_pos[j] -= Lbox
    return avg_pos

@njit
def calc_halos_avg_pos(part_pos, npstart, npout, Lbox, dtype=np.float32):
    """
    Compute average positions of all halos and account for
    periodic wrapping of the box.
    """
    # initialize array with averaged halo positions
    halo_avg_pos = np.zeros((len(npstart), 3), dtype=dtype)

    # loop over all halos
    for i in range(len(npstart)):

        # skip if no particle subsamples
        if npout[i] == 0: continue

        # compute the average position for this halo
        nst = npstart[i]
        nout = npout[i]
        halo_avg_pos[i] = calc_avg_pos(part_pos[nst:nst+nout], Lbox, dtype=dtype)
    return halo_avg_pos

def compress_asdf(asdf_fn, table, header):
    """
    Given the file name of the asdf file, the table and the header, compress the table info and save as `asdf_fn'
    """
    # cram into a dictionary
    data_dict = {}
    for field in table.keys():
        data_dict[field] = table[field]

    # create data tree structure
    data_tree = {
        "data": data_dict,
        "header": header,
    }

    # set compression options here
    compression_kwargs=dict(typesize="auto", shuffle="shuffle", compression_block_size=12*1024**2, blosc_block_size=3*1024**2, nthreads=4)
    with asdf.AsdfFile(data_tree) as af, open(asdf_fn, 'wb') as fp: # where data_tree is the ASDF dict tree structure
        af.write_to(fp, all_array_compression='blsc', compression_kwargs=compression_kwargs)


def main(sim_name, redshift, subsample_AB, sim_dir, save_dir, want_sanity=False):

    # path to halos of interest
    halo_path = Path(sim_dir) / sim_name / "halos" / f"z{redshift:.3f}" / "halo_info"
    print(str(halo_path))
    save_dir = Path(save_dir)

    # save_dir
    save_path = save_dir / sim_name / f"z{redshift:.3f}"
    os.makedirs(save_path, exist_ok=True)

    # fields that need to be loaded
    fields = ["npstartA", "npoutA"]
    if want_sanity:
        fields.append("x_L2com")

    # if subsample B is needed, simulation data needs to be copied from tape
    assert "B" not in subsample_AB, "Not yet implemented"

    # find all halo info files
    halo_info_fns = sorted(halo_path.glob("halo_info_*.asdf"))

    # loop over each superslab
    for i in range(len(halo_info_fns)):
        print("file", i, len(halo_info_fns))
        print("loading", str(halo_info_fns[i]))

        # define name of final file
        save_fn = save_path / f"halo_lagr_pos_{subsample_AB}_{i:03d}.asdf"

        # particle info that needs to be loaded
        subsamples = {}
        subsamples['A'] = True if "A" in subsample_AB else False
        subsamples['B'] = True if "B" in subsample_AB else False
        subsamples['rv'] = True
        subsamples['pid'] = True
        
        # load halo catalog
        cat = CompaSOHaloCatalog(halo_info_fns[i], subsamples=subsamples, fields=fields)
        print("loaded catalog")

        # load header information
        Lbox = cat.header['BoxSizeHMpc']
        ppd = cat.header['ppd']
        Mpart = cat.header['ParticleMassHMsun']

        # load particle information
        npstart = np.asarray(cat.halos['npstartA'])
        npout = np.asarray(cat.halos['npoutA'])
        if want_sanity:
            x_L2com = np.asarray(cat.halos['x_L2com'])
        pos = np.asarray(cat.subsamples['pos'])
        pid = np.asarray(cat.subsamples['pid'])
        lagr_pos = unpack_pids(pid, box=Lbox, ppd=ppd, pid=False, lagr_pos=True, tagged=False, density=False, lagr_idx=False)['lagr_pos']

        # compute distance between current and IC particle position
        if want_sanity:
            part_dist = dist(pos, lagr_pos, L=Lbox)
            print("part IC dist stats min, max, mean, median", np.min(part_dist), np.max(part_dist), np.mean(part_dist), np.median(part_dist))
            print("number of halos", len(npstart))

        # compute average positions
        halo_avg_lagr_pos = calc_halos_avg_pos(lagr_pos, npstart, npout, Lbox)
        print("calculated avg lagr pos")
        if want_sanity:        
            # compute distance between current and IC halo position
            halo_avg_pos = calc_halos_avg_pos(pos, npstart, npout, Lbox)
            halo_dist = dist(halo_avg_pos, halo_avg_lagr_pos, L=Lbox)
            print("halo IC dist stats min, max, mean, median", np.min(halo_dist[halo_dist > 0.]), np.max(halo_dist), np.mean(halo_dist[halo_dist > 0.]), np.median(halo_dist[halo_dist > 0.]))

            # halo distance
            halo_dist = dist(halo_avg_pos, x_L2com, L=Lbox)
            mask = ~np.isclose(np.product(halo_avg_pos, axis=1), 0.)
            assert len(halo_dist) == len(mask)
            print("halo dist stats min, max, mean, median", np.min(halo_dist[mask]), np.max(halo_dist[mask]), np.mean(halo_dist[mask]), np.median(halo_dist[mask]))

        # record header
        header = {}
        header['sim_name'] = sim_name
        header['subsample_AB'] = subsample_AB
        header['redshift'] = redshift

        # record table
        data = {}
        data['lagr_pos_avg'] = halo_avg_lagr_pos

        # save as asdf file
        print("compressing")
        compress_asdf(save_fn, data, header)
    
class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":

    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_dir', help='Path to simulations', default=DEFAULTS['sim_dir'])
    parser.add_argument('--save_dir', help='Path to save output', default=DEFAULTS['save_dir'])
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--subsample_AB', help='Subsample type', default=DEFAULTS['subsample_AB'])
    parser.add_argument('--redshift', help='Redshift', default=DEFAULTS['redshift'], type=float)
    parser.add_argument('--want_sanity', help='Want a sanity check?', action='store_true')
    args = vars(parser.parse_args())
    main(**args)
