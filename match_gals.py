import os
from pathlib import Path

import numpy as np
import fitsio
import asdf
from astropy.table import Table

from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from match_searchsorted import match

# simulation info
sim_dir = Path("/global/cfs/cdirs/desi/cosmosim/Abacus/")
n_files = 34
z = 0.2
sim_name = "AbacusSummit_base_c000_ph000"
ph = int(sim_name.split('_ph')[-1])
assert "base" in sim_name # otherwise number of files is different and I am lazy to code up a glob search
fields = ["id"]
subsample_AB = ""

# halo lagrangian position info
halo_lagr_dir = Path("/global/cfs/cdirs/desi/users/boryanah/halo_IC/")

# load tracer mock
mock_dir = Path("/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CubicBox/")
tracer_type = "BGS"
t = Table(fitsio.read(mock_dir / f"{tracer_type}/v0.1/z{z:.3f}/{sim_name}/{tracer_type}_box_ph{ph:03d}.fits"))
gal_halo_id = np.array(t['HALO_ID'], dtype=np.int64)
gal_halo_lagr_pos = np.zeros((len(gal_halo_id), 3), dtype=np.float32)

# loop over all halo catalogs
sum = 0
for i in range(n_files):
    print("file index", i)
    halo_info_fn = sim_dir / f"{sim_name}/halos/z{z:.3f}/halo_info/halo_info_{i:03d}.asdf"
    
    # particle info that needs to be loaded
    subsamples = {}
    subsamples['A'] = True if "A" in subsample_AB else False
    subsamples['B'] = True if "B" in subsample_AB else False
    subsamples['rv'] = False
    subsamples['pid'] = False

    # load halo catalog
    cat = CompaSOHaloCatalog(halo_info_fn, subsamples=subsamples, fields=fields)
    print("loaded catalog")

    # load header information
    Lbox = cat.header['BoxSizeHMpc']
    ppd = cat.header['ppd']
    Mpart = cat.header['ParticleMassHMsun']
    print(Lbox, ppd, Mpart)

    # load halo information
    halo_id = np.array(cat.halos['id'], dtype=np.int64) # very important; otherwise weird behavior in in1d

    # load the lagrangian positions (row-by-row matched)
    halo_lagr_pos = asdf.open(halo_lagr_dir / f"{sim_name}/z{z:.3f}/halo_lagr_pos_A_{i:03d}.asdf")['data']['lagr_pos_avg']
    assert halo_lagr_pos.shape[0] == len(halo_id)
    
    # match the two id's
    ptr = match(gal_halo_id, halo_id)
    matched = ptr != -1
    gal_halo_lagr_pos[matched] = halo_lagr_pos[ptr[matched]]
    sum += np.sum(matched)
    print("how many halos from the mock are matched with halos in this file", np.sum(matched))
    print("what fraction of the halos from the mock are matched with halos in this file", np.sum(matched)/len(matched))
assert len(gal_halo_id) == sum
    
save_path = halo_lagr_dir / f"mocks/{sim_name}/z{z:.3f}/"
os.makedirs(save_path, exist_ok=True)
np.save(save_path / f"halo_lagr_pos_{tracer_type}_box_ph{ph:03d}.npy", gal_halo_lagr_pos)
