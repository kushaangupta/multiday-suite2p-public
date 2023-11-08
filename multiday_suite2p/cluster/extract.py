import os
import numpy as np
from pathlib import Path
from suite2p.io import compute_dydx, BinaryFileCombined
from suite2p.extraction.masks import create_masks
from suite2p.extraction.extract import extract_traces
from suite2p.extraction import dcnv
from multiday_suite2p.process import demix_traces

def extract_traces_session(multiday_folder,data_folder,bin_folder,data_path):
    """Main extraction function. Collects traces based on registered masks for one session (for parallelization)

    Args:
        multiday_folder (string): [description]
        data_folder ([type]): [description]
        bin_folder ([type]): [description]
        session_ind ([type]): [description]
    """
    # convert to path
    multiday_folder = Path(multiday_folder)
    data_folder = Path(data_folder)
    bin_folder = Path(bin_folder)
    data_path = Path(data_path)

    # create save folder.
    save_folder = multiday_folder/'sessions'/data_path
    save_folder.mkdir(parents=True, exist_ok=True)
    # remove present files.
    if save_folder.is_dir():
        print(f'\nRemoving files in {save_folder}')
        files = save_folder.glob('*')
        for f in files:
            os.remove(f)

    print('\nCollecting data')
    # load info.
    info = np.load(multiday_folder/'info.npy',allow_pickle=True).item()
    # load in all planeX folder ops
    plane_folders=list((bin_folder/data_path/info['suite2p_folder']).glob('plane[0-9]'))
    ops1 = [np.load(plane_folder/'ops.npy', allow_pickle=True).item() for plane_folder in plane_folders]
    # all the registered binaries
    reg_loc = [plane_folder/'data.bin' for plane_folder in plane_folders]
    # plane/ROI positions
    dy, dx = compute_dydx(ops1)
    # plane/ROI sizes
    Ly = np.array([ops['Ly'] for ops in ops1])
    Lx = np.array([ops['Lx'] for ops in ops1])
    LY = int(np.amax(dy + Ly))
    LX = int(np.amax(dx + Lx))

    # find index of cell masks for this data path
    session_ind = None
    for i, j in enumerate(info['data_paths']):
        if j==data_path.as_posix():
            session_ind = i
    if session_ind == None:
        raise NameError(f'Could not find cell masks for {data_path.as_posix()}')
    # load stats of cell masks
    stats_combined = np.load(multiday_folder/'backwards_deformed_cell_masks.npy',allow_pickle=True)[session_ind]
    if 'overlap' not in stats_combined[0]:
        for stat in stats_combined:
            stat['overlap'] = True
    # load combined ops
    ops_file = data_folder/data_path/info['suite2p_folder']/'combined'/'ops.npy'
    ops_combined = np.load(ops_file, allow_pickle=True).item()
    ops_combined['allow_overlap'] = True
    # create masks in global view
    print('\nCreating masks')
    cell_masks, neuropil_masks = create_masks(ops_combined, stats_combined)
    # extract traces in global view
    print('\nExtracting traces')
    with BinaryFileCombined(LY, LX, Ly, Lx, dy, dx, reg_loc) as f:    
        F, Fneu, ops = extract_traces(ops_combined, cell_masks, neuropil_masks, f)
    
    # save files.
    print(f"\nSaving results in {save_folder}..")
    np.save(save_folder/'ops.npy', ops)
    np.save(save_folder/'F.npy', F)
    np.save(save_folder/'Fneu.npy', Fneu)
