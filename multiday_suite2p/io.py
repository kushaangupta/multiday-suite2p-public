import sys
import yaml
import shutil
import os
import re
import json
import numpy as np
from multiday_suite2p.utils import create_mask_img, add_overlap_info
from pathlib import Path
from tqdm import tqdm,trange
import pandas as pd

def find_session_folders(main_folder,days,verbose=False, suite2p_folder_name = "suite2p"):
    """Find folders with suite2p subdirectories in main folder
    
    Arguments:
        main_folder {string} -- Directory that contains all the session folders.
        days {list of strings} -- List of sessions dates to include.
    
    Keyword Arguments:
        verbose {bool} -- Print results. (default: {False})
        suite2p_folder_name {string} -- Suite2p folder name (default: "suite2p)
    
    Returns:
        [list of strings] -- Found directories.
    """
    sessionfolders = []
    for day in days:
        folder = os.path.join(main_folder,day)
        sessions = [id for id in os.listdir(folder) if re.match("^\d{1,2}$",id)
                and os.path.isdir(os.path.join(folder,id,"suite2p")) ]
        for id in sessions: sessionfolders.append(os.path.join(folder,id))
    if verbose:
        print(f"Found {len(sessionfolders)} session folders")
    return sessionfolders

def import_sessions(data_info, settings,verbose = False):
    """Imports session data needed for multiday registration.
    
    Arguments:
        meta_info {dictionary}              -- List of string of the session folder locations
    
    Keyword Arguments:
        suite2p_folder_name {str}   -- Sub-folder name of suite2p information (default: {"suite2p"})
        verbose {bool}              -- If true: prints output of progress  (default: {False})
        cell_prob_threshold {float} -- Threshold for cell classifier at which masks are included (default: {0.85})
    
    Returns:
        [numpy array]               -- Overview images of sessions (size: num_sessions x num_planes) 
                                        (nested dict: "mean_img", "enhanced_img", "max_img")
        [numpy array]               -- Detected functional cell mask information (size: num_sessions x num_planes)
                                        (dict: "xpix": array of x pixel locations, "ypix": array of y pixel locations, 
                                        "ipix": array of linear index locations, "npix": number of pixels in mask,
                                        "lam": lambda mask weights per pixel, "med": median center of mask)
        [list]                      -- list of image sizes per plane (num_planes)
        [list]                      -- list of image cell masks per plane (num_planes)
    """
    # find data folders.
    data_paths = list(Path(data_info['data']['local_processed_root']).glob('[0-9][0-9][0-9][0-9]_[0-9][0-9]_[0-9][0-9]/[0-9]'))
    if not data_paths:
        raise NameError(f'Could not find any data folders in {data_info["data"]["local_processed_root"]}')
    # filter for specific session if requested.
    if data_info['data']['session_selection']:
        data_paths = filter_data_paths(data_paths, data_info['data']['session_selection'])
    # Check if all flagged individual sessions can be found (safety feature).
    short_data_paths = [(Path(data_path.parts[-2])/data_path.parts[-1]).as_posix() for data_path in data_paths]
    for filter_path in data_info['data']['individual_sessions']:
        if filter_path not in short_data_paths:
            raise NameError(f"Could not find requested individual session {filter_path} in session selection")

    cells = []
    cell_masks = []
    images = []
    sessions = []
    for data_path in data_paths:
        # check if this should be skipped and used as individual session (not registered.)
        if (Path(data_path.parts[-2])/data_path.parts[-1]).as_posix() in data_info['data']['individual_sessions']:
            print(f"{Path(data_path.parts[-2])/Path(data_path.parts[-1])}: skipping (individual session)")
            continue
        combined_folder = data_path/data_info["data"]["suite2p_folder"]/'combined'
        # store sessions info.
        sessions.append({'date': data_path.parts[-2], 'session_id': data_path.parts[-1]})
        # check if combined folder exists.
        if not combined_folder.is_dir():
            raise NameError(f'Could not find combined suite2p folder for: {data_path}')
        else:
            # load data.
            ops = np.load(combined_folder/'ops.npy',allow_pickle = True).item()
            stat = np.load(combined_folder/'stat.npy',allow_pickle = True)
            iscell = np.load(combined_folder/'iscell.npy',allow_pickle = True)
            # Get images.
            images.append({'mean_img':ops['meanImg'], 'enhanced_img':  ops['meanImgE'], 'max_img': ops['max_proj']} )
            # select valid cells and only needed fields.
            selected_cells =  [{ key: mask[key] for key in ["xpix","ypix","lam","med","radius","overlap"]}  for icell, mask in enumerate(stat) if (iscell[icell,1]>
                settings['cell_detection']['prob_threshold']) and (mask['npix']<settings['cell_detection']['max_size']) ]
            # filter based on margin to stripe edge.
            filtered_cells = []
            for cell in selected_cells:
                flag = True
                for border in settings['cell_detection']['stripe_borders']:
                    if (cell['med'][1]>=(border-settings['cell_detection']['stripe_margin'])) & (cell['med'][1]<=(border+settings['cell_detection']['stripe_margin'])):
                        flag = False
                if flag:
                    filtered_cells.append(cell)
            cells.append(filtered_cells)
            if verbose:
                print(f"{Path(data_path.parts[-2])/Path(data_path.parts[-1])} contained info for {len(filtered_cells)} cells")
            # create images.
            im_size = images[0]['mean_img'].shape
            cell_masks.append(create_mask_img(filtered_cells,im_size,mark_overlap=True))
    # create folder to hold registration data files.
    (Path(data_info['data']['local_processed_root'])/data_info['data']['output_folder']/'registration_data').mkdir(parents=True, exist_ok=True)
    return sessions, images, cells, im_size, cell_masks


def export_masks_and_images(deformed_cell_masks, cell_templates, trans_images, images, sessions, data_info, settings):
    """exports masks, images and general info to multi-day folder.
    
    Arguments:
        cell_masks {list}               -- backwards transformed cell mask information for specific session (size: num_sessions)
                                        contains embedded list of dictionaries (size: num_cells) 
                                        (dict: "xpix": array of x pixel locations, "ypix": array of y pixel locations, 
                                        "ipix": array of linear index locations, "npix": number of pixels in mask,
                                        "lam": lambda mask weights per pixel)
        templates {list}                -- template cells mask information (num_cells)
        sessions {list of dict.}        -- session info from import_sessions
        settings
    """
    # (create) main output folder.
    output_folder = Path(data_info['data']['local_processed_root'])/data_info['data']['output_folder']
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Saving multi-day info in {output_folder}.. ")
    # store general info.
    data_paths = [(Path(session['date'])/session['session_id']).as_posix() for session in sessions]
    info = {'suite2p_folder':data_info['data']['suite2p_folder'],
        'data_paths':data_paths}
    np.save(output_folder/'info.npy', info)
    # store session cell masks
    np.save(output_folder/'backwards_deformed_cell_masks.npy',deformed_cell_masks)
    # store templates.
    np.save(output_folder/'cell_templates.npy',cell_templates)
    # store transformed images.
    np.save(output_folder/'trans_images.npy',trans_images)
    # store original images
    np.save(output_folder/'original_images.npy',images)
    # store demix settings
    np.save(output_folder/'demix_settings.npy',settings['demix'])

def import_settings_file():
    """Imports meta data file with path, animal info, and settings.

    Returns:
        dictionary: contents of meta data file
    """
    file_loc = select_meta_file()
    with open(file_loc) as data_file:
            meta_info = yaml.load(data_file, Loader=yaml.FullLoader)
    return meta_info

def registration_data_folder(settings):
    return Path(settings['data']['local_processed_root'])/settings['data']['output_folder']/'registration_data'

def filter_data_paths(data_paths,data_selection):
    """Filter data paths according to selection filters (e.g [['2020_01_01/0',2020_01_10'],['2020_01_02']])
    Args:
        data_paths (list of Path): data paths to filter through
        data_selection ( see above ): filter criteria.
    """
    def data_path_to_datetime(data_path):
        """ Helper function. Converts data paths to DateTimeIndex value.
            session index is set as number of microseconds since start of day. 
            0 microseconds thus means that no session was given.

            These values can be used for range filtering.
        """
        if len(data_path.parts)==1:
            date = data_path
            time = np.timedelta64(0,'m')
        else:
            date = data_path.parts[-2]
            time = np.timedelta64(data_path.parts[-1],'us')
        dt = pd.DatetimeIndex([np.datetime64(f"{str.replace(f'{date}','_','-')}") + time])
        return dt
    # holds selected. 
    selected_data_paths = []
    for data_path in data_paths:
        data_path_dt = data_path_to_datetime(data_path)
        for filter_pattern in data_selection:
            filter_dt = [data_path_to_datetime(Path(f)) for f in filter_pattern]
            # specific date/session.
            if len(filter_pattern)==1:
                # single: no session given (select all)
                if filter_dt[0].microsecond==0:
                    if filter_dt[0].date == data_path_dt.date:
                        selected_data_paths.append(data_path)
                # single: session specified.
                else:
                    if filter_dt[0] == data_path_dt:
                        selected_data_paths.append(data_path)
            # date/session range
            else:
                # if no session specified for max date then select all.
                if filter_dt[1].microsecond==0:
                    filter_dt[1] += np.timedelta64(10,'h')
                if (data_path_dt>=filter_dt[0]) & (data_path_dt<=filter_dt[1]):
                    selected_data_paths.append(data_path)
    return np.sort(np.unique(np.array(selected_data_paths))).tolist()
