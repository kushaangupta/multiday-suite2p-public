from getpass import getpass
from ipyfilechooser import FileChooser
import yaml
from pathlib import Path

def select_settings_file(start_dir='../'):
    fc = FileChooser(start_dir)
    fc.use_dir_icons = True
    fc.filter_pattern = '*.yml'
    display(fc)
    return fc

def parse_settings(file, request_pass=False):
    file=Path(file)
    if file.is_file():
        # read yml
        with open(file) as data_file:
            settings = yaml.load(data_file, Loader=yaml.FullLoader)
        # check for important keywords.
        for key in ['server','cell_detection','registration','clustering','demix']:
            if key not in settings:
                raise NameError(f"Could not find key '{key}' in settings file")
        # request password.
        if request_pass:
            settings['server']['password']= getpass('Enter your server password')
        return settings

def parse_data_info(file):
    file=Path(file)
    if file.is_file():
        # read yml
        with open(file) as data_file:
            settings = yaml.load(data_file, Loader=yaml.FullLoader)
        # check for important keywords.
        for key in ['data','animal']:
            if key not in settings:
                raise NameError(f"Could not find key '{key}' in settings file")
        # check if server_processed_root and server_bin_root are supplied. Otherwise make same as local
        if len(settings['data']['server_processed_root'])==0:
            settings['data']['server_processed_root'] = settings['data']['local_processed_root'] 
        if len(settings['data']['server_bin_root'])==0:
            settings['data']['server_bin_root'] = settings['data']['local_bin_root'] 
        return settings