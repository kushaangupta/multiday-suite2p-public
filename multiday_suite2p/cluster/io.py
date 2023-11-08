from pathlib import Path

def test_extract_result_present(path):
    path = Path(path)
    # test if directory is present.
    if not path.is_dir():
        return(False)
    else:
        spks_file = path/'Fneu.npy' #last file to be written.
        if not spks_file.is_file():
            return False
    return True