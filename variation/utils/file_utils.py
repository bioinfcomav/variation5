import os
from tempfile import gettempdir


def remove_temp_file_in_dir(tmp_dir, fname_suffix):
    if tmp_dir is None:
        tmp_dir = gettempdir()
    for fname in os.listdir(tmp_dir):
        fpath = os.path.join(tmp_dir, fname)
        if fpath.endswith(fname_suffix):
            os.remove(fpath)
