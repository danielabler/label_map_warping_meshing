import os



def get_file_extension(path_to_file):
    """
    Returns file extension by splitting at '.' location, or None if no '.' in file name.
    :param path_to_file: path to file to be tested
    :return: string or None
    """
    # need to test_cases for existence of '.'
    # return None of component has no file extension
    file_name = os.path.split(path_to_file)[-1]
    file_name_split = file_name.split(".")
    if file_name_split[-1] == file_name:
        # there is no '.' in file_name
        return None
    else:
        return file_name_split[-1]


def ensure_dir_exists(path):
    """
    Checks whether path exists and creates main directory if not.
    :param path: path to be tested
    """
    if get_file_extension(path) == None:
        # assume that 'path' is directory, add trailing '/'
        path = path + '/'
    if os.path.exists(os.path.dirname(path)):
        return True
    else:
        try:
            os.makedirs(os.path.dirname(path))
        except:
            pass
        return False


def merge_dicts(a, b, path=None):
    "merges b into a, from https://stackoverflow.com/questions/7204805/dictionaries-of-dictionaries-merge"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a