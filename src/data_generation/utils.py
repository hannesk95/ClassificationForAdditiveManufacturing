import re


def extract_file_name(path, suffix='stl'):
    """
    Extracts the name of the file with the given suffix
    :param path:
    :param suffix:
    :return: Name of the file
    """
    file_name = path.split('/')[-1]
    file_name = re.sub('.' + suffix + '$', '', file_name)
    return file_name
