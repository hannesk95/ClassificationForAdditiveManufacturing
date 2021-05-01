import os
import subprocess
from tqdm import tqdm
from shutil import copyfile, rmtree
from pyunpack import Archive


def start_download():
    """
    This function creates a directory, if it does not exist already, in order to store the
    .stl files and starts the download.
    """

    directory = "stl_files"
    if not os.path.exists(directory):
        os.makedirs(directory)

    os.chdir('./utils/')
    subprocess.call(['sh', './download.sh'])


def unzip_directory():
    """
    This function takes the 7z archive and extracts it.
    """

    os.chdir("../")
    zip_name = os.listdir("./stl_files")
    zip_path = "./stl_files" + "/" + zip_name[0]

    Archive(zip_path).extractall("./stl_files")
    rmtree(zip_path, ignore_errors=True)


def flatten_directory():
    """
    This function copies every .stl file which is stored in a sub-folder into its
    parent directory.
    """

    cwd = os.getcwd()
    dataset_path = cwd + "\\" + "stl_files"

    os.chdir(dataset_path)
    folders = os.listdir()

    for i in tqdm(range(len(folders))):

        if folders[i].endswith("7z"):
            continue

        os.chdir(dataset_path + "\\" + folders[i])
        filename = os.listdir()
        src = dataset_path + "\\" + folders[i] + "\\" + filename[0]
        dst = dataset_path + "\\" + filename[0]
        copyfile(src, dst)
        os.chdir(dataset_path)
        rmtree(dataset_path + "\\" + folders[i], ignore_errors=True)


if __name__ == "__main__":
    start_download()
    unzip_directory()
    flatten_directory()

    print("###############################################################################")
    print("[SUCCESS]: Download and processing of the specified ABC data chunk is complete!")
    print("###############################################################################")
