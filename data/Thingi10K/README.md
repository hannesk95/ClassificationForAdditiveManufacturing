# Thingi10K Dataset

Fechting data from the [Thingi10K](https://ten-thousand-models.appspot.com/results.html?q=genus%3D0) dataset is actually pretty easy. The command in combination with the provided script [fetch_data.py](./fetch_data.py) can be used as follows:

```cmd
$ python3 fetch_Thingi10K.py 
```

In addition, parameters can be used in order to specify the output directory where to store the data:

```cmd
$ python3 fetch_Thingi10K.py -o <output_dir>
```

**General comment:** Do not store data in any repository sturcture neither upload any data to the repository.

---

## Python Packages
The following provides a listing which python libraries are needed in order to run the python script:

```cmd
tqdm
$ pip3 install tqdm

argparse
$ pip3 install argparse
```