# ABC Dataset

Generally, the [ABC dataset](https://deep-geometry.github.io/abc-dataset/) consists of 100 data chunks. Each of them is approx. of 10 GB size. Every data chunk comes in a zipped file format. After unzipping the data chunk, all .stl files which are included in this chunk occupy ~40 GB of data storage. As a consequence, the entire dataset unzipped takes about 4 TB of storing capacity. 

In the follwing, it will be explained how to select a data chunk out of the 100 provided, and how to start the download of it:

1. All URLs to the 100 data chunks are stored in the file [`stl2_v00_all.txt`](./utils/stl2_v00_all.txt), which can be found in the [`utils`](./utils) directory. In order to download one specific data chunk, copy the URL of it into the file [`stl2_v00.txt`](./utils/stl2_v00.txt), which is also in the [`utils`](./utils) directory. Make sure, that the [`stl2_v00.txt`](./utils/stl2_v00.txt) only contains one URL at a time, for which the download should be started. 

2. Once the data chunk is selected, one can easily run the [`fetch_ABC_dataset.py`](./fetch_ABC_dataset.py) by executing the following command on the CLI:

```cmd
   $ python3 fetch_ABC_dataset.py
```

3. By executing the python script as described above, everythin is done automatically. Once the entire process has finished, you'll see a output on the CLI which states that everything was done successfully. 

4. All plain .stl files will be stored in a folder denoted `stl_files`. This folder gets created automatically during the execution of the python file metioned above. 

5. If you would like to store the files at another location, just copy the `fetch_ABC_dataset.py` file and the `utils` folder to the location on your workstation or hard drive and start the download there. 

**General comment:** Do not store data in any repository sturcture neither upload any data to the repository.



