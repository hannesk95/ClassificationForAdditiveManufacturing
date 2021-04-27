# Thingi10K Dataset

Fechting data from the [Thingi10K](https://ten-thousand-models.appspot.com/results.html?q=genus%3D0) dataset is actually pretty easy. The command in combination with the provided script [fetch_data.py](./fetch_data.py) can be used as follows:

```cmd
python3 fetch_data.py 
```

In addition, parameters can be used in order to specify the output directory where to store the data:

```cmd
python3 fetch_data.py -o <output_dir>
```