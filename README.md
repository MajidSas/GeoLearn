Code for paper "Towards Learned Geospatial Data Analysis & Exploration".


*Datasets:* https://drive.google.com/drive/folders/19L_QbU1noJ-i2LWfDvdrUQWP5-UgkBVM?usp=share_link

## Setup

Install requirements with `python -m pip install -r requirements.txt`


## Data Synopsis Experiments

Change the directory to `./01_data_synth`.

In the file `dataset.py` change the path `/rhome/msaee007/bigdata/pointnet_data/synthetic_data` to the path where the dataset `synthetic_data` is stored, available at the provided link above.

In the file `pointnet_train.py` change the output folder where you want the model to be stored. Change the following:
```python
output_folder = '/rhome/msaee007/bigdata/pointnet_data/synthetic_data/main_exp_outputs'
```

In the same file, change `/rhome/msaee007/bigdata/pointnet_data/p1_real_data` to where the dataset `p1_real_data` is stored.

Similarly, change the same paths in the file `resnet_train.py`

To train the pointnet and resnet models, you can use the following commands:

```bash
python resnet_train.exp synth
python resnet_train.exp weather
python pointnet_train.py
```

To produce the for these models, first modify the file `p1_pred_summary.py`, and modify these paths with you stored the datasets: `/rhome/msaee007/bigdata/pointnet_data/p1_real_data`, `/rhome/msaee007/bigdata/pointnet_data/synthetic_data_test` and `/rhome/msaee007/bigdata/pointnet_data/synthetic_data`.

Also, change the paths for where the trained models are stored: replace `/main_exp_outputs/all_values_parametrized_weather_0.212609.ckpt` with the path to the pointnet model trained with weather data, replace `/main_exp_outputs/all_values_parametrized_synth_0.014546.ckpt` with the path to the pointnet model trained with the synthetic data, replace `/main_exp_outputs/resnet_weather_0.259091.ckpt` with the path of the resnet model trained with the weather data, and finally the resent model with the synthetic data `/main_exp_outputs/resnet_synth_0.055634.ckpt`.

Then, you can run these commands:

```bash
python p1_pred_summary.py synth pointnet
python p1_pred_summary.py synth resnet
python p1_pred_summary.py weather pointnet
python p1_pred_summary.py weather resnet
```

In the `./poly2vec_transformer` directory, modify the file `p1_poly2vec_transformer.py` and change the `output_path` and the path to the `p1_real_data` to where ever it is stored. You can then train the model with the command:

```bash
python p1_poly2vec_transformer.py
```
Change the same paths in `p1_pred_summary.py` similar to the same file in the previous folder. Then, from inside this folder `poly2vec_transformer` you can run this command:
```bash
python p1_pred_summary.py
```

Finally, in the folder `./01_data_synth`, modify the paths in the file `p1_tables.py` to whereever you stored the results produced by the `p1_pred_summary.py` files and the datasets. The last part of this file `p1_tables.py` produces the values for the validation loss per epoch. Change those paths to the standard output of the training scripts, if you saved them, or just comment these lines.

This should produce all the results and tables shown in the paper.


## Clustering Experiments


## Selectivity Experiments


## Walkability Experiments



