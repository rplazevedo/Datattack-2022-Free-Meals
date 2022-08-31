# Datattack

## Dataset
You can find the dataset [here](https://drive.google.com/drive/folders/1ZyWpCiTUPMbSkMmzSgxD63mhaD7n_6kI?usp=sharing). You should copy its content to the dataset folder of this repository.

There are 295 serial number images inside the dataset folder. The file `labels_train.json` contains the annotations in COCO format.

By the end of the hackaton, 74 test images will be provided to the participants, who should run their model and export their predictions 
using the script `export_predictions.py`

## Scripts

`data_viz.py`: reference script for visualizing images and annotation labels

`export_predictions.py`: reference script for exporting predictions in the expected submission format
