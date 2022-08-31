import os
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import fiftyone.utils.coco as fouc


# Load COCO formatted train set
# The dataset_dir, should contain a 'data' folder inside with all the images of the dataset
train = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    dataset_dir ="dataset",
    labels_path="labels_train.json",
    include_id=True,
    extra_attrs = True,
    name='train'
)

if __name__ == "__main__":
    # Ensures that the App processes are safely launched on Windows
    session = fo.launch_app()
    session.wait()
