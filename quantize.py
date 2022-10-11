import os
import cv2 as cv
import numpy as np
import argparse
import cv2

from addict import Dict
from openvino.tools.pot import DataLoader
from openvino.tools.pot import IEEngine
from openvino.tools.pot import load_model, save_model
from openvino.tools.pot import create_pipeline
from openvino.tools.pot import Metric

from torchvision.transforms import Compose
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

parser = argparse.ArgumentParser(description="Quantizes OpenVino model to int8.",
                                    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--xml", default="midas.xml", help="XML file for OpenVINO to quantize")
parser.add_argument("--model_name", default="midas_opt", help="OpenVINO model name")
# parser.add_argument("--annotation", default="val.txt", help="Manifest file (txt file with filenames of images and labels)")
parser.add_argument("--data", default="../data/ReDWeb_V1/Imgs", help="Data directory root")
parser.add_argument("--int8_dir", default="./model/optimized", help="INT8 directory for calibrated OpenVINO model")

argv = parser.parse_args()

net_w = net_h = 384
transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
        ]
    )

class ImageLoader(DataLoader):
    """ Loads images from a folder """
    def __init__(self, dataset_path):
        # Use OpenCV to gather image files
        # Collect names of image files
        self._files = []
        all_files_in_dir = os.listdir(dataset_path)
        for name in all_files_in_dir:
            file = os.path.join(dataset_path, name)
            if cv.haveImageReader(file):
                self._files.append(file)

        self._shape = (384,576)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index):
        """ Returns image data by index in the NCHW layout
        Note: model-specific preprocessing is omitted, consider adding it here
        """
        print("!!__getitem__!!")
        if index >= len(self):
            raise IndexError("Index out of dataset size")

        image = cv.imread(self._files[index]) # read image with OpenCV
        # image = transform({"image": image})["image"]
        image = cv2.resize(image, self._shape) # resize to a target input size
        image = np.expand_dims(image, 0)  # add batch dimension
        image = image.transpose(0, 3, 1, 2)  # convert to NCHW layout

        return image, None   # annotation is set to None


# Dictionary with the FP32 model info
model_config = Dict({
    'model_name': argv.model_name,
    "model": argv.xml,
    "weights": argv.xml.split('.xml')[0] + '.bin'
})

# Dictionary with the engine parameters
engine_config = Dict({
    'device': 'CPU',
    'stat_requests_number': 4,
    'eval_requests_number': 4
})

# Quantization algorithm settings
algorithms = [
    {
        "name": "DefaultQuantization",
        "params": {
            "target_device": "ANY",
            "stat_subset_size": 300,
            "stat_batch_size": 1
        },
    }
]

# Load the model.
model = load_model(model_config)

# Initialize the data loader and metric.
data_loader = ImageLoader(argv.data)

# Initialize the engine for metric calculation and statistics collection.
engine = IEEngine(engine_config, data_loader)

# Initialize the engine for metric calculation and statistics collection.
pipeline = create_pipeline(algorithms, engine)

# Execute the pipeline.
compressed_model = pipeline.run(model)

# Save the compressed model.
save_model(compressed_model, argv.int8_dir)
