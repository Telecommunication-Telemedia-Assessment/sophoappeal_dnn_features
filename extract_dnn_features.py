#!/usr/bin/env python3

import argparse
import sys
import glob
import json
import bz2
import gzip

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
from tqdm import tqdm


def file_open(filename, mode="r"):
    """ Open a file (depending on the mode), and if you add bz2 or gz to filename a compressed file will be opened,
    file_open can replace a typical with open(filename) statement
    """
    if "bz2" in filename:
        return bz2.open(filename, mode + "t")
    if "gz" in filename:
        return gzip.open(filename, mode + "t")
    return open(filename, mode)


def main(_):
    # argument parsing
    parser = argparse.ArgumentParser(description='extract dnn image features',
                                     epilog="stg7 2023",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("image_dir", type=str, nargs="+", help="image directory to handle")
    parser.add_argument("--result_file", type=str, default="dnn_features.ldjson.bz2", help="file to store the features")

    a = vars(parser.parse_args())

    print(a["image_dir"])
    print(a["result_file"])


    model = VGG19(weights='imagenet', include_top=False)
    with file_open(a["result_file"], "w") as rfp:
        for idir in a["image_dir"]:
            for img_path in tqdm(list(glob.glob(idir +"/*"))):
                #img_path = 'elephant.jpg'
                img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                features = model.predict(x).flatten()
                res = {
                    "img_path": img_path,
                    "features": [float(x) for x in features]
                }
                rfp.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))