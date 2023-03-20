# Sophoappeal dnn feature extractor
DNN feature extractor, using VGG19, for sophoappeal.

This repository is part of [Sohpappeal](https://github.com/Telecommunication-Telemedia-Assessment/sophoappeal).
Please use the main repository as starting point.

This repository is part of the DFG project [Sophoappeal (437543412)](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-elektrotechnik-und-informationstechnik/profil/institute-und-fachgebiete/fachgebiet-audiovisuelle-technik/forschung/dfg-projekt-sophoappeal).


## Requirements


* python3, python3-pip, git, imagemagick, wget
* tensorflow, numpy, tqdm install with `pip3 install -r requiremnts.txt`

Run `./prepare.sh` to download the pre calculated features

## Structure and scripts

* `./extract_dnn_features.py`: extract dnn based features

## Usage
```
usage: extract_dnn_features.py [-h] [--result_file RESULT_FILE]
                               image_dir [image_dir ...]

extract dnn image features

positional arguments:
  image_dir             image directory to handle

optional arguments:
  -h, --help            show this help message and exit
  --result_file RESULT_FILE
                        file to store the features (default:
                        dnn_features.ldjson.bz2)

stg7 2023
```

Convert the generated `ldjson.bz2` output with `bzcat dnn_features.ldjson.bz2 | ./ldjson2json.py | bzip2 - > dnn_features.json.bz2` to `json.bz2` for later usage.



## Acknowledgments

If you use this software or data in your research, please include a link to the repository and reference the following paper.

```bibtex
@article{goering2023imageappeal,
  title={Image Appeal Revisited: Analysis, new Dataset and Prediction Models},
  author={Steve G\"oring and Alexander Raake},
  journal={IEEE Access},
  year={2023},
  publisher={IEEE},
  note={to appear}
}
```

## License
GNU General Public License v3. See [LICENSE.md](./LICENSE.md) file in this repository.