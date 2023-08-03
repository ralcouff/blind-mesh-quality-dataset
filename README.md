# Blind Mesh Quality Dataset
This repository contains the dataset I used for the paper : [Blind Quality of a 3D reconstructed Mesh](https://hal.science/hal-03821315)

```
Rémy Alcouffe, Simone Gasparini, Geraldine Morin, Sylvie Chambon.
Blind Quality of a 3D Reconstructed Mesh.
29th IEEE International Conference on Image Processing (ICIP 2022),
IEEE, Oct 2022, Bordeaux, France. pp.3406 - 3410,
⟨10.1109/ICIP46576.2022.9897783⟩. ⟨hal-03821315⟩
```

## Overall Architecture
- `/3D_models` : contains all the models used for my study.
- `generate_3D_models.py` is a script to generate deformations.
- `obj_parser.py` is a custom-made parser for obj models. It was first developed by @tforgione and then adapted to this study.
- `requirements.txt` contains all the needed libraries to install to run the scripts.
- `config.py` contains all the parameters that the user might want to change to generate its own dataset.
- `utils.py` contains some functions usefull for the creation of deformed models.

## Motivation
This repository aims at offering the different 3D models used for my studies conducted during my Ph.D.
Alongside with the scripts that I used to generate the deformed 3D models.

## Installation
### Pre-requisites
Make sure you have a recent version of Python (min. 3.8)
```bash
python3 --version
```
### Creating a new python environment and installing the required packages
First clone the repository
```bash
cd thesis-dataset
python3 -m venv python_env
source python_env/bin/activate
pip install -r requirements.txt
```

### Update the config file
In the config file you will find several parameters corresponding to the several deformations available in this script.
You may change those parameters if you want.
They are all explained in the comments.

### Running the script to generate the deformed models
```bash
python generate_3D_models.py
```

## Acknowledgements
Thanks to @tforgione for the help and design of the obj_parser_module.

```
PyMeshLab
All rights reserved.

VCGLib  http://www.vcglib.net                                 o o
Visual and Computer Graphics Library                        o     o
                                                           _   O  _
Paolo Cignoni                                                \/)\/
Visual Computing Lab  http://vcg.isti.cnr.it                /\/|
ISTI - Italian National Research Council                       |
Copyright(C) 2020 
```
