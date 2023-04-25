I pushed the working tumor-stroma configuration I ended up with to the following branch in kaiko's fork of the repo: 
https://github.com/kaiko-ai/ahcore/tree/tumor-stroma-config
This branch also includes the `train.py` script and a `requirements.txt` file with the exact versions of the python modules used in this working setup.

## Steps reproduce the setup
1. Install Linux dependencies
```
sudo apt-get update 
sudo apt-get install build-essential 
sudo apt-get install gfortran
sudo apt-get install libvips-dev
sudo apt-get install openslide-tools	
```  
2. Create fresh venv with python 3.10 (3.9 is not compatible with latest dlup version)
3. pip install -r requirements.txt
4. Create the following environment variables (adjust paths according to your directory structure), and make sure the paths point to folders / files that exist:
```
export DATA_DIR=/mnt/data/all/ml-datasets-public/multi/tcga-open/tcga-2-open 
export ANNOTATIONS_DIR=/mnt/data/all 
export DATASET_SPLIT=/mnt/data/all 
export MANIFEST_PATH=/mnt/data/all 
export HYDRA_FULL_ERROR=1 
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 
export PROJECT_ROOT=/home/ahcore 
export PYTHONPATH=/home/ahcore 
```

The file `config/data_description/tissue_subtypes/training.yaml` makes use of the paths specified in the above env variables, revealing also what files are required to run trainings:
```
data_dir: ${oc.env:DATA_DIR}
annotations_dir: ${oc.env:ANNOTATIONS_DIR}/tissue_subtypes/v20230111 # specify in .env
manifest_path: ${oc.env:MANIFEST_PATH}/tissue_subtypes/v20230111/manifest.json
dataset_split_path: ${oc.env:DATASET_SPLIT}/tissue_subtypes/v20230111/train_val_test_split_mc.json
center_info_path: ${oc.env:DATASET_SPLIT}/tissue_background/v20230125/tissue_source_site_codes.csv 
```

5. Run the following command from the repo's root directory
```bash
python tools/train.py data_description=tissue_subtypes/training datamodule=dataset datamodule.num_workers=16 datamodule.batch_size=3 pre_transform=segmentation augmentations=segmentation metrics=segmentation losses=segmentation_ce lit_module=attention_unet
```

## Tested with the following system config:
- Ubuntu 20.04
- python 3.10.11 (python 3.9 ended up being incompatible with dlup==0.3.24)
- A100 GPU (first tried on T4 with 16GB GPU memory, but process got killed during validation run due to insufficient memory)
- cuda 11.6

## Troubleshooting
Error 1
```
    from openslide import lowlevel
  File "/home/nkaenzig/miniconda/envs/ahcore3/lib/python3.9/site-packages/openslide/lowlevel.py", line 84, in <module>
    _lib = cdll.LoadLibrary('libopenslide.so.0')
  File "/home/nkaenzig/miniconda/envs/ahcore3/lib/python3.9/ctypes/__init__.py", line 460, in LoadLibrary
    return self._dlltype(name)
  File "/home/nkaenzig/miniconda/envs/ahcore3/lib/python3.9/ctypes/__init__.py", line 382, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: /lib/x86_64-linux-gnu/libgobject-2.0.so.0: undefined symbol: ffi_type_uint32, version LIBFFI_BASE_7.0
```
Solution:
`export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7`


Error 2
```
 _SystemError: ffi_prep_closure(): bad user_data (it seems that the version of the libffi library seen at runtime is different from the 'ffi.h' file seen at compile-time)_:
```
Solution:
`pip install --force-reinstall --no-binary :all: cffi`


Error 3
If you run into CUDA issues, reinstall pytorch with the following command: `pip install torch=1.12 --index-url https://download.pytoch.org/whl/cu116`