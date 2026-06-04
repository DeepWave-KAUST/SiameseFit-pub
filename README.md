![LOGO](https://github.com/DeepWave-KAUST/SiameseFit/blob/main/asset/Figure1_TWINWAVE.png)

Official reproducible material for **SiameseFit: Latent Seismic Data Comparison for Enhanced Full Waveform Inversion and Cycle Skipping Mitigation - Omar M. Saad, and Tariq Alkhalifah**


# Project structure
This repository is organized as follows:

* :open_file_folder: **asset**: folder containing logo;
* :open_file_folder: **data**: folder containing Marmousi2 and overthrust models data;
* :open_file_folder: **Model**: containing Siamese network;
* :open_file_folder: **deepwave-old**: containing the old version of the DeepWave package.
  

## Notebooks
The following notebooks are provided:

- :orange_book: ``SiameseFit_Flat_Marmousi2.ipynb``: the main notebook performing the SiameseFit for Marmousi2 model (Flat initial model);
- :orange_book: ``SiameseFit_Constant_Marmousi2.ipynb``: the main notebook performing the SiameseFit for Marmousi2 model (Constant initial model);
- :orange_book: ``SiameseFit_Flat_Overthrust.ipynb``: the main notebook performing the SiameseFit for the Overthrust model (Flat initial model);
- :orange_book: ``SiameseFit_Constant_Overthrust.ipynb``: the main notebook performing the SiameseFit for the Overthrust model (Constant initial model);
- :orange_book: ``SiameseFit_MultiSource_Marmousi2.ipynb``: the main notebook performing the SiameseFit for Marmousi2 model (Multi-Source FWI);

## Getting started :space_invader: :robot:
- To ensure the reproducibility of the results, we suggest using the `FWIGAN.yml` file when creating an environment.
- Please install deepwave 0.0.8 version, which is used in this project.


Run:
```
./install_env.sh
```
It will take some time, but if you see the word `Done!` on your terminal you are ready to go. 

Remember to always activate the environment by typing:
```
conda activate FWIGAN
```
To install the old version of the DeepWave package, navigate to the "deepwave-old" folder, and run:
```
cd ./deepwave-old/
python setup.py install
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX A6000 GPU. Different environment configurations may be required for different combinations of workstation and GPU.
