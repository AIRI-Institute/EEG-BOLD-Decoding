# Decoding BOLD signal from EEG

This is an official repository for the project dedicated to decoding Blood-oxygen-level-dependent (BOLD) signal,
typically registered with functional magnetic resonance imaging (fMRI), from Electroencephalography (EEG) data.

## Repository structure

This repository follows the next structure:
```
├── bw_linker               # Source code
|   ├── brain_wave_pipeline # Code for the neural network pipeline
|   ├── data_preprocessing  # Code to load and preprocess EEG and BOLD signals
|   ├── pls_pipeline        # Code for the partial least squares pipeline
│   ├── utils               # Code with utility functions and constants
│   └── visualization       # Code for making plots
├── configs                 # Configuration files for neural network experiments
│   ├── subcort             # Subfolder with configs for Subcortical structured without removing the trend
│   ├── subcort-trend       # Subfolder with configs for Subcortical structured with the trend removed
│   └── ...       
├── NaturalViewingDataset   # Directory for the dataset files
├── pls_logs                # Directory created by default for partial least squares experiments
├── visualizations          # Directory created by default for images and LaTeX table with metrics
├── wandb_logs              # Directory created by default for neural network experiments
├── README.md               # README file
└── requirements.txt        # A file with requirements 
```

## Dataset

For this project we preprocessed the data from Natural Viewing Dataset (Telesford, Q.K., Gonzalez-Moreira, E., Xu, T. 
et al. An open-access dataset of naturalistic viewing using simultaneous EEG-fMRI. Sci Data 10, 554 (2023).
https://doi.org/10.1038/s41597-023-02458-8). We only preprocessed and utilize the subjects which have all the 
recordings with all the sessions and tasks. To run this code, firstly download our [preprocessed files](https://zenodo.org/records/11246524)
and put them into the ```NaturalViewingDataset``` directory in the root of the project. If you utilize this data in
your study please credit us and the original creators of the Natural Viewing Dataset.

## Environment setup

To set up the conda environment run the following code:

```
conda create -n EEG-BOLD-Decoding python=3.11.10
conda activate EEG-BOLD-Decoding
pip install -r requirements.txt
```

## BrainWaveLinker

For this project we developed a compact and interpretable neural network architecture called BrainWaveLinker. 
To train it on Natural Viewing Dataset run the following code:

```
conda activate EEG-BOLD-Decoding
export PYTHONPATH=./
python bw_linker/brain_wave_pipeline/train_model.py --config-root path-to-subfolder-with-configs
```

To see all the arguments accepted by the training function please refer to the
```bw_linker/brain_wave_pipeline/train_model.py``` module

After training, the checkpoints, files with metrics and output logs will be available at ```wandb_logs/```

## Partial Least Squares

Currently, the State-of-the-Art algorithm for decoding BOLD from EEG is presented in (Singer, N., Poker, G., 
Dunsky-Moran, N. et al. Development and validation of an fMRI-informed EEG model of reward-related ventral striatum 
activation. Neuroimage 276:120183 (2023). doi: 10.1016/j.neuroimage.2023.120183). It relies on a classical feature 
extraction algorithm, followed by the selection of up to 10 main EEG channels using Lassoed principal components, and 
finally the application of the Partial Least Squares Regression to get predictions. We call this Partial Least Squares 
(PLS) pipeline. To run training of our implementation of the PLS approach use the following code:

```
conda activate EEG-BOLD-Decoding
export PYTHONPATH=./
python bw_linker/pls_pipeline/pls_pipeline.py --rois "roi_1" "roi_2" ... "roi_n" --n-workers N 
```

To see all the arguments accepted by the training function please refer to the
```bw_linker/pls_pipeline/pls_pipeline.py``` module

After training, trained models (in .onnx format), files with metrics and output logs will be available at 
```pls_logs/experiment```

### Testing the PLS implementation

Note: this is our implementation of the original algorithm by Singer, et al. as the original paper did not have an 
open-source code. To test the correctness of our implementation we utilize the synthetic data with different
signal-to-noise ratios (SNR) and check how the accuracy of the main channels detection and the pipeline overall
degrades with lowering SNR. In our tests we get a predictable result of lower SNRs degrading performance of both steps 
of the pipeline, signaling about the correctness of the implementation. To perform these tests you can run the 
following code:

```
conda activate EEG-BOLD-Decoding
export PYTHONPATH=./
python bw_linker/pls_pipeline/verify_pls_pipeline.py
```

## Visualizations

Weights and biases saves run information under a run identifier rather than its name. Therefore, after training before 
you can start plotting results you have to create a .json file with a mapping between run names and run identifiers. 
You can do that by running the following code:

```
conda activate EEG-BOLD-Decoding
export PYTHONPATH=./
python bw_linker/visualization/get_wandb_mapping.py --entity weights-and-biases-entity-name
```

After, you can obtain final graphs and a LaTeX tables saved in the .txt files using the following
code. 
To plot graphs with correlations and get LaTeX table run the next code:

```
conda activate EEG-BOLD-Decoding
export PYTHONPATH=./
python bw_linker/visualization/plot_results.py
```

To plot the remaining graphs with brain patterns run the following code:

```
conda activate EEG-BOLD-Decoding
export PYTHONPATH=./
python bw_linker/visualization/plot_topographies.py --suffix experiment-suffix
```

Note: ```plot_topographies.py``` need to load configuration file used during experiment. This module will load configs
from configs directory using standardized config file names. 
Therefore, while running it make sure that your config in configs directory did not change relative to the one you used 
while training.

The visualizations will be saved to the ```visualizations/``` directory

Note: by default these functions will try to plot all the graphs for all the subjects and experiments. Therefore, it 
could take a long time. You can adjust what you want to plot. If you want to adjust some 
parameters look into the modules that are being called

## Citing BrainWaveLinker

If you use our data or code please cite the following paper:
!LINK!

If you use our data please also credit the creators of the original dataset:
Telesford, Q.K., Gonzalez-Moreira, E., Xu, T. et al. An open-access dataset of naturalistic viewing using simultaneous 
EEG-fMRI. Sci Data 10, 554 (2023). https://doi.org/10.1038/s41597-023-02458-8

If you use our code for the PLS pipeline please also credit the creators of the pipeline's idea:
Singer, N., Poker, G., Dunsky-Moran, N. et al. Development and validation of an fMRI-informed EEG model of 
reward-related ventral striatum activation. Neuroimage 276:120183 (2023). doi: 10.1016/j.neuroimage.2023.120183
