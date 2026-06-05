This is the official implementation for the paper "Diffusion Reconstruction-based Active Defense Method against Voice Conversion Forgery".

The repository is structured as follows:

checkpoint/: Contains the pre-trained diffusion vocoder weight checkpoints.

data/: Contains scripts and steps for data preprocessing.

metrics/: Contains scripts for computing objective evaluation metrics.

Surrogate Models Download Links:
AdaIN-VC: https://github.com/cyhuang-tw/AdaIN-VC

VQVC+: https://github.com/ericwudayi/SkipVQVC

AGAIN-VC: https://github.com/KimythAnly/AGAIN-VC

TriAAN-VC: https://github.com/winddori2002/TriAAN-VC

Core Scripts:
__main__.py: The main entry script of the project, supporting one-click training invocation via the command line.

eva.py: The core evaluation script, used for batch calculating the imperceptibility (quality) and defense success rate of the protected speech.

FrequencyLoss.py: Implementation of the multi-scale frequency domain reconstruction loss function, used to constrain the perceptual quality of the reconstructed speech.

hyperparameter.py / params.py: Modules for defining global hyperparameters and parsing variables during runtime.

inference.py: The defense generation script. Takes clean speech as input, performs diffusion reconstruction via reverse denoising, and outputs adversarial speech equipped with defense capabilities.

learner.py: The model training engine, encapsulating the complete epoch process including forward noise addition, reverse reconstruction, gradient updating, and logging.

model.py: Defines the core backbone architecture of the diffusion vocoder and the reconstruction defense network.

Data Preparation:
Run the prepare_data.py script to complete the following operations:

Downsample the VCTK dataset.

Extract the required data from the VCTK dataset.

Obtain the metadata information of the data.

Dataset Download Link:

VCTK: https://datashare.ed.ac.uk/handle/10283/2950

Environment Setup:Use the following command to create the conda environment:
conda env create -f re.yaml