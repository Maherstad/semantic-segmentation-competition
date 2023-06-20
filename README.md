# semantic-segmentation-and-domain-adaptation
This repository contains my implementation of the competition: Semantic segmentation and domain adaptation challenge proposed by the French National Institute of Geographical and Forest Information (IGN). <br /><br />


![flair-1_patches](https://github.com/Maherstad/semantic-segmentation-competition/assets/30124102/34a2151c-24a4-45ad-a396-51d8c2029c87)
<br />

## Repository Structure

The repository is organized as follows:

- `demo/`: Contains the dataset files, including aerial imagery and land cover annotations, provided by the competition organizer.
- `notebooks/`: Includes the implementation of the various architectures with pre-trained encoders.
- `data_display.py/`: Contains utility functions and scripts for data evaluation, and visualization.
- `train.py`: The main script for training the semantic segmentation model.
- `torchlightning_module.py`: A script that implements the code using the lightning library.
- `dataset_module.py`: A script that creates a class of the dataset to use it later during the training/testing phases.
- `preprocess_data.py`: A script that preprocesses the data to prepare it for the train/dev/test split.
- `on-start.py`: Script for preparing the dataset and structure of the project when launching a virtual machine from scratch.
- `requirements.txt`: Lists the required Python dependencies.

## Getting Started

To participate in the challenge, follow these steps:

1. Clone this repository to your local machine or a virtual machine using `git clone <repository-url>`.
2. Install the required Python dependencies by running `pip install -r requirements.txt`.
3. Preprocess the dataset using `preprocess.py` to prepare it for training.
4. Run `train.py` to train the semantic segmentation model.
5. Evaluate the trained model on the test set using `evaluate.py`. to-be-added
6. Feel free to experiment with different model architectures, data augmentation techniques, or domain adaptation methods to improve the results.
