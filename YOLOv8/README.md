Readme
Data folder:
data/: contains the turtle.yaml file. Includes the path of training, validation, and testing data.
datasets/turtle_merge/images: Copies the original image
datasets/turtle_merge/labels: stores converted images
datasets/turtle/ : Stores partitioned datasets, training sets, verification sets, and test sets.
images/ : Stores raw data
weights/: Saves the trained model weights file for loading the pre-trained model or saving the training results.
runs/: Stores results generated during training or inference (e.g. logs, model outputs, visualizations, etc.).

Metadata file:
metadata.csv and metadata_splits.csv contain detailed descriptions, labels, split information, etc., of the data set.


Code file:
train.py and train-detect.py: Scripts for training models.
detect.py: script used to perform inference.
eval.py: A script used to evaluate model performance.


all.ipynb is a visual flow of the operation of the entire project, and you only need to run this file to train, verify and test the model.