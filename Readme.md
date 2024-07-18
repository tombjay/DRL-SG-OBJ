# Evaluation of Different Image Representations for Reinforcement Learning Agents.

This repo contains the code for the work on evaluating different image representations for training reinforcement agents. This repo contains mostly single file implementations. To get to know our project more clearly,  kindly have a look at our ['report'](https://git.hcics.simtech.uni-stuttgart.de/theses/msc2023_jayakumar/src/branch/main/Final_Thesis_Report.pdf)

## Getting started
- Clone this repository and install the necessary packages listed in ['requirements.txt'](https://git.hcics.simtech.uni-stuttgart.de/theses/msc2023_jayakumar/src/branch/main/requirements.txt) file by running the command,

```
pip install -r /path/to/requirements.txt
```
## Object-centric model.
To train the object-centric model along with model evaluation and wandb tracking, navigate to the object-centric folder and run the following line along with the command line arguments,
```
python object-centric-model.py --track --evaluate
```

## Scene graph model
The scene graph based models are grouped based on the Atari environments and are mostly single file implementations. Information about the implementations are provided in the starting of the each file. To train the scene graph models, navigate to the file path and run the following command,

```
python file-name.py --track --evaluate
```
Similarly, different experiments are done using the scene graph generation models for certain environments. However, to try out these experiments for other untested environments, modify the scene graph generation methods accordingly as described in respective game environment folders. For each game environment, the methods: create_graph_input() and get_object_coordinates() differs along with certain other object definition variables. Carefully modify them as done in files in respective game environments.

## Evaluation
To evaluate the saved models, run the respective single file implementaions of evaluation files under the model-evaluation folder using the following command,

```
python model_evaluation_(SG/obj/baseline).py
```

## Faster R-CNN
To train the Faster R-CNN model for Atari-head dataset, run the following command,

```
python fasterRCNN_train.py
```