# Spatial Decomposition and Transformation Network (SDTNet)

**Tensorflow implementation of SDTNet**

Code (will be out soon!) for reproducing results in the paper:

> Valvano, G., Chartsias, A., Leo, A., & Tsaftaris, S. A.,  
> *Temporal Consistency Objectives Regularize the Learning of Disentangled Representations*, MICCAI Workshop on Domain Adaptation and Representation Transfer, 2019.
 
The overall block diagram of the SDTNet is the following:

<img src="https://gitlab.com/gabriele_valvano/sdtnet---spatial-decomposition-and-transformation-network/raw/master/results/images/SDTNet_block_diagram.png" alt="SDTNet_block_diagram" width="600"/>

**Data:**

Automatic Cardiac Diagnostic Challenge 2017 database. In total there are images of 100 patients, for which manual segmentations of the heart cavity, myocardium and right ventricle are provided.

Database at: [*acdc_challenge*](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html).\
An atlas of the heart in each projection can be found at this [*link*](http://tuttops.altervista.org/ecocardiografia_base.html).

# How to use it

**Overview:**
The code is organised with the following structure:

|    File/Directory            |Content                               
|---------------|----------------------------------------|
|architectures	| This directory contains the architectures used to build each module of the SDTNet (Encoders, Decoder, Segmentor, Discriminator, Transformer)|
|data			| folder with the data						|
|data_interfaces| dataset interfaces (dataset iterators)	|
|idas			| package with tensorflow utilities for deep learning (smaller version of [idas](https://github.com/gvalvano/idas)) |
results		| folder with the results (tensorboard logs, checkpoints, images, etc.)| 
|*config_file.py*| configuration file and network hyperparameters 
|*model.py*| file defining the general model and the training pipeline (architecture, dataset iterators, loss, train ops, etc.)
|*model_test.py*| simplified version of *model.py* , for test (lighter and faster)
|*split_data.py*| file for splitting the data in train/validation/test sets
|*prepare_dataset.py*| file for data pre-processing
|*train.py*| file for training



**Train:**
1. Put yourself in the project directory
2. Download the ACDC data set and put it under: *./data/acdc_data/*
3. Split the data in train, validation and test set folders. You can either do this manually or you can run: ```python split_data.py ```
4. Run ```python prepare_dataset.py ``` to pre-process the data. The code will pre-process the data offline and you will be able to train the neural network without this additional CPU overload at training time (there are expensive operations such as interpolations). The pre-processing consists in the following operations:
    - data are rescaled to the same resolution
    - the slice index is placed on the first axis
    - slices are resized to the desired dimension (i.e. 128x128)
    - the segmentation masks will be one-hot encoded
5. Run ```python train.py ``` to train the model.

Furthermore, you can monitor the training results using TensorBoard if you run the following command in your bash:
```bash
tensorboard --logdir=results/graphs
```
**Test:**

The file *model.py* contains the definition of the SDTNet architecture and of the training pipeline . Compiling all this stuff may be quite slow for a quick test: for this reason, we share a lighter version of *model.py*, namely *model_test.py* that avoids instantiating variables that are not used during test. You can run a test using this file as:

```python
from model_test import Model
...
RUN_ID = 'SDTNet'
ckp_dir = project_root + 'results/checkpoints/' + RUN_ID

# Given: x = a batch of slices to test
model = Model()
model.get_data()
model.define_model()
model.test(input_data=x, checkpoint_dir=ckp_dir)
model.test_future_frames(input_data=x, checkpoint_dir=ckp_dir)
```
Remember that, due to architectural constraints of the SDNet [*Chartsias et al. (2019)*,  [*k_code*](https://github.com/agis85/anatomy_modality_decomposition),  [*tf_code*](https://github.com/gvalvano/sdnet) ], the batch size that you used during training remains fixed at test time. 

# Results:

Anatomical factors extracted by the SDTNet from the image on the left:

<img src="https://gitlab.com/gabriele_valvano/sdtnet---spatial-decomposition-and-transformation-network/raw/master/results/images/decomposition.png" alt="SDTNet_block_diagram" width="600"/>

<br/><br/>
Predicted segmentations obtained from the UNet, SDNet, SDTNet after being trained with different percentages of the labelled data:

<img src="https://gitlab.com/gabriele_valvano/sdtnet---spatial-decomposition-and-transformation-network/raw/master/results/images/segmentations.png" alt="SDTNet_block_diagram" width="600"/>

---------------------

If you have any questions feel free to contact me at the following email address:

  *gabriele.valvano@imtlucca.it* 
  
Enjoy the code! :)

**Gabriele**