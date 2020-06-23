# e4040-2019Fall-Project

A comprehensive review and TensorFlow implementation of [Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](http://arxiv.org/pdf/1312.6082.pdf) 


## Architectural Block Diagram

![Graph](https://github.com/cu-zk-courses-org/e4040-2019fall-project-zxxz-rx2166-wz2466-yz3075/blob/master/Image/diagram.png?raw=true)


## Flow Chart
![Accuracy](https://github.com/cu-zk-courses-org/e4040-2019fall-project-zxxz-rx2166-wz2466-yz3075/blob/master/Image/flow_chart.png?raw=true)


## Dataset
![Dataset](https://github.com/cu-zk-courses-org/e4040-2019fall-project-zxxz-rx2166-wz2466-yz3075/blob/master/Image/dataset.png?raw=true)



> digit "10" means no digits


## Results

### Accuracy
![Accuracy](https://github.com/cu-zk-courses-org/e4040-2019fall-project-zxxz-rx2166-wz2466-yz3075/blob/master/Image/accuracy.png?raw=true)

> Accuracy 86.02% on test dataset after about 15 hours

### Loss
![Loss](https://github.com/cu-zk-courses-org/e4040-2019fall-project-zxxz-rx2166-wz2466-yz3075/blob/master/Image/loss.png?raw=true)


## Requirements
    ```
    Python 3.5
    Tensorflow 1.13.0
    h5py
    ```


## Setup

1. Download [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/) format 1

2. Extract and split raw images to three data folders, now your folder structure should be like below:
    ```
    Folder Structure
            - test
                - image
                    - 1.png 
                    - 2.png
                    - 3.png
                    - ...
                - mat
                    - digitStruct.mat
            - train
                - image
                    - 1.png 
                    - 2.png
                    - 3.png
                    - ...
                - mat
                    - digitStruct.mat
            - validation
                - image
                    - 1.png 
                    - 2.png
                    - 3.png
                    - ...
                - mat
                    - digitStruct.mat
    ```

3. We upload the data into liondrive and the link is shared. Please download the data and structure it as above for proper use.

4. Complete folder structure overview:

    ```
    Folder Structure
            - test
            - train
            - validation
            
            - graphs
            - Image
            - logs/train
            - trainhist
            
            - main.ipynb
            
            - evaluator.py
            - model.py
            - preproc.py
            - training.py
            
    ```

    


## Usage

0. Run the jupyter notebook which serves as the main UI file. 
    ```
    Open `main.ipynb` in Jupyter
    ```

   More detailed steps are listed as follow.
   
1. Convert raw data into TFRecord format.

    ```
    python       preproc.py
    class        DataManager
    function     write_tfrecord
    ```

2. Take a look at the images with labels.

    ```
    python       preproc.py
    class        DataManager
    function     data_generator
    ```

3. Train.

    ```
    python       training.py
    function     _train
    ```

4. Evaluate the model on the test dataset.

    ```
    python       evaluator.py
    function     evaluate
    ```
    
5. Save the models in the following structure.

    ```
    - logs
        - train
            - checkpoint 
            - events.out.tfevents.......
            - latest.ckpt.data-00000-of-00001
            - latest.ckpt.index
            - latest.ckpt.meta
            - model.ckpt-xxxxx.data-00000-of-00001
            - model.ckpt-xxxxx.index
            - model.ckpt-xxxxx.meta
    ```

