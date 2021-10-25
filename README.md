# Model Monitoring Pipeline

## Overview

The goal of the project is to create a pipeline that can handle a binary classifcation
problem and monitor the adequateness of the trained model for subsequent batches of 
data. To do so, the pipeline controls for two different types of drift:
   
    - Covariate drift: changes in the distribution of covariates used to make the predictions
    - Concept drift: changes in the output distribution 

The pipeline also checks the performance of the model over the different datasets. 

By default, the pipeline saves all results to `Outputs`, but there is the possibility to 
launch a visualization app as well. The app is interactive and allows the user to input new
batches and have an overview of the drift metrics.

## Setup

To create the required folder structure, run from the project's root directory:

```{bash}
python setup.py
```

The requirements are in `requirements.txt`. To install them using pip, run:

```{bash}
pip install -r requirements.txt
```

## Running the pipeline

The first step is to make sure a model is trained on the data. Please make sure that
all the configurations are set in the `params/conf/conf.json` file. Then, the model
can be trained by running at the `src/` level:

```{bash}
python train_model.py
```

**N.B.**: the training data must be placed in the `Inputs/` folder.

The pipeline can be run independently from the command line by running the
following commands from the root directory: 

```{bash}
main.py [-h] [-m {process,evaluate}] --batch-id BATCH_ID
```

optional arguments:

-m {process,evaluate}, --mode {process,evaluate}
                    Defines the type of monitoring to perform
                    
--batch-id BATCH_ID   Id of batch to be parsed

**N.B.**: the batch data must be placed in the `Inputs/Batches` folder. And follow 
the naming template: `batch{id}.csv`

Another option is to run the pipeline directly from the app, by running:

```{bash}
streamlit run launch_app.py
```
