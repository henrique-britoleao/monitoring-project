# Model Monitoring Pipeline

## Overview

The goal of the project is to create a pipeline that can handle a binary classifcation
problem and monitor the adequateness of the trained model for subsequent batches of 
data. To do so, the pipeline controls for two different types of drift:
   
    - Covariate drift: cahnges in the distribution of covariates used to make the predictions
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

Then the pipeline can be run by:

```{bash}
python main.py
```

The app can be launched by running:

```{bash}
streamlit run launch_app.py
```