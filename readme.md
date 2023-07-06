# Model Optimization

this git is used to research optimized parameters and models for classification task regarding the project of Axel Rocheteau during his internship at ASI.
The git is composed of three parts :
- the training folder stores training datas for the models
- the lib folder contains every functions used for the models' training and optimization
- the notebook folder contains jupyter notebooks (each jupyter notebook solve one task)

## Context
The goal of this project is to display french housings possible renovation according to areas and criteria. This project is based on Tremi survey. It took place in 2017 and collect information about french refurbishments between 2014 and 2016. After mastering the data from this questionary and storing relevant information into a cloud datalake, We need to extend the knowledge from this survey to every french building (i.e do they need a refurbishment). For this reason we will classify building from DPE dataset on the fact that they need a renovation or not.

## Tasks
this optimization process can be divided into three major tasks:
- Model 1 named pred_Tremi will fullfill missing data from Tremi survey
- Model 2 named pred_dpe will input necessary informations about housings from Tremi survey
- Model 3 named pred_renov will classify buildings from DPE datasets on weither they need a refurbishment or not.

## dependancies
this project uses the folowing librairies :
- Pyspark (v 3.4.0)
- scikit-learn (v 1.3.0)
- scipy (v 1.11.0)
- xgboost (v 1.7.6)
- numpy (v 1.24.2)
- matplotlib (v 3.7.1)
- pandas (v 1.5.3)

## lib folder
The lib folder is divided into five files:
- `usefull.py` stores functions usefull in any circumstances
- `prepare_data.py` contains every function necessary to prepare data to input in a sklearn model
- `train.py` contains every functions needed to train a model (score and train)
- `show.py` stocks functions to display valuable insights on model optimization
- `all.py` stores a function handle the training of a model from the beginning to the end (prepare data, train model and display insights)

## training folder

## Jupiter files

