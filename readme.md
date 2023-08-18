# Model Optimization

This git is used to research optimized parameters and models for classification tasks regarding the project of Axel Rocheteau during his internship at ASI.
The git is composed of three parts:
- The training folder stores training data for the models.
- The lib folder contains every function used for the models' training and optimization.
- The notebook folder contains Jupyter notebooks (each Jupyter notebook solves one task).

## Context
The goal of this project is to display French housing's possible renovation according to areas and criteria. This project is based on the Tremi survey. It took place in 2017 and collected information about French refurbishments between 2014 and 2016. After mastering the data from this questionnaire and storing relevant information in a cloud data lake, We need to extend the knowledge from this survey to every French building (i.e., do they need refurbishment?). For this reason, we will classify buildings from the DPE dataset based on whether they need renovation or not.

## Tasks
This optimization process can be divided into three major tasks:
- Model 1, named pred_Tremi, will fill in the missing data from the Tremi survey.
> Variable surface, heating_emission and heating production are missing for housings that did not achieve refurbishment between 2014 and 2016.

- Model 2, named pred_dpe, will input necessary information about housing from the Tremi survey.
> predict the variables dpe and ges for every housing unit in the Tremi dataset. Those two variables indicate how much energy a building is consuming (electricity and greenhouse gases).

- Model 3, named pred_renov, will classify buildings from DPE datasets based on whether they need refurbishment or not.
> The variable has_to_renov is missing in the DPE dataset.

## dependancies
This project uses the following libraries (listed in pyproject.toml as well):
- Python (v 3.11.1)
- Pyspark (v 3.4.0)
- scikit-learn (v 1.3.0)
- scipy (v 1.11.0)
- xgboost (v 1.7.6)
- numpy (v 1.24.2)
- matplotlib (v 3.7.1)
- Pandas (v 1.5.3)

## lib folder
The lib folder is divided into six files:
- `__init__.py` used to package the modules
- `useful.py` stores functions useful in any circumstances.
- `prepare_data.py` contains every function necessary to prepare data to be input into a SKLEARN model.
- `train.py` contains every function needed to train a model (score and train).
- `show.py` stocks functions to display valuable insights on model optimization.
- `merge.py` stores a function to handle the training of a model from the beginning to the end (prepare data, train the model, and display insights).

## training folder
The following files are in the training folder:
- dico.csv contains the meaning of the different values of each variable.
- pred_dpe.csv contains the training dataset for the second model.
- pred_tremi contains the training dataset for model 1.
- pred_dpe_old.csv contains the training dataset for the second model with continuous values for DPE consumption and GES emission.
- pred_tremi_old contains the training dataset for model 1 with continuous values for surface variable.

## notebook folder
The following files handle the different tasks:
- ``insights.ipynb`` shows the repartitions of the variables in the training sets.
- ``to_categorical.ipynb`` The attempt to predict the continuous variable surface was not conclusive. Therefore, this notebook determines the best possible breaks to predict the categorical surface.
- ``multivariate_pred_tremi.ipynb`` Given the data with missing values, try to fill in the blanks of every missing variable (model 1).
- ``pred_tremi_v2.ipynb`` for every variable missing, optimize hyperparameters of different models to input missing data.
- ```balanced.ipynb``` The results for predicting the heating production were not as good as expected. Therefore, I tried different solution to enhance accuracy of the model
- ``pred_dpe.ipynb`` optimize hyperparameters of different models to classify housings according to DPE concumption and GES emission.
- ``pred_renov.ipynb`` optimize hyperparameters to classify housings on whether they need a renovation or not.
- ``show_old_results_dpe`` compare DPE regression and classification.
- ``show_old_results_surface`` compare surface regression and classification.

