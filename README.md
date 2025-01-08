# Closing the Loop: An Automated End-to-End 3D Printing Pipeline for Personalised Drug Formulations  
**Authors**:  
Youssef Abdalla, Martin Ferianc, Haya Alfassam, Atheer Awad, Ruochen Qiao, Miguel Rodrigues, Mine Orlu, Abdul W. Basit, David Shorthouse  

---

## Overview

This project demonstrates a fully automated pipeline for formulation design and 3D printing of personalised drug formulations. By leveraging advanced computational methods and deep learning, we significantly reduce the time and effort required for drug formulation development.

Three-dimensional (3D) printing offers the potential to revolutionise personalised medicines through customised drug formulations. However, conventional trial-and-error approaches remain time-consuming and demand substantial expertise. This repository demonstrates an **automated pipeline** that unifies formulation design, model-based prediction, and 3D printing using selective laser sintering (SLS). By integrating a **Differential Evolution-based optimiser** with a **Deep Learning Ensemble**, we streamline the creation and validation of novel drug formulations with confidence intervals on printing parameters.

Please note that the data used to train the models is private, for access to the training and evaluation data please contact the authors.

### Key Features
- **Regressor**: Replaced with an Ensemble NN for enhanced predictive accuracy. The main script for the model can be found here: `src/recommenders/ensemble_mlp.py`
- **Optimisation Algorithms**: All optimisers are available in the `src/recommenders/proportion` directory.
- **Tuning Scripts**: Utilities for hyperparameter tuning can be found in the `src/utils` directory.

![Pipeline Diagram](docs/workflow.svg)

---

## Abstract
Three-dimensional (3D) printing offers a promising approach to creating personalised medicines. However, traditional trial-and-error methods are costly and require significant expertise, posing challenges for tailoring treatments to individual patients. 

To address these challenges, we developed a novel pipeline for formulation design and 3D printing using selective laser sintering (SLS). This pipeline integrates a Differential Evolution-based optimiser to generate formulations for desired drugs and a Deep Learning Ensemble to predict optimal printing parameters along with confidence intervals.

---

## Highlights
- **Diverse Formulations**: The pipeline generated formulations with a wide variety of materials and high printability probabilities.
- **High Success Rate**: 80% of generated drug formulations were successfully printed, with 92% accuracy in predicting printing parameters.
- **Efficiency**: Development and printing time for new drug formulations was reduced from approximately one week to a single day.

---

## Contact and Citation

If you use this pipeline in your research, kindly cite our associated manuscript. For enquiries regarding the data or the underlying algorithms, please [open an issue](#) or contact the authors directly.

- **Lead Author**: Youssef Abdalla (youssef.abdalla.16@ucl.ac.uk)  
- **Corresponding Author**: David Shorthouse  

Please note that the data used to train the models is private. For access to the training and evaluation data, please contact the authors.