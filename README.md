# Ant-based metaheuristics in CGP

This repository contains the associated code for the paper __Ant-based Metaheuristics Struggle to Solve the Cartesian Genetic Programming Learning Task__ as well as all the results we achieved in our large benchmark.


## Source

The code folder contains the source code used to produce the results. It is based on an open source Rust implementation of CGP ([https://github.com/CuiHen/Equidistant_Reorder](https://github.com/CuiHen/Equidistant_Reorder)). We chose to also use Rust as it runs fast which we needed for the large number of benchmark datasets


## Results

The results folder contains the formated tables for all results achieved across the full benchmark, including some additional details not in the papers table(s).
The paper featured only a limited subset of the datasets to safe space and focus on the essentials. We made that selection based on the resoning described in the text. The full list is:

### Symbolic Regression Datasets

- Koza-3
- Nguyen-7
- Pagie-1

### Regression Datasets

- Abalone
- Appliances Energy Prediction
- Bike Sharing Dataset - Day
- Bike Sharing Dataset - Hour
- California Housing
- Forest Fires
- Air Quality
- Wine Quality (Red)
- Wine Quality (White)

### Classification Datasets

- Adult
- Apnea-ECG
- Bach Chorales Harmony
- Car Evaluation
- Chronic Kidney Disease
- Diabetes
