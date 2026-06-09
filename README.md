# CTGAN
CTGAN-Based Synthetic Breast Cancer Data Generation
Project Overview

This project demonstrates how to generate realistic synthetic breast cancer patient records using CTGAN (Conditional Tabular Generative Adversarial Network).

The Breast Cancer dataset from Scikit-Learn is used as the source dataset. CTGAN learns the statistical distributions and relationships between tumor-related features and diagnosis labels, then generates entirely new synthetic patient records that preserve these patterns without exposing real patient information.

Problem Statement

Medical datasets often contain sensitive patient information and cannot be freely shared due to privacy concerns. This limits research, experimentation, and machine learning development.

The goal of this project is to create privacy-preserving synthetic data that maintains the characteristics of the original dataset while protecting patient confidentiality.

Dataset

Source: Scikit-Learn Breast Cancer Dataset

Dataset Characteristics:

569 patient records
30 tumor-related features
1 target column (Diagnosis)
Binary Classification Problem

Target Values:

0 = Malignant
1 = Benign
Technologies Used
Python
Pandas
Scikit-Learn
CTGAN
Table Evaluator
Matplotlib
Seaborn
Project Workflow
Load the Breast Cancer dataset
Train a CTGAN model on real patient records
Generate synthetic patient records
Compare real and synthetic data samples
Evaluate data quality using visualization and similarity metrics
Key Concepts
CTGAN

CTGAN (Conditional Tabular GAN) is a Generative Adversarial Network specifically designed for tabular datasets.

It consists of:

Generator: Creates synthetic records
Discriminator: Distinguishes between real and synthetic records

Through adversarial training, the generator learns to create realistic tabular data.

Synthetic Data

Synthetic data is artificially generated data that resembles real-world data while avoiding direct exposure of sensitive information.

Benefits:

Privacy Preservation
Data Sharing
Research and Experimentation
Machine Learning Development
Results

The generated synthetic data closely resembles the statistical properties of the original breast cancer dataset while maintaining privacy.

Future Improvements
Increase CTGAN training epochs
Compare CTGAN with TVAE
Evaluate additional synthetic data quality metrics
Apply synthetic data generation to larger healthcare datasets
Author

Sivakiran Polu
