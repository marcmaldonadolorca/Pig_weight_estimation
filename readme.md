# Pig Weight Estimation Project

This project is part of a **Final Degree Project (TFG)** in **Computer Engineering** at the **Autonomous University of Barcelona (UAB)**. The main objective is to develop a reliable model to estimate the weight of pigs using machine learning, contributing to the field of **Smart Farming**. The project is powered by **computer vision techniques** and **deep learning**, utilizing depth images of pigs and regression models.

## Project Overview
The project addresses the challenge of estimating pig weights automatically, which is crucial for maximizing farm efficiency and ensuring that pigs meet the optimal weight of 120kg before being sent to slaughterhouses. The project integrates a 3D camera system with traditional scales to collect depth images and weight data. These data are then processed using **Convolutional Neural Networks (CNNs)** to develop a predictive model.

You can find the paper following this path reports/informe_final/informe_final.pdf

## Technologies Used
- **Python**: Core programming language.
- **TensorFlow/Keras**: Framework for building deep learning models.
- **OpenCV**: Used for image processing.
- **Google Colab**: Platform for model training and development.
- **Jira**: Project management and planning tool.
- **Open3D**: Library for 3D data processing.

## Methodology
The project follows an agile **Kanban methodology** with tasks structured in four columns: To Do, In Progress, Review, and Done. The dataset includes:
- **Depth images** taken at the time of weighing.
- **XLS file** with metadata including pig ID, weight, and timestamp.
- **Infrared images** for supplemental data analysis.

The main phases of development included:
1. **Data Preparation**: Mapping images to their corresponding weights.
2. **Segmentation**:
   - **Morphological techniques** for background subtraction.
   - **Object detection** using **YOLOv5**.
   - **Semantic segmentation** with **U-Net** architectures.
3. **Regression**:
   - Initial linear regression with extracted features.
   - **CNN-based regression** for improved accuracy.
4. **3D Data Processing**:
   - Construction of 3D point clouds and meshes using **Open3D**.

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/pig-weight-estimation.git
