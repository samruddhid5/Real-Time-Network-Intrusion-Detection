# Real-Time-Network-Intrusion-Detection

This project focuses on detecting network intrusions in real-time using machine learning techniques like Convolutional Neural Networks (CNN). The model has been tested on the UNSW-NB15 dataset and achieves high accuracy in detecting intrusions.

## Project Overview
The main objectives of this project are:
- To preprocess network traffic data.
- To implement and train a CNN model for real-time intrusion detection.
- To evaluate the model's performance based on various metrics (accuracy, precision, recall, F1-score).

## Technologies Used
- **Python**: Core programming language.
- **Jupyter Notebook**: For running and testing the model.
- **TensorFlow/Keras**: For implementing CNN.
- **Pandas, NumPy**: For data manipulation.
- **Matplotlib, Seaborn**: For data visualization.

## Dataset
- The dataset used is [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) for evaluating the model.
- It includes network traffic data with labeled attack types.

## Model Performance
- Test Accuracy: **99.94%**
- Precision: **99.93%**
- Recall: **99.94%**
- F1 Score: **99.93%**

## How to Run
1. Clone the repository:
    ```
    git clone https://github.com/yourusername/Real-Time-Network-Intrusion-Detection.git
    ```
2. Install the required libraries:
    ```
    pip install -r requirements.txt
    ```
3. Open the Jupyter notebook and run the model:
    ```
    jupyter notebook Real_time_network_intrusion_detection.ipynb
    ```

## Future Improvements
- Explore more datasets for model evaluation.
- Implement additional deep learning models (e.g., GRU, LSTM).
