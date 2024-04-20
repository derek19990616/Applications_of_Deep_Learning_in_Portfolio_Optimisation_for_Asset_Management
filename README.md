# Applications of Deep Learning in Portfolio Optimisation for Asset Management
This repository contains the implementation of various deep learning models to enhance asset management strategies and optimize portfolio allocations. The focus is on utilising Convolutional Neural Networks (CNNs), Long Short-Term Memory networks (LSTMs), and Temporal Convolutional Networks (TCNs) to predict asset returns and allocate portfolios dynamically.

## Project Overview
The project explores advanced machine learning techniques to predict financial market movements and optimises asset allocations to maximise returns while minimising risk. Techniques such as Bayesian Model Averaging (BMA) are employed to enhance model performance by integrating predictions from multiple models.

## Installation
To set up a local copy of the project, follow these steps:

Clone the repository:
bash
Copy code
git clone https://github.com/derek19990616/DLProject.git
cd DLProject
Install required packages:
bash
Copy code
pip install -r requirements.txt
## Dependencies
This project requires the following packages:

Python 3.8 or later
TensorFlow 2.x
Keras
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook
QuantStats
Scipy
Statsmodels
## Usage
Open the Jupyter Notebooks in the repository to view the implementations:

bash
Copy code
jupyter notebook
Navigate to the desired notebook and run the cells to see the results.

## Models Implemented
CNN: Utilises convolutional layers to capture the temporal dependencies in asset prices.
LSTM: Employs memory cells to understand long-term dependencies in time series data.
TCN: Applies causal convolutions, providing a robust architecture for sequence modeling.
Ensemble Learning: Combines predictions from multiple models using Bayesian Model Averaging to improve accuracy.
## Results
The models were evaluated based on their mean squared error (MSE), accuracy, and their ability to maximize the Sharpe ratio through strategic portfolio allocations. Each model's performance is detailed in the respective Jupyter Notebook with visual insights and statistical analysis.

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your enhancements.

## License
Distributed under the MIT License. See LICENSE for more information.

## Contact
Haochen Pan - panhaochen0616@gmail.com
##Acknowledgements
Thanks to all the contributors who have invested their time in improving this project.
Special thanks to the community for providing insights and feedback on machine learning in finance.
