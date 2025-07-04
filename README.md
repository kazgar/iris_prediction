# Iris Prediction

Despite iris species prediction done on this specific dataset (available here https://archive.ics.uci.edu/dataset/53/iris) being a rather trivial task (SOTA performance ~100%), I decided to give it a shot with 6 different models, while learning how to maintain a proper repository structure. I also wanted to practice data inspection and manipulation in order to enhance my understanding of the data (by utilizing those techniques), and in turn, model performance.

## Features

- Data loading and exploration
- Feature engineering (bit of an overstatement)
- Model training
- Model evaluation and accuracy reporting
- Prediction on new data

## Getting Started

1. **Clone the repository:**
    ```bash
    git clone https://github.com/kazgar/iris_prediction.git
    cd iris_prediction
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the main script:**
    ```bash
    python inference.py <model>
    ```

## Project Structure

- `src/iris_prediction` — Main scripts for data visualization, preparation, model training and inference.
- `data/` — Dataset files
- `models/` — Saved model parameters
- `figures/` - Saved graphs


## Usage

In order to use the script, all you nede to is:
    ```bash
    python inference.py <model>
    ```

Where model is one of ['lr', 'svc', 'lsvc', 'gbc', 'hgbc', 'rf'], which stand for <b>Logistic Regression</b>, <b>Support Vector Classifier</b>, <b>Linear Support Vector Classifier</b>, <b>Gradient Boosting Classifier</b>, <b>Histogram-based Gradient Boosting Classifier</b>, and <b>Random Forest Classifier</b> respectively. Parameters for the models have been selected in advance, leveraging scikit-learns' GridSearchCV to find best parametrs (saved in `models/`). 

After executing the script you will see that the model you selected will give 10 predictions one-by-one with the result (correct/incorrect) printed out in the terminal.

Altough pretty simple, I enjoyed working on this project. Feel free to use it in any way you wish (but accordingly to MIT License). 

## License

This project is licensed under the MIT License.

## Acknowledgements

- [UCI Machine Learning Repository: Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris)
- [scikit-learn](https://scikit-learn.org/)