# Football Match Predictor

This project develops a machine learning model to predict the outcome of football (soccer) matches. It leverages historical match data to identify patterns and predict whether a team will win (`target` = 1) or not win (`target` = 0) a given match.

## Project Overview

The goal of this project is to build a predictive model using `RandomForestClassifier` to forecast match results. The process involves several key steps:

1.  **Data Loading and Initial Exploration**: Loading match data from a CSV file and understanding its structure and basic statistics.
2.  **Feature Engineering**: Transforming raw data into features suitable for machine learning, including:
    *   Converting `date` and `time` columns into numerical representations (`day_code`, `hour`).
    *   Encoding categorical features like `venue` and `opponent` into numerical codes (`venue_code`, `opp_code`).
    *   Creating rolling average statistics (e.g., goals for, goals against, shots, shots on target) for each team to capture recent performance trends.
3.  **Model Training**: Splitting the data into training and testing sets based on date and training a `RandomForestClassifier`.
4.  **Model Evaluation**: Assessing the model's performance using accuracy and precision scores, and analyzing predictions in detail.

## Setup and Dependencies

To run this notebook, you will need the following Python libraries:

*   `pandas`
*   `scikit-learn` (sklearn)

You can install them using pip:
```bash
pip install pandas scikit-learn
```

## Data

The project uses a `matches.csv` file, which is expected to contain various statistics for football matches, including date, time, competition, teams, results, and other match metrics.

## Key Features Engineered

*   `venue_code`: Numerical encoding for match venue (Home/Away).
*   `opp_code`: Numerical encoding for the opponent team.
*   `hour`: Hour of the match derived from the `time` column.
*   `day_code`: Day of the week derived from the `date` column.
*   `gf_rolling`, `ga_rolling`, `sh_rolling`, `sot_rolling`, `dist_rolling`, `fk_rolling`, `pk_rolling`, `pkatt_rolling`: 3-match rolling averages for goals for, goals against, shots, shots on target, distance, free kicks, penalties scored, and penalty attempts, respectively. These provide a measure of a team's recent form.

## Model and Prediction

A `RandomForestClassifier` is used for prediction due to its robustness and ability to handle various data types. The model is trained on data before '2022-01-01' and tested on data from '2022-01-01' onwards.

### Evaluation Metrics

*   **Accuracy**: Measures the overall correctness of predictions.
*   **Precision**: Particularly important in this context, precision measures the proportion of predicted wins that were actually wins. It helps understand how reliable the model's "win" predictions are.

### Interpreting Combined Results

The `merged` DataFrame combines predictions for both teams in a match, allowing for a deeper analysis. For instance, the line `merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] ==0)]["actual_x"].value_counts()` helps identify matches where our model predicted Team X to win and Team Y to not win. It then shows how many of these predictions were actual wins for Team X (`actual_x=1`) versus not wins (`actual_x=0`). This can reveal how often the model correctly identifies a clear winner versus situations where it makes conflicting predictions or false positives.