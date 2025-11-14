# Music & Mental Health: An Exploratory Analysis

This project performs an in-depth exploratory data analysis (EDA) and predictive modeling on the "Music & Mental Health Survey Results" dataset from Kaggle. The goal is to uncover relationships between an individual's musical preferences, listening habits, and their self-reported mental health.

### Dataset

The dataset was sourced from a survey conducted on Kaggle and includes 736 responses with data on:
- Demographics (Age)
- Music Habits (Hours per day, Favorite Genre)
- Musical Background (Instrumentalist, Composer)
- Perceived effect of music on mood
- Self-reported scores (0-10) for Anxiety, Depression, Insomnia, and OCD

[Link to the original dataset on Kaggle](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results)

### Project Workflow

The analysis follows a structured, multi-step process:

1.  **Exploratory Data Analysis (EDA):** A deep dive into the data using visualizations to understand distributions, identify correlations, and generate initial hypotheses.
2.  **Statistical Hypothesis Testing:** Using a formal ANOVA test to statistically validate the strongest relationship found during the EDA.
3.  **Predictive Modeling:** Building machine learning models to determine if mental health outcomes can be predicted from the survey data.
4.  **Feature Importance Analysis:** Identifying the most influential factors in the predictive models.

---

### Key Findings & The Story of the Analysis

The analysis revealed a compelling and nuanced story that evolved with each step.

#### Part 1: Initial Exploration Reveals Genre and Music's Effect Matter

The initial EDA showed clear patterns. A key finding was that individuals who felt music **worsens** their mood reported significantly higher levels of Anxiety, Depression, and Insomnia.

![Distribution of All Health Metrics by Perceived Music Effect](plots/09_all_metrics_by_music_effect.png)

Furthermore, there were noticeable differences in anxiety and depression scores across favorite genres. Fans of Rock, Metal, and Folk tended to report higher anxiety on average.

![Anxiety Scores by Favorite Genre](plots/04_anxiety_by_genre.png)

This was confirmed with a formal **ANOVA test**, which resulted in a **p-value of 0.0003**. This provided statistical proof that the perceived effect of music has a significant relationship with self-reported anxiety scores.

#### Part 2: The "Accuracy Paradox" in Predictive Modeling

The next step was to build a model to predict whether a person has "High Anxiety" (defined as a score of 7/10 or higher). An initial Logistic Regression model achieved **95% accuracy**, which seemed incredible.

However, a closer look at the results revealed a classic pitfall: **class imbalance**. The model had learned it could be highly accurate by simply guessing "High Anxiety" for everyone, completely failing to identify anyone who did *not* have high anxiety.

#### Part 3: Building a Realistic Model and a Surprising Conclusion

After balancing the dataset by adjusting the "High Anxiety" threshold to `Anxiety >= 7`, we trained two models:
1.  **Logistic Regression (Baseline):** Achieved **~53% accuracy**.
2.  **Random Forest (Complex Model):** Achieved **~49% accuracy**.

The simpler Logistic Regression model performed better, indicating that the relationships in the data are not overly complex and the more powerful model was likely overfitting.

While an accuracy of 53% seems low, it's significantly better than a random 50% guess. This tells us that the survey data contains a real, but limited, predictive signal. The low accuracy is not a failure of the model, but a reflection of the inherent difficulty in predicting a complex human condition like anxiety from a few survey questions.

The final and most important step was to ask the model which features it found most useful. The results were surprising and provided the project's main conclusion.

![Top 15 Most Important Features in Predicting High Anxiety](plots/12_feature_importance.png)

### Final Conclusion

While musical taste (like favoring Rock or Metal) does have a connection to self-reported anxiety, it is not the primary driver. The most powerful predictors of high anxiety in this dataset are:

1.  **Age:** A person's age is the single most predictive feature.
2.  **Hours per day:** The amount of time spent listening to music is the second most important.
3.  **Musical Background:** Whether a person is an instrumentalist or composer is more predictive than their favorite genre.

The story of this dataset is that **who a person is (their age) and their habits (listening duration) are more significant predictors of their mental health than their specific musical preferences.**

---

### How to Run This Project

1.  Clone the repository.
2.  Ensure you have Python and the necessary libraries installed:
    ```bash
    pip install pandas matplotlib seaborn scikit-learn kagglehub scipy
    ```
3.  Run the Python script:
    ```bash
    python music_mental_health_analysis.py
    ```
4.  The script will download the dataset, perform the analysis, print results to the console, and save all generated plots to a `/plots` directory.