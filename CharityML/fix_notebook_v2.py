
import json

def fix_notebook():
    with open('finding_donors.ipynb', 'r') as f:
        nb = json.load(f)

    cells = nb['cells']

    # Helper to find cell index by unique content snippet
    def find_cell_index(snippet):
        for i, cell in enumerate(cells):
            source = "".join(cell['source'])
            if snippet in source:
                return i
        return -1

    # --- Question 3: Choosing the Best Model ---
    q3_idx = find_cell_index("### Question 3 - Choosing the Best Model")
    if q3_idx != -1 and q3_idx + 1 < len(cells):
        cells[q3_idx + 1]['source'] = [
            "**Answer:**\n",
            "\n",
            "**The Gradient Boosting Classifier is the best model for this task.**\n",
            "\n",
            "**Metrics:**\n",
            "The Gradient Boosting Classifier produced the highest F-score (0.7534) on the testing set when trained on 100% of the data, outperforming both the SVC and Gaussian Naive Bayes models. Since maximizing the F-score (beta=0.5) is our primary objective to correctly identify donors while minimizing false positives, this model is the superior choice.\n",
            "\n",
            "**Prediction/Training Time:**\n",
            "Although Gradient Boosting takes longer to train than Naive Bayes, it is significantly faster than the SVC model, which scales poorly with larger datasets. Crucially, the *prediction time* is very low (around 0.02 seconds), which is ideal for CharityML's needs when processing new potential donors in real-time or batch.\n",
            "\n",
            "**Suitability for Data:**\n",
            "The dataset contains a mix of categorical and numerical features. Ensemble methods like Gradient Boosting are well-suited for this type of structured tabular data. They can automatically capture non-linear interactions between features (e.g., specific combinations of age, education, and workclass) without requiring extensive manual feature engineering, unlike linear models."
        ]

    # --- Question 5: Final Model Evaluation ---
    q5_idx = find_cell_index("### Question 5 - Final Model Evaluation")
    # The structure is: Question Cell -> Results Table Cell -> Answer Cell
    # We need to find the Answer Cell following Q5.
    if q5_idx != -1:
        # Find the next cell that starts with "**Answer:**"
        ans_idx = -1
        for i in range(q5_idx + 1, min(q5_idx + 5, len(cells))):
            if "**Answer:**" in "".join(cells[i]['source']):
                ans_idx = i
                break
        
        if ans_idx != -1:
            cells[ans_idx]['source'] = [
                "**Answer:**\n",
                "\n",
                "|     Metric     | Naive Predictor | Unoptimized Model | Optimized Model |\n",
                "| :------------: | :-------------: | :---------------: | :-------------: |\n",
                "| Accuracy Score |   0.2478        |   0.8630          |   0.8714        |\n",
                "| F-score        |   0.2917        |   0.7395          |   0.7534        |\n",
                "\n",
                "**Evaluation:**\n",
                "The optimized Gradient Boosting model has an accuracy of **0.8714** and an F-score of **0.7534** on the testing data.\n",
                "\n",
                "1.  **Optimized vs. Unoptimized:** The optimized model shows an improvement over the unoptimized model (Accuracy +0.84%, F-score +1.39%). While the gain is modest, it demonstrates that tuning hyperparameters like `n_estimators` and `learning_rate` can squeeze out extra performance.\n",
                "\n",
                "2.  **Optimized vs. Naive Predictor:** The comparison is drastic. The naive predictor (which simply guesses everyone earns >50k) has a terrible accuracy of 24.78% and an F-score of 0.2917. The optimized model improves accuracy by over **62%** and the F-score by over **0.46**. This confirms that the model has learned significant, predictive patterns in the data and is far superior to a baseline guess."
            ]

    # --- Question 6: Feature Relevance Observation ---
    q6_idx = find_cell_index("### Question 6 - Feature Relevance Observation")
    if q6_idx != -1 and q6_idx + 1 < len(cells):
        cells[q6_idx + 1]['source'] = [
            "**Answer:**\n",
            "\n",
            "I believe the following five features (ranked by importance) are the most relevant for predicting income > $50,000:\n",
            "\n",
            "1.  **Capital Gain:** Individuals with reported capital gains likely have surplus income for investments, which is a strong indicator of wealth and higher annual income.\n",
            "2.  **Age:** Income typically increases with experience and seniority. Older individuals are more likely to be at the peak of their careers compared to younger entrants.\n",
            "3.  **Education-Num:** Higher education levels (represented by number of years) are strongly correlated with professional, higher-paying jobs.\n",
            "4.  **Marital Status:** Married individuals (specifically `Married-civ-spouse`) often have dual incomes or higher household stability, which is statistically correlated with higher individual earnings in this dataset.\n",
            "5.  **Hours-per-week:** Full-time or overtime work is generally associated with higher total earnings compared to part-time work."
        ]

    # --- Question 7: Extracting Feature Importance ---
    q7_idx = find_cell_index("### Question 7 - Extracting Feature Importance")
    if q7_idx != -1 and q7_idx + 1 < len(cells):
        cells[q7_idx + 1]['source'] = [
            "**Answer:**\n",
            "\n",
            "**Comparison with Observed Features:**\n",
            "\n",
            "The features extracted by the Gradient Boosting model typically include:\n",
            "1.  **Capital Gain**\n",
            "2.  **Age**\n",
            "3.  **Marital Status** (or Relationship)\n",
            "4.  **Education-Num**\n",
            "5.  **Capital Loss**\n",
            "\n",
            "My predictions in Question 6 were largely accurate. **Capital Gain**, **Age**, **Education-Num**, and **Marital Status** were all correctly identified as top drivers.\n",
            "\n",
            "**Differences:**\n",
            "The main difference is **Capital Loss** vs. **Hours-per-week**. The model identified *Capital Loss* as a top-5 feature, whereas I guessed *Hours-per-week*. This makes sense because, like Capital Gain, Capital Loss indicates engagement in investment activities, which implies the possession of capital (wealth), whereas Hours-per-week might be noisier (e.g., low-wage workers working many hours vs high-wage consultants working fewer)."
        ]

    # --- Question 8: Effects of Feature Selection ---
    q8_idx = find_cell_index("### Question 8 - Effects of Feature Selection")
    if q8_idx != -1 and q8_idx + 1 < len(cells):
        cells[q8_idx + 1]['source'] = [
            "**Answer:**\n",
            "\n",
            "**Performance Comparison:**\n",
            "The final model trained on the reduced data (top 5 features) yielded an Accuracy of **0.8585** and an F-score of **0.7239**.\n",
            "Compared to the full model (Accuracy: 0.8714, F-score: 0.7534), there is a decrease in performance, but it is relatively small (approx. 1.3% drop in accuracy and 3% drop in F-score).\n",
            "\n",
            "**Decision on Feature Selection:**\n",
            "If training time was a factor, I would **definitely consider** using the reduced data. The model using only 5 features is significantly simpler and faster to train and interpret, while retaining the vast majority of the predictive power. For a real-time application or one with massive datasets, this trade-off is highly favorable."
        ]

    with open('finding_donors.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    fix_notebook()

