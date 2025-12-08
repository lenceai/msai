
import json
import pandas as pd
import numpy as np

def add_xgboost_section():
    notebook_path = 'finding_donors.ipynb'
    
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # Remove existing Q9 cells to allow updates
    # We filter out cells that contain specific markers of our section
    cells_to_keep = []
    for cell in nb['cells']:
        src = "".join(cell.get('source', []))
        if "Question 9 - XGBoost vs Gradient Boosting" in src:
            continue
        if "xgb.XGBClassifier" in src and "RandomizedSearchCV" in src:
            continue
        if "Model Comparison: Gradient Boosting vs XGBoost" in src:
            continue
        if "Comparing Models on Testing Data" in src and "XGBoost" in src:
            continue
        if "XGBoost (eXtreme Gradient Boosting)" in src and "Answer:" in src:
            continue
        cells_to_keep.append(cell)
        
    nb['cells'] = cells_to_keep

    # Create new cells
    new_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "----\n",
                "## Question 9 - XGBoost vs Gradient Boosting\n",
                "\n",
                "Does XGBoost score better than the optimized Gradient Boosting model? We will use Hyperparameter Optimization (HPO) to tune XGBoost on approximately 20 different parameters to find the best possible configuration."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import XGBoost and RandomizedSearchCV\n",
                "import xgboost as xgb\n",
                "from sklearn.model_selection import RandomizedSearchCV\n",
                "\n",
                "# Initialize XGBoost Classifier\n",
                "xgb_clf = xgb.XGBClassifier(random_state=42, objective='binary:logistic')\n",
                "\n",
                "# Define a large hyperparameter space (approx 20 parameters)\n",
                "param_dist = {\n",
                "    # 1. General Parameters\n",
                "    'n_estimators': [100, 200, 300, 400, 500, 1000],\n",
                "    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],\n",
                "    'booster': ['gbtree', 'dart'],\n",
                "    \n",
                "    # 2. Tree Booster Parameters\n",
                "    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],\n",
                "    'min_child_weight': [1, 2, 3, 4, 5, 6],\n",
                "    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5], # Minimum loss reduction required to make a further partition\n",
                "    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0], # Subsample ratio of the training instances\n",
                "    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0], # Subsample ratio of columns when constructing each tree\n",
                "    'colsample_bylevel': [0.6, 0.7, 0.8, 0.9, 1.0], # Subsample ratio of columns for each level\n",
                "    'colsample_bynode': [0.6, 0.7, 0.8, 0.9, 1.0], # Subsample ratio of columns for each node (split)\n",
                "    \n",
                "    # 3. Regularization Parameters\n",
                "    'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 1], # L1 regularization term on weights\n",
                "    'reg_lambda': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 1], # L2 regularization term on weights\n",
                "    'max_delta_step': [0, 1, 2, 3, 4, 5, 10], # Maximum delta step we allow each leaf output to be\n",
                "    \n",
                "    # 4. Others\n",
                "    'scale_pos_weight': [1, 2, 3, 4, 5], # Control the balance of positive and negative weights\n",
                "    'base_score': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], # Initial prediction score of all instances\n",
                "    'max_leaves': [0, 10, 20, 30, 40, 50], # Maximum number of nodes to be added\n",
                "    'max_bin': [256, 512, 1024], # Maximum number of discrete bins to bucket continuous features\n",
                "    'grow_policy': ['depthwise', 'lossguide'],\n",
                "    'tree_method': ['auto', 'approx', 'hist'],\n",
                "    'eval_metric': ['logloss', 'error', 'auc']\n",
                "}\n",
                "\n",
                "# Initialize RandomizedSearchCV\n",
                "# n_iter=20 to sample 20 different combinations from the vast space\n",
                "xgb_random = RandomizedSearchCV(estimator=xgb_clf, param_distributions=param_dist, \n",
                "                                n_iter=20, scoring=scorer, cv=3, verbose=2, random_state=42, n_jobs=-1)\n",
                "\n",
                "# Fit the random search model\n",
                "xgb_random.fit(X_train, y_train)\n",
                "\n",
                "# Get the best XGBoost model\n",
                "best_xgb = xgb_random.best_estimator_\n",
                "\n",
                "# Make predictions\n",
                "xgb_predictions = best_xgb.predict(X_test)\n",
                "\n",
                "# Report scores\n",
                "print(\"\\nXGBoost Optimized Model\\n------\")\n",
                "print(\"Best Parameters:\", xgb_random.best_params_)\n",
                "print(\"Accuracy on testing data: {:.4f}\".format(accuracy_score(y_test, xgb_predictions)))\n",
                "print(\"F-score on testing data: {:.4f}\".format(fbeta_score(y_test, xgb_predictions, beta = 0.5)))\n",
                "\n",
                "print(\"\\nTop 20 Trials:\\n------\")\n",
                "results_df = pd.DataFrame(xgb_random.cv_results_)\n",
                "results_df = results_df.sort_values('rank_test_score')\n",
                "display(results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].head(20))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Model Comparison: Gradient Boosting vs XGBoost"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Compare the optimized Gradient Boosting model (from Q5) with XGBoost\n",
                "print(\"Comparing Models on Testing Data:\")\n",
                "print(\"Gradient Boosting (sklearn) - Accuracy: {:.4f}, F-score: {:.4f}\".format(\n",
                "    accuracy_score(y_test, best_predictions), \n",
                "    fbeta_score(y_test, best_predictions, beta = 0.5)))\n",
                "\n",
                "print(\"XGBoost (Optimized)         - Accuracy: {:.4f}, F-score: {:.4f}\".format(\n",
                "    accuracy_score(y_test, xgb_predictions), \n",
                "    fbeta_score(y_test, xgb_predictions, beta = 0.5)))\n",
                "\n",
                "# Simple Bar Chart Comparison\n",
                "import matplotlib.pyplot as plt\n",
                "models = ['Gradient Boosting', 'XGBoost']\n",
                "acc_scores = [accuracy_score(y_test, best_predictions), accuracy_score(y_test, xgb_predictions)]\n",
                "f_scores = [fbeta_score(y_test, best_predictions, beta=0.5), fbeta_score(y_test, xgb_predictions, beta=0.5)]\n",
                "\n",
                "x = np.arange(len(models))\n",
                "width = 0.35\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(8, 6))\n",
                "rects1 = ax.bar(x - width/2, acc_scores, width, label='Accuracy')\n",
                "rects2 = ax.bar(x + width/2, f_scores, width, label='F-score')\n",
                "\n",
                "ax.set_ylabel('Scores')\n",
                "ax.set_title('Model Comparison: Gradient Boosting vs XGBoost')\n",
                "ax.set_xticks(x)\n",
                "ax.set_xticklabels(models)\n",
                "ax.legend(loc='lower right')\n",
                "ax.set_ylim(0, 1.0)\n",
                "\n",
                "def autolabel(rects):\n",
                "    for rect in rects:\n",
                "        height = rect.get_height()\n",
                "        ax.annotate('{:.4f}'.format(height),\n",
                "                    xy=(rect.get_x() + rect.get_width() / 2, height),\n",
                "                    xytext=(0, 3),  # 3 points vertical offset\n",
                "                    textcoords=\"offset points\",\n",
                "                    ha='center', va='bottom')\n",
                "\n",
                "autolabel(rects1)\n",
                "autolabel(rects2)\n",
                "\n",
                "fig.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Answer:**\n",
                "\n",
                "XGBoost (eXtreme Gradient Boosting) is a highly efficient and scalable implementation of gradient boosting. By tuning a wide range of hyperparameters (20 in this case), including regularization terms (`reg_alpha`, `reg_lambda`) which standard Gradient Boosting in sklearn may lack or handle differently, XGBoost often achieves superior performance or similar performance with faster training times.\n",
                "\n",
                "In this comparison, the XGBoost model [DID/DID NOT] outperform the sklearn Gradient Boosting model. (Please verify with the output above). Often, XGBoost provides a slight edge in F-score due to its better handling of overfitting via regularization and its specialized tree pruning algorithms."
            ]
        }
    ]
    
    # Append the new cells to the end of the notebook
    nb['cells'].extend(new_cells)
    
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Successfully updated XGBoost section in {notebook_path}")

if __name__ == "__main__":
    add_xgboost_section()
