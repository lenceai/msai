# CharityML Project

## Description

CharityML is a fictitious charity organization located in the heart of Silicon Valley that was established to provide financial support for people eager to learn machine learning. After nearly 32,000 letters were sent to people in the community, CharityML determined that every donation they received came from someone that was making more than $50,000 annually.

To expand their potential donor base, CharityML has decided to send letters to residents of California, but to only those most likely to donate to the charity. With nearly 15 million working Californians, CharityML has brought you on board to help build an algorithm to best identify potential donors and reduce overhead cost of sending mail.

**Goal**: Evaluate and optimize several different supervised learners to determine which algorithm will provide the highest donation yield while also reducing the total number of letters being sent.

---

## Getting Started

### Prerequisites

To use CharityML, you should have a Python environment set up. It is recommended to use Conda.

**Important**: Always run the following command before running any code in this project:

```bash
conda activate CharityML
```

### Installation

1.  **Clone the repository** (if you haven't already).
2.  **Install Dependencies**:
    The project requires Python 3.12 (or a compatible version). Install the required packages using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

---

## Project Outline

The project proceeds through the following stages:

1.  **Data Exploration**:
    *   Load the census dataset.
    *   Analyze the distribution of features (age, workclass, education, etc.).
    *   Identify skewed continuous features and normalize them.

2.  **Data Preprocessing**:
    *   **Transform Skewed Data**: Log-transform features like capital-gain and capital-loss.
    *   **Normalizing Numerical Features**: Scale numerical features to a range of 0-1.
    *   **Data Encoding**: Convert categorical variables (e.g., 'workclass', 'education_level') into numerical values using One-Hot Encoding.
    *   **Shuffle and Split Data**: Split data into training and testing sets to evaluate model performance.

3.  **Evaluating Model Performance (Benchmark)**:
    *   Establish a naive predictor (e.g., guessing everyone donates) to serve as a baseline.
    *   Calculate accuracy and F-beta score for the naive predictor.

4.  **Supervised Learning Models**:
    *   Select three supervised learning algorithms suitable for the problem (e.g., AdaBoost, SVC, Gradient Boosting).
    *   Initialize and train these models on the training set.
    *   Predict outcomes on the test set.
    *   Compare models based on Accuracy, F-score, and Training Time.

5.  **Model Tuning**:
    *   Choose the best candidate model.
    *   Perform Grid Search Optimization to fine-tune hyperparameters.
    *   Evaluate the optimized model against the unoptimized benchmark.

6.  **Feature Importance**:
    *   Extract feature importance from the best model.
    *   Identify the top 5 features driving the prediction.
    *   Attempt to reduce the feature set and re-evaluate performance to check for efficiency gains.

---

## Project Workflow

```mermaid
graph TD
    A[Start] --> B[Load Data];
    B --> C{Data Exploration};
    C --> D[Preprocessing];
    D --> E[Log Transform Skewed Features];
    E --> F[Normalize Numerical Features];
    F --> G[One-Hot Encode Categorical Features];
    G --> H[Split Data Train/Test];
    H --> I[Benchmark Naive Predictor];
    I --> J[Select Candidate Models];
    J --> K[Train & Evaluate Models];
    K --> L{Select Best Model};
    L --> M[Hyperparameter Tuning (Grid Search)];
    M --> N[Final Model Evaluation];
    N --> O[Extract Feature Importance];
    O --> P[Conclusion];
```

