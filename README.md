<!-- Badges: build your brand at the top -->
[![GitHub stars](https://img.shields.io/github/stars/hemathens/kaggle-projects?style=social)](https://github.com/hemathens/kaggle-projects/stargazers)
[![Kaggle Profile](https://img.shields.io/badge/Kaggle-hem%20ajit%20patel-20BEFF?logo=kaggle)](https://www.kaggle.com/hemajitpatel)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Hem%20Ajit%20Patel-0A66C2?logo=linkedin)](https://www.linkedin.com/in/hem-patel19)
[![GitHub](https://img.shields.io/badge/GitHub-hemathens-181717?logo=github)](https://github.com/hemathens)

# Machine Learning Projects, Notebooks and Datasets

Welcome to my repository of Machine Learning projects, notebooks, and datasets.  
This repo showcases my journey and experiments in Machine Learning, Data Science, and Exploratory Data Analysis through real-world problems and custom-created datasets.

---

## Projects

| #  | Notebook                         | Description                                      | Link |
|----|----------------------------------|--------------------------------------------------|------|
| 1  | **Digits Prediction**            | Classify handwritten digits (MNIST).             | [![Digits](https://img.shields.io/badge/-Digits-7b61ff?style=for-the-badge&logo=python&logoColor=white)](https://www.kaggle.com/hemajitpatel/digits-prediction-hem) |
| 2  | **House Price Prediction**       | Predict house prices with regression models.     | [![HousePrice](https://img.shields.io/badge/-HousePrice-00b894?style=for-the-badge&logo=googlecloud&logoColor=white)](https://www.kaggle.com/hemajitpatel/house-price-hem) |
| 3  | **Titanic Survival**             | Who survived the Titanic? Feature engineering + models. | [![Titanic](https://img.shields.io/badge/-Titanic-0f4c81?style=for-the-badge&logo=gitlab&logoColor=white)](https://www.kaggle.com/hemajitpatel/titanic-hem) |
| 4  | **Heads or Tails**               | Predict heads or tail from a section of an image | [![HeadsOrTails](https://img.shields.io/badge/-HeadsOrTails-ff7b00?style=for-the-badge&logo=opencv&logoColor=white)](https://www.kaggle.com/code/hemajitpatel/heads-or-tails-hem) |
| 5  | **Superheros Abilities Dataset** | Sample usage notebook for superheroes dataset    | [![Superheroes](https://img.shields.io/badge/-Superheroes-1abc9c?style=for-the-badge&logo=superuser&logoColor=white)](https://www.kaggle.com/code/hemajitpatel/superheros-abilities) |
| 6  | **Rock vs Mine**                 | Predicts if an object is a rock or a mine using sonar data. | [![RvsM](https://img.shields.io/badge/-RvsM-ff3b30?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1yoUOlJD6ch8ZlxdqiLbBfI6iT6ozt-Al?usp=sharing) |

---

## Notebooks

---

### 1) Digits Prediction  
**File:** `digits-prediction.ipynb`

**What’s inside**
- Dataset: MNIST (60k train / 10k test) — EDA showing sample digits and class balance.  
- Preprocessing: normalize pixel values to [0,1], reshape for CNN (`28x28x1`), one-hot encode labels.  
- Augmentation (optional): small rotations, shifts, random zoom to improve generalization.  
- Models included:
  - Baseline MLP (dense → relu → dropout → softmax).  
  - CNN (Conv2D → BatchNorm → ReLU → MaxPool → Dropout → Dense).  
  - Optionally a Transfer Learning variant using a small pretrained encoder (for experiments).
- Training artifacts: training/validation curves, confusion matrix, per-class accuracy, SavedModel export.

**How model is trained**
- Algorithm: Convolutional Neural Network (Keras/TensorFlow).  
- Loss / Optimizer: `categorical_crossentropy` with `Adam` (lr = `1e-3` default).  
- Batch size / Epochs: `batch_size=64`, `epochs=20–50` with `EarlyStopping(patience=5, restore_best_weights=True)`.  
- Callbacks: `ModelCheckpoint`, `ReduceLROnPlateau`, `TensorBoard` for visual checks.  
- Regularization: Dropout (`0.25–0.5`), BatchNorm, L2 (optional).

**Data split**
- Use MNIST standard split: 60,000 training, 10,000 test.  
- Create validation from train (e.g., 90/10): → `train=54k`, `val=6k`, `test=10k`.  
- Example snippet:
```py
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.10, stratify=y_train_full, random_state=42)
```


### 2) House Price Prediction  
**File:** `house-price.ipynb`

**What’s inside**
- Dataset: Kaggle House Prices — rich tabular dataset with numeric & categorical features.  
- EDA: distribution plots, missingness heatmap, correlations, target skew (log-transform).  
- Feature engineering: date features, polynomial & interaction terms, categorical encoding (One-Hot / Target Encoding), outlier handling.  
- Pipelines: `ColumnTransformer` for numeric + categorical, `Pipeline` for end-to-end training.  
- Models compared:
  - Linear Regression  
  - Lasso  
  - RandomForest  
  - XGBoost / LightGBM  
  - Stacking ensemble (meta-model).  
- Explainability: feature importance (tree SHAP), partial dependence plots.  

**How model is trained**
- Algorithm: Ensemble regression models (RandomForest, XGBoost, LightGBM) + stacking.  
- Loss / Objective: regression metrics (`MSE`, `RMSE`). Target often `log1p` transformed to stabilize variance.  
- Cross-validation: 5-fold CV with Out-of-Fold (OOF) predictions for stacking.  
- Hyperparameter tuning: `RandomizedSearchCV` or `Optuna` for RF, XGB, LGBM.  
- Typical tuning knobs:
  - XGBoost → `n_estimators=200–2000`, `learning_rate=0.01–0.2`, `max_depth=3–10`, `subsample=0.5–1.0`.  

**Pipeline sketch**
```py
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
preprocessor = ColumnTransformer([...])
model = Pipeline([
    ('prep', preprocessor),
    ('clf', xgb.XGBRegressor(...))
])
```


### 3) Titanic Survival  
**File:** `titanic.ipynb`

**What’s inside**
- EDA: survival rate by sex/class/age, missingness patterns (Age, Cabin).  
- Feature engineering: extract `Title` from Name, family size (`SibSp + Parch`), deck from Cabin (where possible), fill missing Age via median or model imputation.  
- Encoding: Sex → binary, Embarked → one-hot; Fare binned if useful.  
- Models compared:
  - Logistic Regression (baseline)  
  - RandomForest  
  - XGBoost  
- Model interpretability: coefficient table for Logistic Regression, SHAP or Permutation Importance for tree-based models.  

**How model is trained**
- Algorithms: `LogisticRegression` (L2), `RandomForestClassifier`, `XGBClassifier`.  
- Cross-validation: `StratifiedKFold(n_splits=5)` to preserve class balance.  
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC — plus confusion matrix.  
- Typical hyperparameters:
  - RandomForest → `n_estimators=100–500`, `max_depth=None` or tuned.  
  - XGBoost → `learning_rate=0.01–0.2`, `max_depth=3–8`.  

**Data split**
- Use stratified split to keep survival ratio consistent:  
```py
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
```


### 4) Heads or Tails  
**File:** `heads-or-tails.ipynb`

**What’s inside**
- Problem: classify an image patch as **heads** or **tails**. Dataset: custom / cropped images.  
- Preprocessing: resize (e.g., `128x128`), normalize, augmentation pipeline (flips, rotations, brightness jitter).  
- Model choices:
  - Lightweight custom CNN (baseline).  
  - Transfer learning using **MobileNetV2** or **EfficientNetB0** with fine-tuning for better accuracy.  
- Training utilities: class weights (if imbalance), `ImageDataGenerator` or `tf.data` pipeline, **Grad-CAM** for explainability.  

**How model is trained**
- Loss / Optimizer: `binary_crossentropy`, **Adam** or **SGD with momentum** for fine-tuning.  
- Batch size / Epochs: `batch_size=32`, `epochs=20–40`, with `EarlyStopping` + `ReduceLROnPlateau`.  
- Fine-tuning schedule: freeze base model for *N* epochs, then unfreeze top *k* layers and continue training with a lower learning rate (`1e-4 → 1e-5`).  

**Data split**
- Example: `train : val : test = 70 : 15 : 15`, or keep `train/val = 80/20` with a separate test set if available.  
- Ensure stratified split by label.  


### 5) Superheros Abilities Dataset  
**File:** `superheros-abilities.ipynb`

**What’s inside**
- EDA: distribution of powers, missing data handling for free-text attributes.  
- Feature engineering: converting textual powers into categorical features (one-hot or embeddings), aggregate scores (e.g., `power_score`).  
- Experiments:
  - **Clustering**: KMeans / HDBSCAN to discover archetypes.  
  - **Classification / Regression**: predict alignment or power level using RandomForest / XGBoost.  
  - **Dimensionality reduction**: PCA / t-SNE / UMAP for visualizations.  

**How model is trained**
- Supervised tasks: RandomForest / XGBoost with 5-fold CV.  
- Clustering: hyperparameter sweep on `k` with silhouette scores / Davies-Bouldin index.  
- Text features: simple TF-IDF → PCA, or embedding pipeline if needed.  

**Data split**
- Predictive tasks: `train_test_split(X, y, test_size=0.20, random_state=42)` → 80/20.  
- Unsupervised tasks: use full dataset, with holdouts only for downstream validation.  


### 6) Rock vs Mine (RvsM)  
**File:** `rock-vs-mine.ipynb` (Colab link available)

**What’s inside**
- Dataset: Sonar / echo features (e.g., UCI Sonar dataset or similar).  
- Preprocessing: scaling with `StandardScaler`, optional signal preprocessing (smoothing, FFT features).  
- Models compared:
  - Logistic Regression (baseline)  
  - SVM (RBF, `probability=True`)  
  - RandomForest / XGBoost (strong tree baselines)  
- Model evaluation: Accuracy, ROC-AUC, confusion matrix, Precision, Recall — important for safety-critical detection.  

**How model is trained**
- Algorithms: SVM (RBF), RandomForest, XGBoost.  
- Cross-validation: `StratifiedKFold(n_splits=5)` with grid/random search.  
- Example hyperparameters:
  - SVM → `C ∈ {0.1, 1, 10}`, `gamma ∈ {'scale','auto'}` or tuned via logspace.  
  - XGBoost → `learning_rate=0.01–0.1`, `max_depth=3–6`.  
- Calibration / Tuning: decision threshold analysis using precision-recall tradeoff.  
  - If false negatives are costly, optimize for **recall** with constrained precision.  

**Data split**
- Typical:  
```py
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=1
)
```

> 📁 Notebooks live inside `/notebooks/` and include well-commented, reproducible code.

---

## Datasets

### 1. 🔗 [Code Similarity Dataset (Python Variants)](https://www.kaggle.com/datasets/hemajitpatel/code-similarity-dataset-python-variants)

- **Description**: This dataset contains Python code variants with labels indicating similarity or difference, useful for training models in code comparison, plagiarism detection, and embeddings.
- **Files**: `.csv` with `code1`, `code2`, `label` columns
- **Use Case**: Code similarity models, Siamese Networks, NLP + CodeBERT experiments

### 2. 🔗 [Superheroes Abilities Dataset](https://www.kaggle.com/datasets/hemajitpatel/superheros-abilities-dataset)

- **Description**: Contains detailed profiles of superheroes including powers, stats, publisher, alignment, and more.
- **Files**: `.csv` or `.xlsx` format
- **Use Case**: EDA, visualization, classification models (e.g., predict alignment, power levels)
- 
---

## 🔧 Technologies Used

| Tool / Library       | Purpose                          |
|----------------------|----------------------------------|
| Python               | Core programming language        |
| Pandas, NumPy        | Data manipulation & analysis     |
| Matplotlib, Seaborn  | Data visualization               |
| Scikit-learn         | ML modeling                      |
| TensorFlow / PyTorch | Deep learning (future work)      |
| Jupyter Notebook     | Interactive code & documentation |
| Kaggle API           | Dataset handling automation      |
| Git & GitHub         | Version control and collaboration|

---

## 🚀 How to Use

```bash
# Clone the repository
git clone https://github.com/hemathens/kaggle-projects.git
cd kaggle-projects

# View datasets
cd datasets/
# View notebooks
cd notebooks/
```

---

## Licences

These datasets are published under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** License.

You are free to:

- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material for any purpose

Under the following terms:

- **Attribution** — You must give appropriate credit by linking to [my Kaggle profile](https://www.kaggle.com/hemajitpatel), provide a link to the license, and indicate if changes were made.

Full License: https://creativecommons.org/licenses/by/4.0/
