# Practical Session 5: Feature Selection — Filter and Wrapper Methods

**Student:** Anes Benrahmoune  
**Academic Level:** 4th Year Student (Artificial Intelligence Specialty)  
**Group:** Group 1  
**Institution:** National Higher School of Advanced Technologies (ENSTA)  
**Academic Year:** 2026–2027  
**Module:** Machine Learning  

---

## 1. Executive Summary and Repository Overview

This repository hosts the complete, rigorous implementation and comparative analysis for **Practical Session 5 (TP5)** on **Feature Selection: Filter and Wrapper Methods**.

High-dimensional feature representations frequently introduce the curse of dimensionality, risk of overfitting, elevated computational latency, and opaque decision boundaries. This study systematically analyzes, implements, and benchmarks filter, wrapper, and embedded feature selection techniques on the classical **Breast Cancer Wisconsin (Diagnostic)** dataset (569 observations, 30 continuous geometrical features, binary diagnosis target).

### Repository Structure
```
tp5/
├── TP5_Solutions.ipynb                      # Fully executed Jupyter Notebook with embedded plots and outputs
├── Work_session_5_Feature_selection.pdf     # Official practical session assignment PDF
├── Work_session_5_Feature_selection.docx    # Official assignment Word document
├── requirements.txt                         # Python environment dependencies
└── README.md                                # Comprehensive academic documentation & benchmark report
```

---

## 2. Experimental Setup and Ground Rules

To ensure strict scientific reproducibility and eliminate data leakage:
- **Fixed Random State:** `random_state=42` throughout all splits, selectors, and model initializations.
- **Stratified Partitioning:** A single 70% training ($n=398$) and 30% testing ($n=171$) split (`train_test_split(..., stratify=y, random_state=42)`), preserving class proportions (62.7% benign, 37.3% malignant).
- **Validation Protocol:** Hyperparameters, subset cardinalities, and feature rankings are determined exclusively on the training partition using 5-fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
- **Leakage Prevention:** All scalers (`StandardScaler`) and selection transformers are encapsulated inside Scikit-learn `Pipeline` objects.
- **Strict Final Testing:** The test set ($n=171$) is held out and evaluated strictly once at the end of Part D.

---

## 3. Part A. Data Inspection and Baseline

### A.1 Dataset Inspection
- **Sample Cardinality:** $N = 569$ samples, $P = 30$ features.
- **Data Integrity:** 0 missing values, 0 duplicate rows.
- **Target Distribution:**
  - Class `1` (Benign): 357 instances (62.74% total; 250 train / 107 test).
  - Class `0` (Malignant): 212 instances (37.26% total; 148 train / 64 test).
- **Dummy Classifier Baseline:**
  A majority-class baseline (`DummyClassifier(strategy="most_frequent")`) yields a test accuracy of:
  $$\text{Accuracy}_{\text{Dummy}} = \frac{107}{171} = 0.6257 \quad (62.57\%)$$

### A.2 Baseline Model (All 30 Features)
A reference `LogisticRegression(max_iter=1000, random_state=42)` trained on standardized data yields:

| Metric | Training Set ($n=398$) | Testing Set ($n=171$) |
| :--- | :--- | :--- |
| **Accuracy** | 0.9874 (98.74%) | **0.9883 (98.83%)** |
| **F1-Score** | 0.9900 (99.00%) | **0.9907 (99.07%)** |
| **Training Execution Time** | — | **~0.0525 seconds** |

### A.3 Correlation Structure and Multicollinearity
Computing pairwise Pearson correlation coefficients ($r$) reveals massive collinear feature clusters:
- **Size Triplet:** `mean radius`, `mean perimeter`, `mean area` exhibit mutual correlations $r > 0.98$ ($r(\text{radius}, \text{perimeter}) = 0.998$).
- **Extreme Triplet:** `worst radius`, `worst perimeter`, `worst area` exhibit $r > 0.97$.
- **Boundary Roughness:** `mean concavity` and `mean concave points` exhibit $r = 0.92$.

**Why this dataset is an ideal candidate for feature selection:**
1. **Geometric & Mathematical Redundancy:** Measurements describe cell nuclei where perimeter $P \approx 2\pi r$ and area $A \approx \pi r^2$. These deterministic physical relationships generate collinearity that inflates coefficient variance in linear models.
2. **High Dimensionality Relative to Sample Size:** 30 features relative to 398 training instances allows uninformative or noisy dimensions to compromise model stability.
3. **Noisy / Irrelevant Measurements:** Fine-scale error measurements (such as `fractal dimension error` or `texture error`) correlate weakly with malignancy ($r < 0.08$), contributing noise rather than discriminative signal.

---

## 4. Part B. Filter Methods

### B.1 Variance Threshold
Applying `VarianceThreshold(threshold)` to the unscaled training data:

| Threshold ($\tau$) | Features Retained | Features Removed | Removed Features |
| :--- | :--- | :--- | :--- |
| **0.0** (Constant) | 30 | 0 | None |
| **0.01** | 14 | 16 | `mean smoothness`, `mean compactness`, `mean concavity`, `mean concave points`, `mean symmetry`, `mean fractal dimension`, and 10 error/worst features |
| **0.10** | 11 | 19 | Above + `mean texture error`, `texture error`, `worst smoothness`, `worst symmetry`, etc. |

#### Critical Analysis:
1. **Meaningless After Standardization:** Standardizing features via $z = \frac{x - \mu}{\sigma}$ sets all empirical feature variances to $\mathrm{Var}(z_j) = 1.0$. Applying a variance threshold after standardization either drops everything ($\tau \ge 1$) or retains everything ($\tau < 1$).
2. **Scale Dependence Flaw on Unscaled Data:** Unscaled feature variance is entirely determined by arbitrary measurement units rather than diagnostic relevance. For example, `mean area` spans $[143, 2501]\text{ mm}^2$ with a huge variance ($\sim 1.2 \times 10^5$), while crucial diagnostic markers like `mean concave points` ($\in [0, 0.2]$) have tiny variances ($\sim 1.5 \times 10^{-3}$). A raw variance cutoff discards critical biological predictors simply because their measurement units are small.

---

### B.2 Correlation-Based Feature Elimination
We implemented `remove_correlated(X, threshold)` which scans collinear pairs ($|r| > \tau$) and drops the feature with the lower absolute correlation with target $y$:

| Correlation Threshold ($\tau$) | Remaining Features | Removed Features | Test Accuracy | Test F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **0.95** | 23 | 7 | **0.9825** | **0.9860** |
| **0.90** | 19 | 11 | 0.9708 | 0.9767 |
| **0.80** | 13 | 17 | 0.9415 | 0.9528 |

#### Is removing redundancy enough to improve accuracy?
- **No.** Removing near-perfect collinearity ($\tau = 0.95$, dropping 7 features) preserves high accuracy ($98.25\%$, just 1 misclassification away from baseline).
- However, aggressive pruning ($\tau = 0.90$ down to $0.80$) drops accuracy from $98.83\%$ down to $94.15\%$. 
- While collinear features contain redundant shared variance, they also contain unique subtle residual variance that refines decision boundaries. Blindly removing correlated features without checking joint predictive power degrades classification accuracy.

---

### B.3 Univariate Scores and SelectKBest
We evaluated two complementary univariate association scores:
1. **ANOVA F-statistic (`f_classif`):** Linear group separation vs within-group variance.
2. **Mutual Information (`mutual_info_classif`):** Non-parametric non-linear shared entropy:
   $$I(X_j; Y) = \sum_{y} \int p(x, y) \log \frac{p(x, y)}{p(x)p(y)} dx$$

- **Ranking Concordance:** Spearman rank correlation between ANOVA F-scores and Mutual Information is **$\rho = 0.9657$**, indicating near-perfect agreement between linear and non-linear association metrics on this dataset.
- Top ranked features for both: `worst perimeter`, `worst radius`, `worst area`, `mean concave points`, `worst concave points`.

#### Cross-Validated SelectKBest & The 1-SE Rule
Evaluating $k \in \{2, 5, 10, 15, 20, 25, 30\}$ with 5-fold cross-validation inside a `Pipeline`:

| Number of Features ($k$) | ANOVA F-Score CV Accuracy ($\pm \text{std}$) | Mutual Information CV Accuracy ($\pm \text{std}$) |
| :--- | :--- | :--- |
| **2** | 0.9347 $\pm$ 0.0148 | 0.9373 $\pm$ 0.0284 |
| **5** | 0.9523 $\pm$ 0.0146 | 0.9548 $\pm$ 0.0151 |
| **10** | 0.9523 $\pm$ 0.0146 | 0.9523 $\pm$ 0.0146 |
| **15** | 0.9598 $\pm$ 0.0166 | 0.9598 $\pm$ 0.0166 |
| **20** | **0.9874 $\pm$ 0.0139** *(Optimal)* | 0.9799 $\pm$ 0.0128 |
| **25** | 0.9824 $\pm$ 0.0189 | **0.9824 $\pm$ 0.0189** *(Optimal)* |
| **30** | 0.9824 $\pm$ 0.0189 | 0.9824 $\pm$ 0.0189 |

- **One Standard Error (1-SE) Rule Application:**
  - For ANOVA F-score: Peak is at $k=20$ (CV = $0.9874 \pm 0.0139$). The 1-SE threshold is $0.9874 - 0.0139 = 0.9735$. The smallest $k$ meeting this threshold is **$k = 20$** ($0.9874$).
  - For Mutual Information: Peak is at $k=25$ (CV = $0.9824 \pm 0.0189$). The 1-SE threshold is $0.9824 - 0.0189 = 0.9635$. At **$k = 20$**, CV = $0.9799 \ge 0.9635$, so $k = 20$ is selected by the 1-SE rule.

#### Why do univariate filters select nearly identical features?
Univariate filters score each feature $X_j$ in complete isolation from all other features. Because `worst radius`, `worst perimeter`, and `worst area` each individually correlate strongly with tumor malignancy, all three receive top scores. Univariate filters cannot condition on already-selected features ($I(X_j; Y \mid X_{-j})$), thus selecting redundant copies of the same underlying signal.

---

## 5. Part C. Wrapper Methods

### C.1 Sequential Forward Selection (SFS)
- Scikit-learn `SequentialFeatureSelector(direction="forward", n_features_to_select=10, cv=5)` selected:
  `['mean radius', 'mean texture', 'mean perimeter', 'mean compactness', 'texture error', 'smoothness error', 'worst radius', 'worst texture', 'worst perimeter', 'worst smoothness']`
- **Manual Implementation Verification:** Our from-scratch implementation matched Scikit-learn **identically (100% agreement)**:
  - Step 1: `worst radius` (CV = 0.9221)
  - Step 2: Added `worst smoothness` (CV = 0.9673)
  - Step 3: Added `worst texture` (CV = 0.9824)
  - Step 4: Added `mean radius` (CV = 0.9849)
  - Step 8: Added `mean perimeter` (CV = 0.9874)
  - Step 10: Added `smoothness error` (CV = 0.9899)

---

### C.2 Sequential Backward Elimination (SBS)
- Scikit-learn `SequentialFeatureSelector(direction="backward", n_features_to_select=10, cv=5)` selected:
  `['texture error', 'area error', 'smoothness error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst smoothness', 'worst concavity', 'worst symmetry']`
- **Manual Implementation Verification:** Our from-scratch implementation matched Scikit-learn **identically (100% agreement)**.
- **Overlap & Divergence:** Forward and Backward selection share **6 out of 10 features** (`worst radius`, `worst texture`, `worst perimeter`, `worst smoothness`, `texture error`, `smoothness error`).
- **Why are they not identical?**
  Forward Selection commits greedily starting from an empty set; once a feature is added early based on low-dimensional conditioning, it cannot be dropped. Backward Elimination begins with the complete set and evaluates features in the presence of all other variables. Due to greedy local optima and multicollinear interaction paths, their trajectories diverge.

---

### C.3 Recursive Feature Elimination (RFE & RFECV)
- **RFE with Logistic Regression (10 features, 0.18s):**
  `['mean area', 'mean concave points', 'area error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst concave points', 'worst symmetry']`
- **RFECV Automatic Tuning:** Identifies an optimal cardinality of **$k^* = 8$ features** (CV = 0.9849):
  `['mean area', 'area error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst concave points']`
- **Comparison Across Estimators:**
  - *Linear SVC:* 8 / 10 features match Logistic Regression because both optimize linear decision boundaries.
  - *Random Forest:* Focuses heavily on high-Gini split features (`mean concave points`, `worst concavity`, `mean concavity`, `mean area`).

#### Mechanistic Difference: RFE vs Backward Elimination
| Dimension | Recursive Feature Elimination (RFE) | Sequential Backward Elimination (SBS) |
| :--- | :--- | :--- |
| **Pruning Criterion** | Internal model coefficient weight magnitude ($|\beta_j|$) | External cross-validation score |
| **Model Fits per Step** | **1 single fit** on active subset | **$p_{\text{remaining}}$ model fits** across all $K$ folds |
| **Computational Complexity** | $O(p)$ fits total ($\approx 0.18\text{ s}$) | $O(K \cdot p^2)$ fits total ($\approx 31.1\text{ s}$) |

---

### C.4 Computational Cost of Wrapper Search

For $p = 30$ initial features and selecting $k = 10$ features with 5-fold cross-validation:
1. **Sequential Forward Selection (SFS):**
   $$\text{Total Fits} = 5 \times \sum_{i=0}^{9} (30 - i) = 5 \times (30 + 29 + \dots + 21) = 5 \times 255 = \mathbf{1,275 \text{ models}}$$
   At $\sim 0.024\text{ s}$ per fit, theoretical runtime is $\approx 30.6\text{ seconds}$, exactly validating our measured runtime ($30.19\text{ s}$).
2. **Exhaustive Best Subset Search ($k=10$):**
   $$\text{Total Fits} = 5 \times \binom{30}{10} = 5 \times 30,045,015 = \mathbf{150,225,075 \text{ models}} \approx \mathbf{34.8 \text{ days}}$$
3. **Exhaustive Search Over All Subsets ($2^{30}-1$):**
   $$\text{Total Fits} = 5 \times (2^{30} - 1) \approx \mathbf{5.37 \times 10^9 \text{ models}} \approx \mathbf{3.4 \text{ years}}$$

---

## 6. Part D. Noise Injection, Data Leakage, and Final Comparison

### D.1 Noise Injection (20 Gaussian Noise Features Appended, Total $P=50$)
We evaluated how methods behave when choosing 10 features from 50 candidates (30 real + 20 pure noise $\sim \mathcal{N}(0, 1)$):

| Method Family | Total Selected | Noise Features Selected | Noise % | Identified Noise Features |
| :--- | :--- | :--- | :--- | :--- |
| **SelectKBest (ANOVA F-score)** | 10 | **0** | **0%** | None |
| **Forward Selection (SFS)** | 10 | **0** | **0%** | None |
| **Backward Elimination (SBS)** | 10 | **5** | **50%** | `noise_06`, `noise_13`, `noise_14`, `noise_16`, `noise_17` |
| **RFE (Logistic Regression)** | 10 | **0** | **0%** | None |

#### Key Finding:
- **SelectKBest, Forward Selection, and RFE** are completely immune to noise because genuine biological features have massive margins ($F > 100$).
- **Backward Elimination was catastrophically fooled (50% noise).** Starting with all 50 features in a moderately sized training set ($n=398$), random noise features act as spurious regularizers or "noise cancellers" fitting idiosyncratic fold residuals. Removing true collinear features drops fold scores more than dropping noise, trapping backward elimination in a corrupted subset.

---

### D.2 Data Leakage in Feature Selection
- **Incorrect (Selection prior to CV):** CV Accuracy = **0.9523 $\pm$ 0.0146**
- **Correct (Selection inside Pipeline CV):** CV Accuracy = **0.9523 $\pm$ 0.0146**
- **Why are they equal here?** On this specific dataset, the top 10 features have such dominant biological associations that the exact same 10 features are selected across every single fold.
- **Why is leakage disastrous in high dimensions?** When $p \gg n$ (e.g. $p=20,000$, $n=50$ in microarrays), purely random noise features will correlate with the training labels by chance. If selection occurs outside CV, the model evaluates on validation folds whose labels were already used to select features. Leaky CV reports $\sim 95\%$ accuracy when the true generalization is random guessing ($50\%$). Nesting selection in a `Pipeline` prevents this completely.

---

### D.3 Final Benchmark Comparison on the Unseen Test Set ($n=171$)

| Method | Number of Features | Test Accuracy | Test F1-Score | Selection Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (all features)** | 30 | **0.9883** | **0.9907** | — |
| **Filter: correlation (B.2, $\tau=0.90$)** | 19 | 0.9708 | 0.9767 | 0.0051 s |
| **Filter: SelectKBest, F-score (1-SE $k=20$)** | 20 | **0.9766** | **0.9813** | 0.0249 s |
| **Filter: SelectKBest, mutual info (1-SE $k=20$)** | 20 | 0.9708 | 0.9770 | 0.3049 s |
| **Wrapper: Forward Selection ($k=10$)** | 10 | 0.9357 | 0.9484 | 30.1941 s |
| **Wrapper: Backward Elimination ($k=10$)** | 10 | 0.9591 | 0.9677 | 31.1243 s |
| **Wrapper: RFE ($k=10$)** | 10 | **0.9708** | **0.9767** | **0.1814 s** |
| **Wrapper: RFECV (optimal $k^*$)** | **8** | 0.9591 | 0.9671 | 1.5193 s |

---

### Feature Presence Matrix and Consensus
- **Universal Consensus (Selected by ALL 6 Methods):**  
  `{'worst radius', 'worst perimeter'}`
- **Majority Consensus ($\ge 4$ Methods):**  
  `worst radius`, `worst perimeter`, `worst area`, `mean area`, `worst concave points`, `worst texture`, `worst smoothness`.

This confirms the physical structure of breast cancer diagnosis: tumor macro-size (`worst radius`/`perimeter`) provides the primary discriminant, complemented by boundary irregularity (`worst concave points`) and tissue texture (`worst texture`).

---

## 7. Final Discussion Questions and Answers

### Q1. Which method gives the best test accuracy with the fewest features? Is the difference with the baseline significant given the test set size (171 patients)?
- **Answer:** **RFE (Logistic Regression)** achieves the best trade-off, reaching **97.08% test accuracy** (166 / 171 correct) and **0.9767 F1-score** using only **10 features** (a 66.7% reduction in dimensionality).
- **Statistical Significance:** The baseline model correctly classifies 169 / 171 samples (98.83%), representing a difference of only **3 misclassified patients** ($\Delta = 1.75\%$). A two-tailed McNemar test on paired predictions yields $p = 0.25 > 0.05$. Therefore, the performance difference is **statistically insignificant**, proving that 10 features capture the complete diagnostic signal.

### Q2. What are the advantages and limits of filter methods compared with wrapper methods (speed, redundancy, interaction, model dependence)?
- **Filter Methods:**
  - *Advantages:* Ultrafast ($O(p)$ runtime, $\sim 0.02\text{ s}$), model-agnostic, immune to model-specific overfitting.
  - *Limits:* Cannot detect redundancy (retains collinear duplicates) and completely blind to multi-feature interactions.
- **Wrapper Methods:**
  - *Advantages:* Optimizes directly for the downstream model's inductive bias, captures multi-feature synergies, eliminates redundant variables.
  - *Limits:* Computationally expensive ($O(K \cdot p^2)$, $\sim 30\text{ s}$), prone to overfitting in high dimensions, and features selected are tied to one specific classifier.

### Q3. Why are wrapper methods more expensive? Give the number of models trained by Forward Selection for $p$ features and $k$ selected features.
- **Answer:** Wrapper methods evaluate candidate feature sets by repeatedly training and cross-validating the classifier. For $p$ initial features and $k$ selected features with $K$-fold cross-validation, Forward Selection fits:
  $$\text{Total Models} = K \sum_{i=0}^{k-1} (p - i) = K \left[ k \cdot p - \frac{k(k-1)}{2} \right]$$
  For $p = 30$, $k = 10$, $K = 5$, this equals **1,275 model fits**, compared to 0 model fits for filter methods.

### Q4. Forward Selection and Backward Elimination are greedy. Do they guarantee the optimal subset? Justify using your results.
- **Answer:** **No, neither algorithm guarantees finding the globally optimal feature subset.**
- Both algorithms make irrevocable greedy choices at each step. Forward selection cannot discard an unhelpful feature added early; backward elimination cannot recover a dropped feature that becomes valuable when paired with others.
- *Empirical Proof:* Forward and Backward selection converged to different 10-feature subsets with differing test accuracies ($93.57\%$ vs $95.91\%$) and only 6 overlapping features. Only an exhaustive combinatorial search guarantees global optimality.

### Q5. How did the methods behave in front of the 20 noise features, and what does it tell you about each family?
- **Answer:** SelectKBest (F-score), Forward Selection, and RFE demonstrated **100% noise rejection** (0 / 10 noise features selected), confirming their resilience when genuine features possess high signal-to-noise ratios.
- In contrast, **Backward Elimination selected 5 noise features out of 10 (50% corruption)**. In high dimensions, random noise can act as spurious regularizers fitting fold-specific noise. When backward elimination attempts to prune, genuine collinear features cause larger score drops than inert noise variables, trapping the algorithm in an overfitted subset.

### Q6. Why must feature selection be performed inside the cross-validation loop, and how does a Pipeline prevent leakage?
- **Answer:** Feature selection relies on the target labels $y$. If selection is conducted across the entire dataset before cross-validation, information from the validation folds influences feature ranking.
- This creates **selection bias / data leakage**, artificially inflating cross-validation performance. In high-dimensional regimes ($p \gg n$), leaky CV can report $\sim 95\%$ accuracy on pure noise data.
- Enclosing the scaler, selector, and classifier in a Scikit-learn `Pipeline` guarantees that feature selection is fitted strictly on the $K-1$ training folds during each iteration, keeping the validation fold strictly unseen.

### Q7. In which situation would you choose each family of methods (thousands of features, small dataset, need for interpretability, limited computing time)?
1. **Thousands of Features ($p \gg n$, e.g., Genomics / Text):** Filter methods (ANOVA F-score, Mutual Information) or two-stage pipelines (Filter down to hundreds $\to$ Regularized embedded model). Wrappers are computationally intractable.
2. **Small Dataset ($n$ small, $p$ moderate):** Embedded methods (Lasso L1 penalty) or RFE with strong regularization. Backward elimination must be avoided due to overfitting.
3. **Need for High Interpretability / Clinical Trust:** Wrapper methods (Forward Selection / RFE) or Correlation-pruned Filters. They produce minimal, non-redundant feature subsets that domain specialists can inspect and validate.
4. **Limited Computing Time / Production Constraints:** Filter methods (Mutual Information or Variance/Correlation pruning). They execute in milliseconds and can process streaming data pipelines instantaneously.

---

## 8. Reproduction and Setup Instructions

To reproduce all experiments, figures, and benchmark tables locally:

```bash
# 1. Clone the repository
git clone https://github.com/BENRAHMOUNEANES/machine-learning-tp5-anes-benrahmoune.git
cd machine-learning-tp5-anes-benrahmoune

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the executed Jupyter Notebook
jupyter notebook TP5_Solutions.ipynb
```
