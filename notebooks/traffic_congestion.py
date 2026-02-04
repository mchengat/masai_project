import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Traffic Congestion Incident Detection Using Machine Learning

    ## Abstract
    Traffic congestion and unexpected incidents pose significant challenges to urban transportation systems. This project presents a machine learning–based approach to detect traffic congestion and road incidents using features derived from aerial camera feeds such as vehicle speed, density, lane occupancy, and queue length. Multiple classification models were evaluated with a primary focus on maximizing recall to avoid missed incidents while maintaining a balanced F1-score. The final model demonstrates stable performance and practical suitability for real-time traffic monitoring systems.
    """)
    return


@app.cell
def _():
    # ============================================================
    # Traffic Congestion Classification
    # ============================================================

    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    return


@app.cell
def _(mo):
    mo.md(r"""
    ##**1. Problem Formulation**

    The task is formulated as a binary classification problem, where the objective is to detect traffic congestion or incident conditions. The target variable is defined as:
    - 0: Normal traffic conditions
    - 1: Congestion or incident conditions

    Given the operational impact of missed congestion events, the problem is treated as cost-sensitive, with a higher penalty assigned to false negatives than false positives.
    """)
    return


@app.cell
def _():
    # ------------------------ IMPORTS ----------------------------
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn import set_config

    set_config(enable_cython_pairwise_dist=False)
    import seaborn as sns
    from sklearn.model_selection import (
        train_test_split,
        StratifiedKFold,
        RandomizedSearchCV,
        learning_curve,
    )
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        confusion_matrix,
        ConfusionMatrixDisplay,
        f1_score,
        classification_report,
        roc_auc_score,
        roc_curve,
        recall_score,
        precision_score,
    )
    from scipy import stats
    from scipy.stats import randint, loguniform
    return (
        ColumnTransformer,
        ConfusionMatrixDisplay,
        KNeighborsClassifier,
        LogisticRegression,
        Pipeline,
        RandomForestClassifier,
        RandomizedSearchCV,
        RobustScaler,
        SVC,
        StandardScaler,
        StratifiedKFold,
        classification_report,
        confusion_matrix,
        f1_score,
        learning_curve,
        loguniform,
        np,
        pd,
        plt,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
        sns,
        stats,
        train_test_split,
    )


@app.cell
def _(pd):
    # --------------------- LOAD DATA ------------------------------
    df = pd.read_csv("./data/traffic_congestion.csv")

    df["flow_rate"] = df["vehicle_density"] * df["avg_vehicle_speed"]
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, df, y


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Dataset Description

    1. vehicle_density: The number of vehicles per unit area. High density is a primary indicator of congestion.
    2. avg_vehicle_speed: The mean speed of all vehicles. Lower speeds directly correlate with increased congestion levels.
    3. speed_std: Standard deviation of speed. High values indicate "stop-and-go" traffic, which is typical during high-congestion periods.
    4. lane_occupancy: The percentage of time a specific point on the road is occupied by a vehicle. Higher occupancy suggests saturated road capacity.
    5. queue_length: The length of the line of stationary or slow-moving vehicles. Longer queues indicate bottlenecks or incident-related backups.
    6. edge_density: A computer-vision metric representing the intensity of edges in an image; more edges typically signify the presence of more vehicles.
    7. optical_flow_mag: Measures the magnitude of motion between video frames. Low magnitude combined with high density indicates a "gridlock" state.
    8. shadow_fraction: The portion of the scene covered by shadows. While environmental, it can impact sensor accuracy and reflects the time of day.
    9. time_of_day_norm: Normalized time. Traffic patterns are highly cyclic (e.g., morning and evening rush hours).
    10. road_width_norm: Normalized width of the segment. Narrower roads (bottlenecks) have lower capacity and higher congestion risk.


    In traffic congestion and incident detection, the primary goal of the system is to identify abnormal or unsafe traffic conditions as early and accurately as possible. Unlike generic classification tasks, the cost of different types of errors is not equal in this domain. For this reason, Recall and F1-score are more appropriate evaluation metrics than accuracy.

    ### Target Label
    - Binary classification:
      - `0` → Normal traffic
      - `1` → Congestion / incident condition
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Exploratory Data Analysis (EDA)
    """)
    return


@app.cell
def _(df):
    # Inspect the class distribution of target column
    df["label"].value_counts(normalize=True, dropna=False)
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.describe().T
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Above statistics shows -
    1. Dataset size consistent (4000 samples)
       * No missing values in feature measurements

    2. Feature behavior insights

       * Average Vehicle Speed
         Vehicle speed ranges from very low values (~5 km/h) to high free-flow speeds (~90 km/h), with a moderate mean. A strong inverse indicator of congestion

       * Vehicle Density
          Vehicle density shows high variability, with a relatively low median but a large maximum value. This long-tailed distribution suggests that severe congestion cases are present but less frequent. Density is expected to be a primary contributor to congestion classification.

       * Lane Occupancy
         Lane occupancy values are bounded between 0 and 1, indicating they are already normalized. The median occupancy is relatively low, while higher values correspond to congested or incident conditions

       * Queue Length
         Queue length exhibits right-skewed behavior, with most samples having short queues and a smaller number of samples showing very long queues

       * Speed Variability
         Speed standard deviation displays significant spread, with higher values indicating unstable traffic flow
    """)
    return


@app.cell
def _(df, plt, sns):
    # Pairplot: All features vs each other
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("Correlation Matrix")
    plt.show()
    return


@app.cell
def _(df, plt, sns):
    target = "label"
    corr_target = df.corr(numeric_only=True)[target].sort_values(ascending=False)

    plt.figure(figsize=(6, 4))
    sns.barplot(x=corr_target.values, y=corr_target.index)
    plt.title(f"Correlation with target: {target}")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Observations

    **Feature Engineering and Categorization**
    The input features are derived from heterogeneous sources and can be grouped into four categories:
        1. Traffic State Features
            - Vehicle density
            - Average vehicle speed
            - Speed standard deviation
            - Lane occupancy
            - Queue length

            These features directly reflect macroscopic traffic flow characteristics and are expected to provide the strongest signal for congestion detection.

        2. Vision-Based Features
            - Optical flow magnitude
            - Edge density

            These features capture motion and visual clutter from video streams, helping to identify stalled or slow-moving traffic

        3. Environmental and Illumination Features

            - Shadow fraction

            This feature accounts for lighting variations and shadow artifacts that can affect visual feature reliability.

        4. Contextual Normalization Features
            - Time-of-day normalization
            - Road-width normalization

            These features provide contextual information that improves generalization across different road geometries and temporal traffic patterns.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Analyze data distribution
    """)
    return


@app.cell
def _(df, plt, sns, stats):
    features_to_check = [
        "vehicle_density",
        "avg_vehicle_speed",
        "lane_occupancy",
        "queue_length",
    ]

    plt.figure(figsize=(12, 10))

    for i, feature in enumerate(features_to_check):
        # Histogram
        plt.subplot(4, 2, 2 * i + 1)
        sns.histplot(df[feature], kde=True, color="skyblue", alpha=0.6)
        plt.title(f"Distribution of feature {feature}")

        plt.subplot(4, 2, 2 * i + 2)
        stats.probplot(df[feature], dist="norm", plot=plt)
        plt.title(f"Q-Q Plot of {feature}")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(X, train_test_split, y):
    # ------------------ TRAIN / TEST SPLIT ------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    mo.md(r"""
    ##4. Feature Scaling
    The dataset contains features with heterogeneous scales (e.g., speed in km/h, density as counts, occupancy as ratios). Without proper scaling, features with larger numeric ranges may disproportionately influence model training, particularly in distance-based and gradient-based learning algorithms.
    """)
    return


@app.cell
def _(ColumnTransformer, RobustScaler, StandardScaler):
    # ------------------- FEATURE GROUPS ---------------------------
    count_features = ["vehicle_density", "queue_length"]
    continuous_features = ["avg_vehicle_speed", "speed_std"]
    normalized_features = ["lane_occupancy"]

    # ------------------ PREPROCESSING -----------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("robust", RobustScaler(), count_features),
            ("standard", StandardScaler(), continuous_features),
            ("passthrough", "passthrough", normalized_features),
        ]
    )
    return (preprocessor,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Model Selection and Hyperparameter Tuning
    Multiple supervised learning models were evaluated to identify the most suitable approach for traffic congestion incident detection.

    The tuning process focused on:
    - Maximizing recall without severely degrading precision
    - Reducing overfitting and model variance
    - Identifying stable hyperparameter regions
    """)
    return


@app.cell
def _(
    KNeighborsClassifier,
    LogisticRegression,
    Pipeline,
    RandomForestClassifier,
    SVC,
    StratifiedKFold,
    loguniform,
    preprocessor,
):
    # ------------------ STRATIFIED CV -----------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ------------------ MODEL PIPELINES ---------------------------
    knn_pipeline = Pipeline(
        steps=[("preprocessing", preprocessor), ("clf", KNeighborsClassifier())]
    )

    svm_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced")),
        ]
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,  # IMPORTANT on M1
                ),
            ),
        ]
    )

    logreg_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="saga",
                    l1_ratio=0.0,
                    C=1.0,
                ),
            ),
        ]
    )

    # ---------------- PARAMETER DISTRIBUTIONS ---------------------
    knn_param_dist = {
        "clf__n_neighbors": range(1, 50, 2),
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"],
    }

    svm_param_dist = {
        "clf__C": loguniform(1e-2, 1e2),
        "clf__gamma": loguniform(1e-3, 1),
    }

    rf_param_dist = {
        "clf__n_estimators": [150, 200, 300, 400, 500],
        "clf__max_depth": [6, 8, 10, 12, 14],
        "clf__min_samples_leaf": [10, 15, 20, 25],
        "clf__min_samples_split": [20, 30, 40],
        "clf__max_features": ["sqrt", 0.5],
    }

    logreg_param_dist = {
        # "clf__C": loguniform(1e-3, 1e2),
        # "clf__l1_ratio": [0.0],
        "clf__C": [1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0, 5.0, 10.0],
        "clf__solver": ["saga"],
        "clf__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
    }
    return (
        cv,
        knn_param_dist,
        knn_pipeline,
        logreg_param_dist,
        logreg_pipeline,
        rf_param_dist,
        rf_pipeline,
        svm_param_dist,
        svm_pipeline,
    )


@app.cell
def _(learning_curve, np, plt):
    def plot_learning_curve(estimator, X, y, title, cv, scoring="f1", n_jobs=-1):
        metrics = ["f1", "recall"]

        fig, axes = plt.subplots(
            nrows=1, ncols=2, figsize=(14, 5), sharex=True, sharey=False
        )

        for ax, scoring in zip(axes, metrics):
            train_sizes, train_scores, val_scores = learning_curve(
                estimator=estimator,
                X=X,
                y=y,
                train_sizes=np.linspace(0.1, 1.0, 8),
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                shuffle=True,
                random_state=42,
            )

            train_mean = train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            val_mean = val_scores.mean(axis=1)
            val_std = val_scores.std(axis=1)

            ax.plot(train_sizes, train_mean, "o-", label="Training score")
            ax.plot(train_sizes, val_mean, "o-", label="Cross-validation score")

            ax.fill_between(
                train_sizes,
                train_mean - train_std,
                train_mean + train_std,
                alpha=0.2,
            )
            ax.fill_between(
                train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2
            )

            ax.set_title(f"{title} ({scoring.upper()})")
            ax.set_xlabel("Training Set Size")
            ax.set_ylabel(scoring.upper())
            ax.grid(alpha=0.3)
            ax.legend(loc="best")

        plt.tight_layout()
        plt.savefig(f"./data/{title}.png")
        plt.show()
    return (plot_learning_curve,)


@app.cell
def _(RandomizedSearchCV, X_train, cv, pd, y_train):
    # ---------------- RANDOM SEARCH FUNCTION ----------------------
    def run_random_search(pipeline, param_dist, model_name, n_iter):
        print(f"\n===== RandomizedSearchCV: {model_name} =====")

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring={"f1": "f1", "recall": "recall"},
            refit="recall",
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )

        search.fit(X_train, y_train)

        print("Best parameters:", search.best_params_)
        print("Best CV F1 score:", search.best_score_)

        return (
            search.best_estimator_,
            pd.DataFrame(search.cv_results_),
            search.best_params_,
        )
    return (run_random_search,)


@app.cell
def _(
    knn_param_dist,
    knn_pipeline,
    logreg_param_dist,
    logreg_pipeline,
    rf_param_dist,
    rf_pipeline,
    run_random_search,
    svm_param_dist,
    svm_pipeline,
):
    # ---------------- TRAIN MODELS --------------------------------
    best_knn, results_knn, params_knn = run_random_search(
        knn_pipeline, knn_param_dist, "KNN", n_iter=20
    )
    best_svm, results_svm, params_svm = run_random_search(
        svm_pipeline, svm_param_dist, "SVM", n_iter=20
    )
    best_rf, results_rf, params_rf = run_random_search(
        rf_pipeline, rf_param_dist, "Random Forest", n_iter=20
    )
    best_logreg, results_logreg, params_logreg = run_random_search(
        logreg_pipeline, logreg_param_dist, "Logistic Regression", n_iter=20
    )
    return (
        best_knn,
        best_logreg,
        best_rf,
        best_svm,
        params_knn,
        params_logreg,
        params_rf,
        params_svm,
        results_knn,
        results_logreg,
        results_rf,
        results_svm,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Model Evaluation

    Model performance was evaluated using multiple metrics and visualization techniques.

    ### Evaluation Metrics
    - Recall
    - Precision
    - F1-score
    - ROC-AUC

    ### Diagnostic Tools
    - Confusion Matrix
    - ROC Curve
    - Learning Curves (training vs validation)
    """)
    return


@app.cell
def _(
    X_train,
    best_knn,
    best_logreg,
    best_rf,
    best_svm,
    cv,
    plot_learning_curve,
    y_train,
):
    # ================Learning Curves==================#
    plot_learning_curve(
        best_rf, X_train, y_train, title="Learning Curve – Random Forest", cv=cv
    )

    plot_learning_curve(
        best_logreg,
        X_train,
        y_train,
        title="Learning Curve – Logistic Regression",
        cv=cv,
    )

    plot_learning_curve(
        best_svm, X_train, y_train, title="Learning Curve – SVM", cv=cv
    )

    plot_learning_curve(
        best_knn, X_train, y_train, title="Learning Curve – KNN", cv=cv
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Learning curve Analysis

    **Logistic regression**: The learning curves show that logistic regression performs consistently, with similar training and validation F1-score and recall across different training sizes. This indicates that the model generalizes well without overfitting, making it a dependable choice for traffic congestion detection, especially in safety-critical applications.

    **Random Forest**: The learning curves show that the Random Forest model performs very well on the training data but less effectively on the validation data, indicating overfitting. Although adding more training data slightly improves performance, the gap between training and validation results remains. This suggests that improving the model through better tuning and regularization is more effective than collecting additional data.

    **SVM**: The SVM learning curves demonstrate rapid convergence and strong generalization, with training and validation F1-score and recall closely aligned after approximately 600 samples. The model achieves higher recall than logistic regression while maintaining low variance, indicating that it effectively captures non-linear congestion patterns without overfitting.

    **KNN**: is massively overfitting and not suitable as a final model for this problem.
    """)
    return


@app.cell
def _(plt, results_rf):
    # ===============Random Forest==================== #

    mean_scores_by_trees = (
        results_rf.groupby("param_clf__n_estimators")[
            ["mean_test_f1", "mean_test_recall"]
        ]
        .mean()
        .reset_index()
    )


    plt.figure(figsize=(8, 5))
    plt.plot(
        mean_scores_by_trees["param_clf__n_estimators"],
        mean_scores_by_trees["mean_test_recall"],
        marker="o",
        label="Recall",
    )
    plt.plot(
        mean_scores_by_trees["param_clf__n_estimators"],
        mean_scores_by_trees["mean_test_f1"],
        marker="s",
        label="F1-score",
    )

    plt.xlabel("Number of Trees")
    plt.ylabel("CV Score")
    plt.title("Random Forest: Recall & F1 vs Number of Trees")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./data/rf-recall-f1.png")
    plt.show()

    # =================================================== #
    return


@app.cell
def _(plt, results_logreg):
    # ===============Logistic Reg==================== #

    mean_scores_by_log_C = (
        results_logreg.groupby("param_clf__C")[
            ["mean_test_f1", "mean_test_recall"]
        ]
        .mean()
        .reset_index()
        .sort_values("param_clf__C")
    )

    # print(mean_scores_by_C)
    plt.figure(figsize=(8, 5))
    plt.semilogx(
        mean_scores_by_log_C["param_clf__C"],
        mean_scores_by_log_C["mean_test_recall"],
        marker="o",
        label="Recall",
    )
    plt.semilogx(
        mean_scores_by_log_C["param_clf__C"],
        mean_scores_by_log_C["mean_test_f1"],
        marker="s",
        label="F1-score",
    )

    plt.xlabel("Regularization Strength (C)")
    plt.ylabel("CV Score")
    plt.title("Logistic Regression: Recall & F1 vs C")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./data/logreg-recall-f1.png")
    plt.show()
    # =================================================== #
    return


@app.cell
def _(plt, results_svm):
    # ===============SVM==================== #

    mean_scores_by_svm_C = (
        results_svm.groupby("param_clf__C")[["mean_test_f1", "mean_test_recall"]]
        .mean()
        .reset_index()
        .sort_values("param_clf__C")
    )

    # print(mean_scores_by_C)
    plt.figure(figsize=(8, 5))
    plt.semilogx(
        mean_scores_by_svm_C["param_clf__C"],
        mean_scores_by_svm_C["mean_test_recall"],
        marker="o",
        label="Recall",
    )
    plt.semilogx(
        mean_scores_by_svm_C["param_clf__C"],
        mean_scores_by_svm_C["mean_test_f1"],
        marker="s",
        label="F1-score",
    )

    plt.xlabel("Regularization Strength (C)")
    plt.ylabel("CV Score")
    plt.title("SVM: Recall & F1 vs C")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./data/svm-recall-f1.png")
    plt.show()
    # =================================================== #
    return


@app.cell
def _(plt, results_knn):
    # ===============KNN==================== #

    mean_scores_by_k = (
        results_knn.groupby("param_clf__n_neighbors")[
            ["mean_test_f1", "mean_test_recall"]
        ]
        .mean()
        .reset_index()
    )

    # print(mean_scores_by_k)

    plt.figure(figsize=(8, 5))
    plt.semilogx(
        mean_scores_by_k["param_clf__n_neighbors"],
        mean_scores_by_k["mean_test_recall"],
        marker="o",
        label="Recall",
    )
    plt.semilogx(
        mean_scores_by_k["param_clf__n_neighbors"],
        mean_scores_by_k["mean_test_f1"],
        marker="s",
        label="F1-score",
    )

    plt.xlabel("Number of Neighbors")
    plt.ylabel("CV Score")
    plt.title("KNN: Recall & F1 vs C")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./data/knn-recall-f1.png")
    plt.show()
    # =================================================== #
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    X_test,
    best_knn,
    best_logreg,
    best_rf,
    best_svm,
    classification_report,
    confusion_matrix,
    f1_score,
    plt,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    y_test,
):
    # ---------------- EVALUATION FUNCTION -------------------------
    def evaluate_model(model, model_name):
        print(f"\n===== Evaluation: {model_name} =====")

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Recall Score
        recall = recall_score(y_test, y_pred)
        print("Recall Score:", recall)

        # F1 Score
        f1 = f1_score(y_test, y_pred)
        print("F1 Score:", f1)

        # Precision
        precision = precision_score(y_test, y_pred)
        print("Precision:", precision)

        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Create figure with 1 row, 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # -------------------------
        # Confusion Matrix (Left)
        # -------------------------
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes[0], colorbar=False)
        axes[0].set_title(f"{model_name} – Confusion Matrix")

        # -------------------------
        # ROC Curve (Right)
        # -------------------------
        roc_auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        axes[1].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
        axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title(f"{model_name} – ROC Curve")
        axes[1].legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"./data/{model_name}-metrics.png")
        plt.show()

        return {
            "Model": model_name,
            "Recall": recall,
            "F1": f1,
            "Precision": precision,
        }


    # ---------------- FINAL EVALUATION ----------------------------
    results = []
    results.append(evaluate_model(best_logreg, "Logistic Regression"))
    results.append(evaluate_model(best_knn, "KNN"))
    results.append(evaluate_model(best_svm, "SVM"))
    results.append(evaluate_model(best_rf, "Random Forest"))
    return (results,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Findings on Recall & F1 Score

    - KNN: Is highly sensitive to the choice of neighbors and does not achieve competitive recall, making it unsuitable as a final mode
    - SVM : SVM achieves strong and stable recall after tuning, making it one of the most reliable models for congestion detection.
    - Logistic Regression: Logistic Regression is stable, easy to tune, and generalizes well, making it a strong baseline and a safe deployment choice.
    - Random Forest: Random Forest shows diminishing returns with more trees and requires structural regularization rather than more estimators.

    The hyperparameter analysis shows that Logistic Regression and SVM provide stable and reliable performance for traffic congestion detection. Logistic Regression offers strong generalization and simplicity, while SVM achieves the highest recall with consistent performance after tuning. In contrast, KNN suffers from overfitting and instability, and Random Forest shows limited improvement despite increased model complexity. Based on these findings, SVM is selected as the best-performing model, with Logistic Regression serving as a robust baseline.
    """)
    return


@app.cell
def _(pd, results):
    results_df = pd.DataFrame(results)
    best_model_row = results_df.loc[results_df["Recall"].idxmax()]

    print("\n" + "=" * 40)
    print(f"🏆 BEST PIPELINE: {best_model_row['Model']}")
    print(f"   Recall: {best_model_row['Recall']:.2%}")
    print("=" * 40)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Traffic Congestion Heatmap Analysis
    """)
    return


@app.cell
def _(X, best_logreg, df, pd, plt, sns):
    # Add predicted congestion probability
    df_vis = df.copy()
    df_vis["congestion_prob"] = best_logreg.predict_proba(X)[:, 1]

    # Bin traffic state variables
    df_vis["occupancy_bin"] = pd.cut(df_vis["lane_occupancy"], bins=10)
    df_vis["queue_bin"] = pd.cut(df_vis["queue_length"], bins=10)

    # Aggregate congestion probability
    heatmap_data = (
        df_vis.groupby(["occupancy_bin", "queue_bin"])["congestion_prob"]
        .mean()
        .reset_index()
    )

    # Pivot for heatmap
    pivot = heatmap_data.pivot(
        index="occupancy_bin", columns="queue_bin", values="congestion_prob"
    )

    # Plot
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        pivot, cmap="hot", cbar_kws={"label": "Average Congestion Probability"}
    )
    # invert y-axis so small->large goes bottom->top
    ax.invert_yaxis()
    plt.title("Traffic Congestion Heatmap (Lane Occupancy vs Queue Length)")
    plt.xlabel("Queue Length")
    plt.ylabel("Lane Occupancy")
    plt.tight_layout()
    plt.savefig("./data/traffic_congestion_hm1.png")
    plt.show()
    return (df_vis,)


@app.cell
def _(mo):
    mo.md(r"""
    - Dark red / black → Low congestion probability
    - Orange / yellow → Medium congestion probability
    - White → Very high congestion probability (≈ 1.0)

    The highest congestion risk is concentrated in the top-right region of the plot, where lane occupancy is high and queue lengths are long, indicating severe congestion conditions caused by roadway capacity saturation and queue spillbac
    """)
    return


@app.cell
def _(df_vis, pd, plt, sns):
    df_vis["density_bin"] = pd.cut(df_vis["vehicle_density"], bins=10)
    df_vis["speed_bin"] = pd.cut(df_vis["avg_vehicle_speed"], bins=10)

    heatmap_data_1 = (
        df_vis.groupby(["density_bin", "speed_bin"])["congestion_prob"]
        .mean()
        .reset_index()
    )

    pivot_1 = heatmap_data_1.pivot(
        index="density_bin", columns="speed_bin", values="congestion_prob"
    )

    ax2 = sns.heatmap(pivot_1, cmap="hot")
    ax2.invert_yaxis()
    plt.title("Traffic Congestion Heatmap (Density vs Speed)")
    plt.xlabel("Average Speed")
    plt.ylabel("Vehicle Density")
    plt.tight_layout()
    plt.savefig("./data/traffic_congestion_hm2.png")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Above heatmap visualizes the average predicted congestion probability across different combinations of vehicle density (y-axis) and average traffic speed (x-axis). Dark colors represent low congestion, while brighter (yellow to white) regions indicate high congestion risk.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ##8. Probability Threshold Optimization
    """)
    return


@app.cell
def _(
    X_test,
    best_knn,
    best_logreg,
    best_rf,
    best_svm,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    y_test,
):
    # Use the trained logistic regression pipeline
    # (assumes variable name: best_pipeline or logreg_pipeline)

    # Predict probabilities
    y_prob = best_svm.predict_proba(X_test)[:, 1]

    # Apply LOWER threshold (35%)
    threshold = 0.40
    y_pred_thresh = (y_prob >= threshold).astype(int)

    # Evaluate
    print("SVM")
    print(f"Decision Threshold: {threshold}")
    print("Recall:", recall_score(y_test, y_pred_thresh))
    print("Precision:", precision_score(y_test, y_pred_thresh))
    print("F1-score:", f1_score(y_test, y_pred_thresh))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred_thresh))


    y_prob = best_logreg.predict_proba(X_test)[:, 1]

    # Apply LOWER threshold (35%)
    y_pred_thresh = (y_prob >= threshold).astype(int)

    # Evaluate
    print("Logistic Regression")
    print(f"Decision Threshold: {threshold}")
    print("Recall:", recall_score(y_test, y_pred_thresh))
    print("Precision:", precision_score(y_test, y_pred_thresh))
    print("F1-score:", f1_score(y_test, y_pred_thresh))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred_thresh))

    y_prob = best_knn.predict_proba(X_test)[:, 1]

    # Apply LOWER threshold (35%)
    y_pred_thresh = (y_prob >= threshold).astype(int)

    # Evaluate
    print("KNN")
    print(f"Decision Threshold: {threshold}")
    print("Recall:", recall_score(y_test, y_pred_thresh))
    print("Precision:", precision_score(y_test, y_pred_thresh))
    print("F1-score:", f1_score(y_test, y_pred_thresh))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred_thresh))


    y_prob = best_rf.predict_proba(X_test)[:, 1]

    # Apply LOWER threshold (35%)
    y_pred_thresh = (y_prob >= threshold).astype(int)

    # Evaluate
    print("Random Forest")
    print(f"Decision Threshold: {threshold}")
    print("Recall:", recall_score(y_test, y_pred_thresh))
    print("Precision:", precision_score(y_test, y_pred_thresh))
    print("F1-score:", f1_score(y_test, y_pred_thresh))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred_thresh))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Plot decision boundaries for different models
    """)
    return


@app.cell
def _(
    KNeighborsClassifier,
    LogisticRegression,
    RandomForestClassifier,
    RobustScaler,
    SVC,
    StandardScaler,
    df,
    np,
    params_knn,
    params_logreg,
    params_rf,
    params_svm,
    plt,
    sns,
    y,
):
    X_2d = df[["vehicle_density", "avg_vehicle_speed"]]
    # We need to scale these 2 features similarly to how they were scaled in full model
    # In full model: density->Robust, speed->Standard
    # We'll replicate that manually for the 2D pipeline
    from sklearn.base import BaseEstimator, TransformerMixin


    class TwoFeatureScaler(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            self.r_scaler = RobustScaler().fit(X[["vehicle_density"]])
            self.s_scaler = StandardScaler().fit(X[["avg_vehicle_speed"]])
            return self

        def transform(self, X):
            d = self.r_scaler.transform(X[["vehicle_density"]])
            s = self.s_scaler.transform(X[["avg_vehicle_speed"]])
            return np.hstack([d, s])


    # Re-instantiate models with best params (stripping 'clf__' prefix)
    knn_2d = KNeighborsClassifier(
        **{k.replace("clf__", ""): v for k, v in params_knn.items()}
    )
    svm_2d = SVC(
        probability=True,
        class_weight="balanced",
        **{k.replace("clf__", ""): v for k, v in params_svm.items()},
    )
    rf_2d = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        **{k.replace("clf__", ""): v for k, v in params_rf.items()},
    )
    logreg_2d = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        **{k.replace("clf__", ""): v for k, v in params_logreg.items()},
    )

    models_2d = [
        ("KNN", knn_2d),
        ("SVM", svm_2d),
        ("Random Forest", rf_2d),
        ("Logistic Regression", logreg_2d),
    ]

    # Train and Plot
    plt.figure(figsize=(15, 12))
    scaler_2d = TwoFeatureScaler()
    X_2d_scaled = scaler_2d.fit_transform(X_2d)
    y_2d = y.values

    # Meshgrid
    h = 0.02
    x_min, x_max = X_2d_scaled[:, 0].min() - 0.5, X_2d_scaled[:, 0].max() + 0.5
    y_min, y_max = X_2d_scaled[:, 1].min() - 0.5, X_2d_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    for idx1, (name, model) in enumerate(models_2d):
        model.fit(X_2d_scaled, y_2d)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.subplot(2, 2, idx1 + 1)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        sns.scatterplot(
            x=X_2d_scaled[:, 0],
            y=X_2d_scaled[:, 1],
            hue=y_2d,
            palette={0: "blue", 1: "red"},
            alpha=0.5,
            s=20,
            legend=False,
        )
        plt.title(f"{name} (Best Params)")
        plt.xlabel("Density (Scaled)")
        plt.ylabel("Speed (Scaled)")

    plt.tight_layout()
    plt.savefig("./data/best_params_boundaries.png")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Final Model Selection & Conclusion
    ### Performance Summary

    The primary objective was to identify a model that **maximizes Recall** (to ensure no accidents/incidents are missed) while maintaining a **high F1-score** (to minimize false alarms and avoid *alert fatigue* for traffic operators).

    ### Model Comparison

    | Model | Peak Recall | Stability | Recommendation |
    |------|------------|-----------|----------------|
    | **Logistic Regression** | ~84% | High | **Selected Model** |
    | **SVM** | ~86% (at `C = 0.01`) | Low | Secondary / Risky |
    | **Random Forest** | ~83% | Moderate | Overfits Training Data |
    | **KNN** | ~80% | High | Underperforms |

    ---

    ### Detailed Model Evaluation

    #### A. The Selection: Logistic Regression

    Logistic Regression emerged as the most robust candidate.

    - **Convergence**:
      The learning curves for both F1-score and Recall show training and cross-validation scores converging tightly at approximately **2,500 samples**, indicating a **low-variance model** that generalizes well.

    - **Optimal Hyperparameters**:
      Stable performance is observed for `C ≥ 10⁻¹`, making the model less sensitive to small tuning changes.

    ---

    #### B. The SVM Anomaly (High Recall vs. High Variance)

    While SVM achieved a peak Recall of **86% at `C = 0.01`**, it was not selected as the primary model due to:

    - **Extreme Sensitivity**:
      A small change in the regularization parameter (`C`) caused Recall to drop sharply from **86% to 65%**.

    - **Information Gap**:
      At the point of highest Recall, the variance (shown by the shaded region in the learning curve) was at its maximum, indicating **unpredictable performance** across different traffic data subsets.

    ---

    #### C. Overfitting in Random Forest

    Random Forest achieved a competitive Recall of approximately **83%**, but the learning curve revealed a persistent gap between training and validation scores.

    - This behavior indicates **overfitting**, where the model memorizes specific traffic patterns rather than learning general congestion characteristics.

    ---

    ### Conclusion

    For **Traffic Congestion Incident Detection**, **Logistic Regression** is recommended.

    #### Justification

    - **Safety First**:
      Achieves a reliable **84% Recall**, ensuring most critical incidents are detected.

    - **Operational Efficiency**:
      Computationally lightweight, enabling **sub-second inference** on live traffic sensor data.

    - **Consistency**:
      Unlike SVM, it maintains stable performance across different training sizes and hyperparameter se
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 12. Future Work

    Future improvements may include:
    - Incorporating spatial data (GPS / camera locations)
    - Temporal modeling using time-series or deep learning approaches
    - Adaptive thresholding based on traffic conditions
    - Integration with live traffic management systems
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
