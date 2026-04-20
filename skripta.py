# %% [markdown]
# # Ucitavanje i eksplorativna analiza podataka

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from imblearn.pipeline import Pipeline

from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import RandomForestClassifier

# %%
data = pd.read_csv('online_shoppers_intention.csv')
data['Revenue'] = data['Revenue'].map({True: 1, False: 0})

X = data.drop(['Revenue'], axis=1)
# y = data['Revenue'].map({True: 1, False: 0})
y = data['Revenue']

# %%
data.head(5)

# %%
data.info()

# %%
data.isnull().sum()

# %%
cat_cols = ['OperatingSystems',
 'Browser',
 'Region',
 'TrafficType',
 'VisitorType',
 'Weekend',
 'Month',
]

num_cols = ['Administrative',
 'Administrative_Duration',
 'Informational',
 'Informational_Duration',
 'ProductRelated',
 'ProductRelated_Duration',
 'BounceRates',
 'ExitRates',
 'PageValues',
 'SpecialDay',
 'Revenue'
]

# %%
data[num_cols].describe().T

# %%
data[cat_cols].nunique()

# %%
plt.figure()
data['Revenue'].value_counts().plot(
    kind='bar'
)
plt.title('Raspodela uzoraka po klasama')
plt.xlabel('Klasa')
plt.ylabel('Broj uzoraka')
plt.xticks([0, 1], ['Negativna', 'Pozitivna'], rotation=0)
plt.show()

data['Revenue'].value_counts(normalize=True)

# %%
grouped = data.groupby('Revenue')[num_cols]

means = grouped.mean()
stds = grouped.std()
    
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    if col == 'Revenue':
        continue
    means[col].plot(kind='bar', yerr=stds[col], ax=axes[i], color='steelblue', edgecolor='black')
    axes[i].set_title(col)
    axes[i].set_xlabel('Klasa')
    axes[i].set_ylabel(col)
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(['Neg', 'Poz'], rotation=0)

plt.suptitle('Srednje vrednosti numeričkih prediktora po klasama', fontsize=20)
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    if col == 'Revenue':
        continue
    data[data['Revenue'] == 0][col].hist(alpha=0.5, label='Negative', ax=axes[i], color='steelblue')
    data[data['Revenue'] == 1][col].hist(alpha=0.5, label='Positive', ax=axes[i], color='tomato')
    axes[i].set_title(f'{col} Distribution by Class')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].legend()

plt.suptitle('Distribucije numeričkih prediktora po klasama', fontsize=20)
plt.tight_layout()
plt.show()

# %%
corr = data[num_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True)
plt.title('Korelacije numeričkih prediktora i ciljne promenljive')
plt.show()

# %% [markdown]
# # Treniranje modela i rezultati

# %%
cat_cols = ['OperatingSystems',
 'Browser',
 'Region',
 'TrafficType',
 'VisitorType',
 'Month'
]

num_cols = ['Administrative',
 'Administrative_Duration',
 'Informational',
 'Informational_Duration',
 'ProductRelated',
 'ProductRelated_Duration',
 'BounceRates',
 'ExitRates',
 'PageValues',
 'SpecialDay',
 'Weekend'
]

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

X_processed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42, stratify=y)

# %%
sss_dummy = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx_d, val_idx_d = next(sss_dummy.split(X, y))

X_train_d = X.iloc[train_idx_d]
y_train_d = y.iloc[train_idx_d]
X_val_d = y.iloc[val_idx_d]
y_val_d = y.iloc[val_idx_d]

dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train_d, y_train_d)

y_pred = dummy.predict(X_val_d)
print("Accuracy:", accuracy_score(y_val_d, y_pred))
print(classification_report(y_val_d, y_pred))

# %%

# 1) Grid Search za SVM hiperparametre

# gss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
# train_idx_gs, val_idx_gs = next(gss.split(X, y))
# X_train_gs = X.iloc[train_idx_gs]
# y_train_gs = y.iloc[train_idx_gs]
# X_val_gs = X.iloc[val_idx_gs]
# y_val_gs = y.iloc[val_idx_gs]

# preprocessor_gs = ColumnTransformer(transformers=[
#     ("num", StandardScaler(), num_cols),
#     ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
# ])
# X_train_gs_proc = preprocessor_gs.fit_transform(X_train_gs)
# X_val_gs_proc = preprocessor_gs.transform(X_val_gs)

# # Grid search za Linear SVM (samo C)
# param_grid_linear = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
# gs_linear = GridSearchCV(
#     SVC(kernel="linear", random_state=42),
#     param_grid_linear,
#     scoring="f1",
#     cv=5,
#     n_jobs=-1
# )
# gs_linear.fit(X_train_gs_proc, y_train_gs)
# best_C_linear = gs_linear.best_params_["C"]
# print(f"SVM Linear best C: {best_C_linear}")

# # Grid search za RBF SVM (C i gamma)
# param_grid_rbf = {
#     "C": [0.001, 0.01, 0.1, 1, 10, 100],
#     "gamma": [0.001, 0.01, 0.1, 1, 10, 100]
# }
# gs_rbf = GridSearchCV(
#     SVC(kernel="rbf", random_state=42),
#     param_grid_rbf,
#     scoring="f1",
#     cv=5,
#     n_jobs=-1
# )
# gs_rbf.fit(X_train_gs_proc, y_train_gs)
# best_C_rbf = gs_rbf.best_params_["C"]
# best_gamma = gs_rbf.best_params_["gamma"]
# print(f"SVM RBF best C: {best_C_rbf}, best gamma: {best_gamma}")

best_C_linear = 1
best_C_rbf = 100
best_gamma = 0.01

# 2) Definisanje modela 

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "SVM Linear": SVC(
        kernel="linear",
        C=best_C_linear,
        random_state=42
    ),
    "SVM RBF": SVC(
        kernel="rbf",
        C=best_C_rbf,
        gamma=best_gamma,
        random_state=42
    )
}

modes = {
    "No OS":   None,
    "With OS": RandomOverSampler
}

# 3) 100 ponavljanja

N_SPLITS = 100

sss = StratifiedShuffleSplit(
    n_splits=N_SPLITS,
    test_size=0.3,
    random_state=42
)

results = {}
for mode_name in modes:
    for model_name in models:
        results[f"{model_name} | {mode_name}"] = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": []
        }

for split_id, (train_idx, val_idx) in enumerate(sss.split(X, y)):
    
    if (split_id + 1) % 10 == 0:
        print(f"  Split {split_id + 1}/{N_SPLITS}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    for mode_name, sampler_class in modes.items():
        for model_name, model in models.items():
            key = f"{model_name} | {mode_name}"

            steps = []
            if sampler_class is not None:
                steps.append(("oversample", sampler_class(random_state=42)))
            
            steps.append(("preprocessing", preprocessor))
            steps.append(("model", model))

            pipeline = Pipeline(steps)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)

            results[key]["accuracy"].append(accuracy_score(y_val, y_pred))
            results[key]["precision"].append(precision_score(y_val, y_pred, zero_division=0))
            results[key]["recall"].append(recall_score(y_val, y_pred, zero_division=0))
            if sampler_class is not None:
                results[key]["f1"].append(f1_score(y_val, y_pred, zero_division=0, average='macro'))
            else:
                results[key]["f1"].append(f1_score(y_val, y_pred, zero_division=0))
            

# 4) Rezultati

summary = []
for key, metrics in results.items():
    model_part, mode_part = key.split(" | ")
    row = {
        "Model": model_part,
        "Mode": mode_part,
        "Accuracy": round(np.mean(metrics["accuracy"])*100, 2),
        "Precision": round(np.mean(metrics["precision"]), 4),
        "Recall": round(np.mean(metrics["recall"]), 4),
        "F1 Score": round(np.mean(metrics["f1"]), 4)
    }
    summary.append(row)

summary_df = pd.DataFrame(summary)

print("\nRezultati BEZ oversampla:")
print(summary_df[summary_df["Mode"] == "No OS"].to_string(index=False))

print("\nRezultati SA oversampleom")
print(summary_df[summary_df["Mode"] == "With OS"].to_string(index=False))

summary_df