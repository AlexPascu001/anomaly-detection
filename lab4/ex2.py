from scipy.io import loadmat
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
import numpy as np

# Load the dataset
data = loadmat('cardio.mat')
X = data['X']
y = data['y'].ravel()  # Labels are in pyod format (0: inlier, 1: outlier)

contamination_rate = 0.096  # From dataset description

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=112)

param_grid = [
    {
        'ocsvm__nu': np.linspace(0.01, 0.1, 10),
        'ocsvm__gamma': ['scale', 'auto'],
        'ocsvm__kernel': ['rbf', 'poly', 'sigmoid']
    },
    {
        'ocsvm__nu': np.linspace(0.01, 0.1, 10),
        'ocsvm__kernel': ['linear']
    }
]

pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('ocsvm', OneClassSVM())
])


def custom_balanced_accuracy(estimator, X, y):
    y_pred = estimator.predict(X)

    y_sklearn = y.copy()
    y_sklearn[y == 0] = 1
    y_sklearn[y == 1] = -1

    return balanced_accuracy_score(y_sklearn, y_pred)


grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring=custom_balanced_accuracy
)
grid_search.fit(X_train, y_train)  # Use pyod format (0, 1) for true labels

best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)


y_test_sklearn = y_test.copy()
y_test_sklearn[y_test == 0] = 1
y_test_sklearn[y_test == 1] = -1


balanced_accuracy = balanced_accuracy_score(y_test_sklearn, y_pred)
print(f"Balanced Accuracy on Test Set: {balanced_accuracy}")
