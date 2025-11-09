# %%
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score

import joblib

# %%
data = pd.read_csv("data/california_housing_train.csv")

# %%
data.info()

# %%
y = data.pop("median_house_value")
X = data

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
rf = RandomForestRegressor()

# %%
rf.fit(X_train, y_train)

# %%
scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
scores 

# %%
metric_df = pd.DataFrame({'split_id': list(range(1, len(scores)+1)), "score": [round(score, 2) for score in scores]})
metric_df

# %%
metric_df.to_csv("training_metrics.csv")

# %%
preds = rf.predict(X_test)

# %%
acc_score = r2_score(preds, y_test)
print("Accuracy Score:", acc_score)

# %%
!mkdir artifacts

# %%
joblib.dump(rf, 'artifacts/model.joblib')

# %%



