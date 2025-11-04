import lightgbm as lgb
import numpy as np
import pandas as pd

COLUMNS = ["District", "Property_Type", "Old/New",
           "Duration", "TOTAL_FLOOR_AREA", "NUMBER_HABITABLE_ROOMS"]

CATEGORICAL_FEATURES = [0, 1, 2, 3, 5]


def load_model():
    model = lgb.Booster(model_file='model.txt')
    return model


def predict(model, input_data):
    df = pd.DataFrame([input_data], columns=COLUMNS)

    df['Duration'] = df['Duration'].map({
        'Leasehold': 'L',
        'Freehold': 'F'
    })
    df['Property_Type'] = df['Property_Type'].apply(lambda x: x[0])

    for i, f_idx in enumerate(CATEGORICAL_FEATURES):
        f = COLUMNS[f_idx]
        df[f] = df[f].astype('category').cat.set_categories(
            model.pandas_categorical[i])

    prediction = model.predict(
        df, categorical_feature=CATEGORICAL_FEATURES)
    return np.exp(prediction)[0]
