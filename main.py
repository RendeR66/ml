from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from MLCore import model, load_and_preprocess_data, MODEL_PATH

app = FastAPI()

class RouteQuery(BaseModel):
    date_from: str
    route_from: str
    route_to: str

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "DateSet.xlsx")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/api/predict")
async def predict_route(query: RouteQuery):
    try:
        df = load_and_preprocess_data(file_path)

        input_data = {
            "date_from": pd.to_datetime(query.date_from, format="%d %m %Y").timestamp(),
            "route_from": query.route_from,
            "route_to": query.route_to
        }
        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df, columns=['route_from', 'route_to'])

        for col in df['route_from'].unique():
            if f'route_from_{col}' not in input_df:
                input_df[f'route_from_{col}'] = 0
        for col in df['route_to'].unique():
            if f'route_to_{col}' not in input_df:
                input_df[f'route_to_{col}'] = 0

        input_df['date_from'] = input_df['date_from'].astype(int) / 10**9  # Convert to seconds
        input_df = input_df.reindex(columns=model.input_shape[1:], fill_value=0)

        scaler = StandardScaler()
        input_df = scaler.fit_transform(input_df)

        prediction = model.predict(input_df)

        return {
            "predicted_cost": prediction[0][0],
            "predicted_duration": prediction[0][1]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/retrain")
async def retrain_model():
    try:
        df = load_and_preprocess_data(file_path)

        X = df[['date_from', 'route_from', 'route_to']].copy()
        y = df[['cost', 'duration']]

        X['date_from'] = X['date_from'].astype('int64') / 10**9
        X = pd.get_dummies(X, columns=['route_from', 'route_to'])

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        model.fit(X, y, epochs=10, batch_size=32)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)

        return {"message": "Model retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
