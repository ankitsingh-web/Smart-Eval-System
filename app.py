from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib
import re

app = FastAPI()

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

topic_resources = {
    "OOP": ["Encapsulation", "Null handling"],
    "Arrays": ["Array indexing practice"],
    "Pointers": ["Memory allocation concepts"],
    "Time Complexity": ["Big-O notation"],
    "Recursion": ["Base condition practice"],
    "Syntax": ["Language syntax rules"],
    "Data Types": ["Primitive vs Non-primitive"],
    "Arithmetic": ["Division & modulo"],
    "Memory Management": ["Heap vs Stack"],
    "Data Structures": ["Dictionary & HashMap"]
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


@app.post("/predict-from-file")
async def predict_from_file(file: UploadFile = File(...)):
    
    content = await file.read()
    error_text = content.decode("utf-8")

    cleaned = clean_text(error_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    return {
        "predicted_topic": prediction,
        "recommended_resources": topic_resources.get(prediction, [])
    }
