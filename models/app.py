import gradio as gr
import pandas as pd
import joblib

random_forest = joblib.load("models/forest_model.joblib")

def predict_survival(name, Pclass, Sex, Age, Sibsp, PaCh, Fare):
    input_data = pd.DataFrame({
        "Pclass": [Pclass],
        "Sex": [Sex],
        "Age": [Age],
        "SibSp": [Sibsp],
        "PaCh": [PaCh],
        "Fare": [Fare]
    })
    
    prediction = random_forest.predict(input_data)[0]
    probabilities = random_forest.predict_proba(input_data)[0] # to explain

    if prediction == 0:
        result = f'Wooo 😮 {name.title()}, you Died 😢'
    else:
        result = f'Cheers 🥂 {name.title()}, you Survived 😁'
    # gr.label expectss a dict of {Class name: probability} to display the probabilities of each class
    probs = {'Died':float(probabilities[0]), 'Survived': float(probabilities[1])}    
    return result, probs

# PILLAR @ THE INPUTS(ONE PER FUNCTIPON PARAMETR, SAME ORDER)
inputs = [
    gr.Textbox(label="Name", placeholder="e.g Olanrewaju Esther"),
    gr.Radio([1,2,3], label="Passenger Class(1 = First)", Value=3),
    gr.Radio([("male", 0), ("female", 1)], label="Sex", value=0),
    gr.Slider(0, 80, value=30, step=1, label="Age"),
    gr.Number(label="Siblings/Spouses aboard", value=0, precision=0),
    gr.Number(label="Parents/Children aboard", value=0, precision=0),
    gr.Number(label="Fare Paid ($)", value=30),
]
        
# PILLAR 3 THE OUTPUTS(one per function return value, same order)
outputs = [
    gr.Textbox(label="Survival Prediction"),
    gr.Label(label="Prediction Probabilities"),
]

# Wire the three pillars together
demo = gr.Interface(
    fn=predict_survival,
    inputs=inputs,
    outputs=outputs,
    title="Titanic Survival Prediction",
    description="Enter a passenger's details and see if our random forest thinks they survived the Titanic disaster. This model was trained on the Titanic dataset from Kaggle and uses a random forest classifier to make predictions."
    examples=[
        ["John Doe", 3, 0, 25, 0, 0, 8],
        ["Tomi Jesu", 1, 1, 17, 0, 1, 150],
    ],
)

#if __name__ == "__main__":
    demo.launch()