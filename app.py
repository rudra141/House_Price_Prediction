from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

model = pickle.load(open('House.pkl', 'rb'))
df = pd.read_csv('Housing.csv')

# Define categorical features
categorical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        area = df['area'].unique()
        bedrooms = df['bedrooms'].unique()
        bathrooms = df['bathrooms'].unique()
        stories = df['stories'].unique()
        mainroad = sorted(df['mainroad'].unique())
        guestroom = sorted(df['guestroom'].unique())
        basement = sorted(df['basement'].unique())
        hotwaterheating = sorted(df['hotwaterheating'].unique())
        airconditioning = sorted(df['airconditioning'].unique())
        parking = df['parking'].unique()
        prefarea = sorted(df['prefarea'].unique())
        furnishingstatus = sorted(df['furnishingstatus'].unique())

        return render_template('index.html',
                               area=area,
                               bedrooms=bedrooms,
                               bathrooms=bathrooms,
                               stories=stories,
                               mainroad=mainroad,
                               guestroom=guestroom,
                               basement=basement,
                               hotwaterheating=hotwaterheating,
                               airconditioning=airconditioning,
                               parking=parking,
                               prefarea=prefarea,
                               furnishingstatus=furnishingstatus)
    except KeyError as e:
        error_message = f"Error: Column '{e.args[0]}' not found in the DataFrame."
        return error_message


@app.route('/predict', methods=['POST'])
def predict():
    area = request.form['area']
    bedrooms = request.form['bedrooms']
    bathrooms = request.form['bathrooms']
    stories = request.form['stories']
    mainroad = request.form['mainroad']
    guestroom = request.form['guestroom']
    basement = request.form['basement']
    hotwaterheating = request.form['hotwaterheating']
    airconditioning = request.form['airconditioning']
    parking = request.form['parking']
    prefarea = request.form['prefarea']
    furnishingstatus = request.form['furnishingstatus']

    prediction = model.predict(pd.DataFrame(columns=['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
                                                     'basement', 'hotwaterheating', 'airconditioning', 'parking',
                                                     'prefarea', 'furnishingstatus'],
                                            data=np.array([area, bedrooms, bathrooms, stories, mainroad, guestroom,
                                                           basement, hotwaterheating, airconditioning, parking,
                                                           prefarea, furnishingstatus]).reshape(1, 12)))

    # Convert prediction to string
    prediction_str = str(prediction[0])
    ## Render template with prediction result
    return render_template('index.html', prediction_result=prediction_str, categorical_features=categorical_features,
                           **get_dropdown_options())


def get_dropdown_options():
    # Get unique values for dropdown options
    dropdown_options = {}
    for feature in categorical_features:
        dropdown_options[feature] = df[feature].unique().tolist()
    return dropdown_options


if __name__ == '__main__':
    app.run()
