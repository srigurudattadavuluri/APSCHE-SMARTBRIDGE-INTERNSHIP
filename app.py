from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the saved artifacts
saved = joblib.load("credit_card_model.pkl")

model = saved['model']
encoder = saved['encoder']
scaler = saved['scaler']

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Capture Categorical Inputs
        gender = request.form['gender']
        own_car = request.form['own_car']
        own_realty = request.form['own_realty']
        income_type = request.form['income_type']
        education = request.form['education']
        family_status = request.form['family_status']
        housing_type = request.form['housing_type']

        # 2. Capture Numerical Inputs
        cnt_children = float(request.form['cnt_children'])
        income = float(request.form['income'])
        work_phone = float(request.form['work_phone'])
        phone = float(request.form['phone'])
        email = float(request.form['email'])
        family_members = float(request.form['family_members'])
        first_month = float(request.form['first_month'])
        age_years = float(request.form['age_years'])
        years_employed = float(request.form['years_employed'])

        # --- APPLY PREPROCESSING FROM NOTEBOOK ---
        if cnt_children > 4:
            cnt_children = 3

        # 3. Create Categorical DataFrame
        cat_df = pd.DataFrame([[
            gender, own_car, own_realty, income_type, education, family_status, housing_type
        ]], columns=[
            'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE',
            'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE'
        ])

        # Transform categorical features
        encoded_array = encoder.transform(cat_df)
        encoded_cols = encoder.get_feature_names_out(cat_df.columns)
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols)

        # 4. Create Numerical DataFrame
        num_cols = [
            'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'FLAG_WORK_PHONE', 'FLAG_PHONE',
            'FLAG_EMAIL', 'CNT_FAM_MEMBERS', 'first_month', 'AGE_YEARS', 'YEARS_EMPLOYED'
        ]
        
        num_df = pd.DataFrame([[
            cnt_children, income, work_phone, phone, email, family_members, 
            first_month, age_years, years_employed
        ]], columns=num_cols)

        # 5. Combine DataFrames BEFORE scaling 
        final_input = pd.concat([num_df, encoded_df], axis=1)

        # --- THE FIX FOR THE 'STATUS' ERROR ---
        # Find any columns the scaler expects that are missing from our web form
        missing_cols = set(scaler.feature_names_in_) - set(final_input.columns)
        for col in missing_cols:
            final_input[col] = 0  # Fill with a dummy zero so the scaler doesn't crash

        # Ensure the column order exactly matches the order seen during training
        final_input = final_input[scaler.feature_names_in_]

        # 6. Transform using the scaler
        scaled_array = scaler.transform(final_input)
        scaled_df = pd.DataFrame(scaled_array, columns=scaler.feature_names_in_)

        # --- PREVENT DATA LEAKAGE TO THE MODEL ---
        # The model should ONLY receive the columns it was trained on.
        # We strip away the dummy 'STATUS' column before predicting.
        if hasattr(model, 'feature_names_in_'):
            scaled_df = scaled_df[model.feature_names_in_]
        elif 'STATUS' in scaled_df.columns:
            scaled_df = scaled_df.drop(columns=['STATUS'])

        # 7. Predict
        prediction = model.predict(scaled_df)

        if prediction[0] == 1:
            result = "Credit Card Rejected"
        else:
            result = "Credit Card APPROVED"

        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)