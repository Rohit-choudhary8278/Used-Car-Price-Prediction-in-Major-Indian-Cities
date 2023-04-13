import streamlit as st
import pandas as pd
import pickle
import base64
# Load trained model and label encoders
with open(r'C:\Users\User\Downloads\predictive_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(r"C:\Users\User\Downloads\predictive_label.pkl", 'rb') as f:
    label_encoders = pickle.load(f)

# Define numerical and categorical column names
num_col = ['age', 'Year', 'Kilometers_Driven', 'Engine CC', 'Power', 'Seats', 'Mileage Km/L']
cat_cols = ['Manufacturer', 'Location', 'Fuel_Type', 'Transmission', 'Owner_Type']
def preprocess_and_predict_price(num_col, cat_cols, input_data, le, model):
    """
    Preprocesses input data and predicts car price using trained model.
    
    Args:
    - num_col (list): List of column names containing numerical features.
    - cat_cols (list): List of column names containing categorical features.
    - input_data (DataFrame): Input data containing both numerical and categorical features.
    - label_encoder (LabelEncoder): Trained LabelEncoder object.
    - model: Trained machine learning model.
    
    Returns:
    - predicted_price (float): Predicted car price.
    """
    
    # Preprocess numerical features
    num_data = input_data[num_col]
    # Preprocess categorical features
    #cat_data = input_data[cat_cols].apply(lambda x: le.transform(x))
    # Concatenate numerical and categorical features
    num_data['age']=2023-num_data['Year']
    cat_data = input_data[cat_cols]
    cat_data = cat_data.apply(lambda x: label_encoders[x.name].transform(x))
    cat_data = pd.DataFrame(cat_data, columns=cat_cols)
    final_cols=['Manufacturer','age','Year','Location','Kilometers_Driven','Fuel_Type','Transmission','Owner_Type','Engine CC','Power','Seats','Mileage Km/L']
    input_data_processed = pd.concat([cat_data,num_data], axis=1)
    input_data_processed= input_data_processed[final_cols]
    # Make predictions
    predicted_price = model.predict(input_data_processed)
    return predicted_price[0]
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    .content {{
            background-color: white;
            padding: 20px;
            border-radius: 10px;
        }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Add background image
add_bg_from_local(r"C:\Users\User\Downloads\background1.png")
# Define Streamlit UI
def main():
    st.title("Used Car Price Prediction")
    st.write("Enter the car details below:")

    # Input fields for numerical features
    year = st.number_input("Year of Manufacture", min_value=2000, max_value=2023, step=1)
    kilometers_driven = st.number_input("Kilometers Driven", min_value=0)
    engine_cc = st.number_input("Engine Displacement (CC)", min_value=0)
    power = st.number_input("Power (bhp)", min_value=0)
    seats = st.number_input("Number of Seats", min_value=0)
    mileage = st.number_input("Mileage (km/l)", min_value=0)

    # Input fields for categorical features
    manufacturer = st.selectbox("Manufacturer", label_encoders['Manufacturer'].classes_)
    location = st.selectbox("Location", label_encoders['Location'].classes_)
    fuel_type = st.selectbox("Fuel Type", label_encoders['Fuel_Type'].classes_)
    transmission = st.selectbox("Transmission", label_encoders['Transmission'].classes_)
    owner_type = st.selectbox("Owner Type", label_encoders['Owner_Type'].classes_)
    
    # Create input data DataFrame
    input_data = pd.DataFrame({
        'Manufacturer': [manufacturer],
        'age': [2023 - year],
        'Year': [year],
        'Location': [location],
        'Kilometers_Driven': [kilometers_driven],
        'Fuel_Type': [fuel_type],
        'Transmission': [transmission],
        'Owner_Type': [owner_type],
        'Engine CC': [engine_cc],
        'Power': [power],
        'Seats': [seats],
        'Mileage Km/L': [mileage]
    })
    
    # Predict car price
    predicted_price = preprocess_and_predict_price(num_col, cat_cols, input_data, label_encoders, model)

    # Display predicted car price
    st.write("Predicted Car Price (in lakhs):", round(predicted_price, 2))

if __name__ == '__main__':
    main()
