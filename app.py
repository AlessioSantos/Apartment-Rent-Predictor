import os
import pandas as pd
import streamlit as st
import joblib
import boto3
from io import BytesIO
from PIL import Image

# ========================================
# 1. AWS S3
# ========================================
s3 = boto3.client("s3")

BUCKET_NAME = "depression-model-storage"
IMAGE_KEY = "Photos/010057antalyaoldcitytour.jpeg"
MODEL_KEY = "LLM_Models/Antalya_rental_price_model_02_deviation_from_mean_106.pkl"
# ========================================
# 2. Функции для загрузки модели и изображений
# ========================================
@st.cache_data
def load_image_from_s3(bucket_name, object_key):
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    return Image.open(BytesIO(response["Body"].read()))

@st.cache_resource
def load_model_from_s3(bucket_name, object_key):
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    model_data = response["Body"].read()
    model = joblib.load(BytesIO(model_data))
    return model

# ========================================
# 3. Загрузка модели
# ========================================
try:
    model = load_model_from_s3(BUCKET_NAME, MODEL_KEY)
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

# ========================================
# 4. Заголовок и изображение
# ========================================
st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
        color: darkblue;
        font-size: 40px;
        font-weight: bold;
    }
    .centered-subtitle {
        text-align: center;
        color: gray;
        font-size: 20px;
        margin-top: -10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="centered-title">Apartment Rent Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="centered-subtitle">Find out the estimated rental price in Antalya (Turkey)</div>', unsafe_allow_html=True)

try:
    image = load_image_from_s3(BUCKET_NAME, IMAGE_KEY)
    st.image(image, use_container_width=True)
except Exception as e:
    st.error(f"Ошибка загрузки изображения: {e}")

# ========================================
# 5. Ввод данных пользователем
# ========================================
st.sidebar.header("Enter apartment details:")

total_area = st.sidebar.number_input("Total Area (m²):", min_value=20, max_value=500, value=80)
living_area = st.sidebar.number_input("Living Area (m²):", min_value=15, max_value=400, value=60)
rooms = st.sidebar.slider("Rooms:", min_value=1, max_value=10, value=2)
building_age = st.sidebar.slider("Building Age:", min_value=0, max_value=100, value=10)
floor = st.sidebar.slider("Floor:", min_value=0, max_value=50, value=2)
total_floors = st.sidebar.slider("Total Floors in Building:", min_value=1, max_value=50, value=10)
bathrooms = st.sidebar.slider("Bathrooms:", min_value=1, max_value=5, value=1)

# Вопросы Yes/No вместо 0 и 1
balcony = st.sidebar.radio("Balcony:", ["Yes", "No"])
elevator = st.sidebar.radio("Elevator:", ["Yes", "No"])
residential_complex = st.sidebar.radio("In Residential Complex:", ["Yes", "No"])
furnished = st.sidebar.radio("Furnished:", ["Yes", "No"])

maintenance_fee = st.sidebar.number_input("Maintenance Fee (USD):", min_value=0, max_value=1000, value=50)
deposit = st.sidebar.number_input("Deposit (USD):", min_value=0, max_value=10000, value=1000)

neighborhood = st.sidebar.selectbox(
    "Neighborhood:",
    [
        "Bahcelievler Mh.", "Balbey Mah.", "Bayindir Mh.", "Caglayan Mh.", 
        "Demircikara Mah.", "Doguyaka Mh.", "Fener Mah.", "Muratpasa Mh.", 
        "Selcuk Mh.", "Sirinyalı Mh.", "Yesilbahce Mh."
    ]
)

heating_type = st.sidebar.selectbox(
    "Heating Type:",
    ["Boiler (Electric)", "Central", "Combi Boiler (Natural Gas)", "Floor Heating", "None"]
)

parking = st.sidebar.selectbox(
    "Parking:",
    ["Open Parking Lot", "Parking Garage", "No Parking"]
)

# ========================================
# 6. Подготовка данных для модели
# ========================================
def prepare_input_data():
    # Получаем список признаков, которые использовались в обучении модели
    feature_names = model.feature_names_in_
    
    # Создаем DataFrame с нулями, но с правильными колонками
    input_data = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Заполняем числовые признаки
    input_data["Total_Area_m2"] = total_area
    input_data["Living_Area_m2"] = living_area
    input_data["Rooms"] = rooms
    input_data["Building_Age"] = building_age
    input_data["Floor"] = floor
    input_data["Total_Floors"] = total_floors
    input_data["Bathrooms"] = bathrooms
    input_data["Maintenance_Fee"] = maintenance_fee
    input_data["Deposit"] = deposit

    # Преобразуем Yes/No в 0/1
    input_data["Balcony"] = 1 if balcony == "Yes" else 0
    input_data["Elevator"] = 1 if elevator == "Yes" else 0
    input_data["In_Residential_Complex"] = 1 if residential_complex == "Yes" else 0
    input_data["Furnished_1"] = 1 if furnished == "Yes" else 0

    # Обрабатываем категориальные признаки (Neighborhood, Heating_Type, Parking)
    neighborhood_col = f"Neighborhood_{neighborhood}"
    heating_col = f"Heating_Type_{heating_type}"
    parking_col = f"Parking_{parking}"

    if neighborhood_col in input_data.columns:
        input_data[neighborhood_col] = 1

    if heating_col in input_data.columns:
        input_data[heating_col] = 1

    if parking_col in input_data.columns:
        input_data[parking_col] = 1

    return input_data

# ========================================
# 7. Кнопка предсказания и отображение результата
# ========================================
if st.sidebar.button("Predict Rent Price"):
    user_data = prepare_input_data()
    prediction = model.predict(user_data)[0]

    st.subheader(f"Predicted Monthly Rent: **{prediction:.2f} USD**")