import streamlit as st
import pandas as pd
import joblib


@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load('obesity_model.pkl')

model = load_model()


st.title("🩺 Previsor de Nível de Obesidade")
st.markdown("""
Este app usa um modelo LightGBM treinado para prever o nível de obesidade  
baseado em características demográficas e de estilo de vida.
""")


st.header("Dados do Paciente")
gender         = st.selectbox("Gênero", ["Female","Male"])
age            = st.number_input("Idade", min_value=0, max_value=120, value=30)
height         = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
weight         = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
family_history = st.selectbox("Histórico Familiar de Obesidade", ["yes","no"])
FAVC           = st.selectbox("Come calóricos com frequência?", ["yes","no"])
FCVC           = st.selectbox("Costuma comer vegetais?", 
                              ["Sometimes","Frequently","Always","no"])
NCP            = st.number_input("Refeições principais por dia", min_value=1, max_value=10, value=3)
CAEC           = st.selectbox("Come algo entre refeições?", ["no","Sometimes","Frequently","Always"])
SMOKE          = st.selectbox("Fuma?", ["yes","no"])
CH2O           = st.number_input("Litros de água por dia", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
SCC            = st.selectbox("Monitora calorias ingeridas?", ["yes","no"])
FAF            = st.number_input("Atividade física (h/semana)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
TUE            = st.number_input("Uso de dispositivos (h/dia)", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
CALC           = st.selectbox("Bebe álcool com que frequência?", ["no","Sometimes","Frequently","Always"])
MTRANS         = st.selectbox("Meio de transporte", 
                 ["Public_Transportation","Walking","Automobile","Motorbike","Bike"])


if st.button("🔍 Prever Nível de Obesidade"):
    # Monta DataFrame de 1 linha
    df_input = pd.DataFrame([{
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history': family_history,
        'FAVC': FAVC,
        'FCVC': FCVC,
        'NCP': NCP,
        'CAEC': CAEC,
        'SMOKE': SMOKE,
        'CH2O': CH2O,
        'SCC': SCC,
        'FAF': FAF,
        'TUE': TUE,
        'CALC': CALC,
        'MTRANS': MTRANS
    }])
    # Predição
    pred = model.predict(df_input)[0]
    proba = model.predict_proba(df_input).max()
    st.success(f"🩺 Nível previsto: **{pred.replace('_',' ')}**")
    st.write(f"Confiança: **{proba*100:.1f}%**")
