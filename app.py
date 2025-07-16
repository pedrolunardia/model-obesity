import streamlit as st
import pandas as pd
import joblib 

dados = pd.read_csv("Obesity.csv")
pipe = joblib.load("obesity_model.pkl")

st.set_page_config(
    page_title="Previsor de Nível de Obesidade",
    layout="wide",        # <— aqui
    initial_sidebar_state="auto"
)

st.title("🩺 Previsor de Nível de Obesidade")
st.markdown("""
Este app usa um modelo LightGBM treinado para prever o nível de obesidade baseado em características demográficas e de estilo de vida.     
Preencha os dados abaixo com as informações do paciente e clique em "🔍 Prever Nível de Obesidade" para obter a previsãos.
""")

col1, col2, col3 = st.columns(3)

yesno_map   = {"Sim": "yes", "Não": "no"}
caec_map    = {"Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always", "Não": "no"}
calc_map    = {"Não": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
mtrans_map  = {
    "Transporte público": "Public_Transportation",
    "Caminhada":          "Walking",
    "Automóvel":          "Automobile",
    "Motocicleta":        "Motorbike",
    "Bicicleta":          "Bike"
}

with col1: 
    st.header("Características físicas")
    st.write("Gênero biológico")
    raw_genero = st.selectbox("Selecione o gênero do paciente", ("Masculino", "Feminino"))
    genero_map = {"Masculino": "Male", "Feminino": "Female"}
    input_genero = genero_map[raw_genero]
    st.write("Idade")
    input_idade = float(st.number_input("Selecione a idade do paciente", 1, 100))
    st.write("Altura")
    input_altura = float(st.number_input("Selecione a altura do paciente (em cm)", 0, 250))
    st.write("Peso")
    input_peso = float(st.slider("Selecione o peso do paciente (em kg)", 0, 250))

with col2:
    st.header("Hábitos alimentares")
    raw_hfam = st.selectbox("Paciente possui histórico familiar de obesidade?", ("Sim", "Não"))
    input_historico_familiar = yesno_map[raw_hfam]

    raw_favc = st.selectbox("Paciente come alimentos calóricos com frequência?", ("Sim", "Não"))
    input_frequencia_caloricos = yesno_map[raw_favc]

    raw_fcvc = st.selectbox("Paciente come vegetais com frequência?", ("Às vezes", "Frequentemente", "Sempre", "Não"))
    input_frequencia_vegetais = caec_map[raw_fcvc]

    input_numero_refeicoes = int(st.number_input("Selecione o número de refeições principais por dia", 1, 10))

    raw_caec = st.selectbox("Paciente come algo entre as refeições?", ("Às vezes", "Frequentemente", "Sempre", "Não"))
    input_consumo_entre_refeicoes = caec_map[raw_caec]


with col3:
    st.header("Outros hábitos")
    raw_smoke = st.selectbox("Paciente fuma?", ("Sim", "Não"))
    input_fuma = yesno_map[raw_smoke]

    input_consumo_agua = float(st.number_input("Selecione a quantidade de água consumida por dia (em litros)", 0.0, 5.0, step=0.1))

    raw_scc = st.selectbox("Paciente monitora as calorias ingeridas?", ("Sim", "Não"))
    input_monitora_calorias = yesno_map[raw_scc]

    input_atividade_fisica = float(st.number_input(
        "Selecione a quantidade de atividade física realizada por semana (em horas)",
        0.0, 20.0, step=0.5
    ))

    input_uso_dispositivos = float(st.number_input("Selecione o tempo de uso de dispositivos por dia (em horas)", 0.0, 24.0, step=0.5))

    raw_calc = st.selectbox("Paciente consome álcool com que frequência?", ("Não", "Às vezes", "Frequentemente", "Sempre"))
    input_consumo_alcool = calc_map[raw_calc]

    raw_mtrans = st.selectbox("Selecione o meio de transporte utilizado pelo paciente", ("Transporte público", "Caminhada", "Automóvel", "Motocicleta", "Bicicleta"))
    input_meio_transporte = mtrans_map[raw_mtrans]

label_map = {
    "Insufficient_Weight":    "Peso Insuficiente",
    "Normal_Weight":          "Peso Normal",
    "Overweight_Level_I":     "Sobrepeso Grau I",
    "Overweight_Level_II":    "Sobrepeso Grau II",
    "Obesity_Type_I":         "Obesidade Grau I",
    "Obesity_Type_II":        "Obesidade Grau II",
    "Obesity_Type_III":       "Obesidade Grau III"
}

if st.button("🔍 Prever Nível de Obesidade"):
    dados_novo = {
        "Gender":      input_genero,
        "Age":         input_idade,
        "Height":      input_altura,
        "Weight":      input_peso,
        "family_history":   input_historico_familiar,
        "FAVC":         input_frequencia_caloricos,
        "FCVC":         input_frequencia_vegetais,
        "NCP":          input_numero_refeicoes,
        "CAEC":         input_consumo_entre_refeicoes,
        "SMOKE":        input_fuma,
        "CH2O":         input_consumo_agua,
        "SCC":          input_monitora_calorias,
        "FAF":          input_atividade_fisica,
        "TUE":          input_uso_dispositivos,
        "CALC":         input_consumo_alcool,
        "MTRANS":       input_meio_transporte
    }
    df_novo = pd.DataFrame([dados_novo])
    raw_pred = pipe.predict(df_novo)[0]
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown("## Resultado da Previsão")
        st.success(
            f"**Nível previsto:**  \n**{label_map[raw_pred]}**",
            icon="🏥"
        )
