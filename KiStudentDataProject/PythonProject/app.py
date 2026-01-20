import streamlit as st
import pandas as pd
st.set_page_config(page_title="Impact of AI Usage ", page_icon="📊 ", layout="wide")
st.title("📊 Einfluss von KI auf Studiennoten")
st.markdown(""" Diese App analysiert die KI Nutzung von Studenten und erstellt
Vorhersagen.""")
@st.cache_data
def load_data():
    df = pd.read_csv('data/ai_students.csv')
    df_old = pd.read_csv('data/Students_old.csv')
    return df, df_old

df, df_old = load_data()
df_students = df.drop_duplicates(subset='Student_ID')



col1, col2, col3, col4 = st.columns(4)
col1.metric("Daten", len(df))
col2.metric("Anzahl Studenten", len(df_students))
col3.metric("Features", len(df.columns))
avg_usage = df_students['Daily_Usage_Hours'].mean()
col4.metric("Durchschnittliche tägliche KI Nutzung", f"{avg_usage:.2f} Std")

st.subheader("1. Explodierter Datensatz")
st.caption("Dieser bereinigte Datensatz enthält alle Einträge mit explodierten Use-Cases. Ein Student kann hier mehrfach vorkommen.")
st.dataframe(df.head())

st.subheader("2. Datensatz ohne Duplikate")
st.caption("In diesem bereinigten Datensatz kommt jeder Student genau einmal vor.")
st.dataframe(df_students.head())

st.subheader("3. Originaler Datensatz")
st.caption("Der Orginale unberenigte Datensatz.")
st.dataframe(df_old.head())
