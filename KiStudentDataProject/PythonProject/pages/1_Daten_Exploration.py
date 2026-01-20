import streamlit as st
import pandas as pd

st.title("Daten Exploration")
@st.cache_data
def load_data():
 return pd.read_csv('data/ai_students.csv')
df = load_data()

# Sidebar: Filter
st.sidebar.header("Filter Optionen")
grade_col = 'Impact_on_Grades'
if grade_col in df.columns:
 grade_range = st.sidebar.slider(
 "Noten:",
 int(df[grade_col].min()),
 int(df[grade_col].max()),
 (int(df[grade_col].min()), int(df[grade_col].max()))
 )
 df = df[(df[grade_col] >= grade_range[0]) & (df[grade_col] <=
grade_range[1])]

if 'Stream' in df.columns:
     all_streams = df['Stream'].unique()
     selected_streams = st.sidebar.multiselect(
         "Studienrichtung:",
         options=all_streams,
         default=all_streams
     )
     df = df[df['Stream'].isin(selected_streams)]

df_students = df.drop_duplicates(subset='Student_ID')

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Übersicht", "📈 Statistiken", "🔢Rohdaten & Export"])
with tab1:
 col1, col2 = st.columns(2)
 col1.metric("Gefilterte Studenten", len(df_students))
 col2.metric("Spalten", len(df_students.columns))
with tab2:
 st.dataframe(df_students.describe())
with tab3:
    st.subheader("Welchen Datensatz möchtest du sehen?")
    dataset_choice = st.radio(
        "Ansicht wählen:",
        ("Einzigartige Studenten", "Doppelte Studenten")
    )

    if dataset_choice == "Einzigartige Studenten":
        data_showing = df_students
        filename = "students_unique.csv"
    else:
        data_showing = df
        filename = "exploded_data.csv"

    st.dataframe(data_showing, use_container_width=True)

    # Download Button
    # https://docs.streamlit.io/knowledge-base/using-streamlit/how-download-pandas-dataframe-csv
    csv = data_showing.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Daten als CSV herunterladen",
        data=csv,
        file_name=filename,
        mime='csv',
    )