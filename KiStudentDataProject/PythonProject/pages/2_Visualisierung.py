import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

st.title("Visualisierungen")


@st.cache_data
def load_data():
    df = pd.read_csv('data/ai_students.csv')
    df_use_case = pd.read_csv('data/use_case.csv')
    return df, df_use_case


df, df_use_case = load_data()
df_students = df.drop_duplicates(subset='Student_ID').copy()

# Sidebar: Plot-Auswahl
plot_type = st.sidebar.selectbox(
    "Wähle Plot:",
    ["Verteilung Nummer Features", "Einschätzung Noten", "Noten Analyse", "KI Modelle", "Anwendungsfall"]
)
# IHR CODE AUS NOTEBOOK 2 HIER:
if plot_type == "Verteilung Nummer Features":
    st.subheader("Verteilung eines Features")

    feature = st.selectbox("Feature auswählen:",
                           df_students.select_dtypes(include='number').columns)

    fig, ax = plt.subplots()
    df_students[feature].hist(bins=20, ax=ax)
    ax.set_xlabel(feature)
    ax.set_ylabel('Häufigkeit')
    st.pyplot(fig)

elif plot_type == "Einschätzung Noten":
    st.subheader("Analyse: Einfluss auf Noten")

    figure = plt.figure(figsize=(10, 6))
    sns.countplot(data=df_students, x='Impact_on_Grades', palette='viridis')
    plt.title('Wie schätzen Studierende den Einfluss von KI auf ihre Noten ein?')
    plt.xlabel('Einfluss auf Noten (-5 = sehr negativ, +5 = sehr positiv)')
    plt.ylabel('Anzahl der Studierenden')
    st.pyplot(figure)
    with st.expander("Analyse"):
        st.write(
            """
            Die Grafik zeigt eine breite Streuung der Meinungen. Zwar liegt der häufigste Einzelwert bei +2, jedoch gibt es fast ebenso viele extrem negative (-5) wie extrem positive (+5) Einschätzungen. 
            Das belegt, dass KI keinen universellen Erfolg garantiert, sondern polarisiert. 
            Für eine große Gruppe scheint sie sehr nützlich zu sein, für eine andere Gruppe hinderlich.
            """
        )

elif plot_type == "Noten Analyse":
    st.subheader("Analyse: Haben Studenten die viel KI nutzen wirklich bessere Noten?")

    # Visualisierung
    df_students['Usage_Group'] = pd.cut(df_students['Daily_Usage_Hours'],
                               bins=[0, 1.5, 3.5, 6],
                               labels=['Wenig (< 1.5h)', 'Mittel (1.5-3.5h)', 'Viel (> 3.5h)'])

    figure = plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_students, x='Usage_Group', y='Impact_on_Grades', palette='viridis')
    plt.xlabel('Nutzungsgruppe')
    plt.ylabel('Wahrgenommener Noteneinfluss')
    plt.axhline(0, color='red', linestyle='--', alpha=0.5, label='Neutral')
    plt.grid(axis='y', alpha=0.3)
    st.pyplot(figure)
    with st.expander("Analyse"):
        st.write("""
                Die Boxplots liegen für alle drei Gruppen auf fast identischem Niveau.
                Dies zeigt, dass eine längere Nutzungsdauer allein nicht automatisch zu einer besseren Wahrnehmung der Studienleistung führt.
                 """)

elif plot_type == "KI Modelle":
    st.subheader("Analyse: KI Tools")
    # Visualisierung 3:
    grouped_data = df_students.groupby('Preferred_AI_Tool')['Impact_on_Grades'].mean().sort_values(ascending=False)

    figure = plt.figure(figsize=(10, 6))

    sns.barplot(x=grouped_data.index, y=grouped_data.values, palette='viridis', edgecolor='black')

    plt.title('Welches Tool bringt laut den Studtenten den größten gefühlten Vorteil?')
    plt.xlabel('Bevorzugtes KI-Tool')
    plt.ylabel('Durchschnittlicher Einfluss (-5 bis +5)')

    # Rote Nulllinie
    plt.axhline(0, color='red', linestyle='--', label='Neutral')

    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    st.pyplot(figure)
    with st.expander("Analyse"):
        st.write("""
           Die Grafik zeigt leichte Unterschiede zwischen den Anbietern. Während Nutzer von Gemini und ChatGPT im Schnitt eine minimale Verbesserung wahrnehmen, berichten Nutzer von Copilot und Bard tendenziell eher von negativen Effekten.
           Die Durchschnittswerte sind insgesamt sehr gering (Range nur von ca. -0,4 bis +0,2 auf einer Skala von -5 bis +5). 
           Das bedeutet, dass sich extrem positive und negative Meinungen innerhalb jeder Tool-Gruppe fast gegenseitig aufheben. Der Erfolg hängt vermutlich weniger vom Tool selbst ab, sondern eher von der individuellen Kompetenz der Studierenden.
           """)


elif plot_type == "Anwendungsfall":
    st.subheader("Analyse: Anwendungsfall")
    # Welcher Anwendungsfall bringt den meisten Erfolg?

    case_impact = df_use_case.groupby('Use_Cases')['Impact_on_Grades'].mean().sort_values(
        ascending=False)  # df --> damit alle cases berücksichtigt werden

    figure = plt.figure(figsize=(12, 6))
    sns.barplot(x=case_impact.index, y=case_impact.values, palette='viridis', edgecolor='black')

    plt.title('Hängt der Erfolg vom Anwendungszweck ab?', fontsize=14)
    plt.xlabel('Anwendungsfall')
    plt.ylabel('Durchschnittlicher Noten-Einfluss')
    plt.axhline(0, color='red', linestyle='-', label='Neutral')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    st.pyplot(figure)
    with st.expander("Analyse"):
        st.write("""
        Studierende, die KI nutzen, um Inhalte zu erstellen, Notizen zu strukturieren,Programmieraufgaben zu lösen oder sich auf Prüfungen vorbereiten, empfinden eine deutliche Verbesserung ihrer Noten. 
        Hingegen zeigt sich bei spezifischen Übungsformaten wie Multiple Choice und beim Klären von Verständnisfragen ein negativer Trend.
        Besonders auffällig ist, dass die Nutzung von KI für "Projects" mit einer schlechteren Benotung korreliert, vielleicht weil die Themen und Strukturen oft zu komplex sind.
        Es kommt also nicht nur darauf an, *dass* man KI nutzt, sondern ob sie als Werkzeug zur Produktivität (Text, Code, Notizen) oder als Ersatz für tiefgehendes Üben (Projekte, MCQs) eingesetzt wird.
        """)
