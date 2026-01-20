import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("Machine Learning")


@st.cache_data
def load_data():
    return pd.read_csv('data/ai_students.csv')


df = load_data()
df_students = df.drop_duplicates(subset='Student_ID')

# Zwei Spalten: Training | Vorhersage
col1, col2 = st.columns(2)
with col1:
    st.header("Modell Training")

    # AB HIER OPTIONAL - Falls ihr die ML-Übung gemacht habt:
    if st.button("Modell trainieren"):
        df_model = df_students[
            ['Daily_Usage_Hours', 'Trust_in_AI_Tools', 'Year_of_Study', 'Impact_on_Grades', 'Stream']].dropna()


        def categorize(grade):
            return 1 if grade > 0 else 0


        y = df_model['Impact_on_Grades'].apply(categorize)
        X = df_model[['Daily_Usage_Hours', 'Trust_in_AI_Tools', 'Year_of_Study', 'Stream']].copy()
        X = pd.get_dummies(X, columns=['Stream'])

        # Train/Test Split
        X_train, X_test, y_train_binary, y_test_binary = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Modell trainieren
        st.write("=== K-Nearest Neighbors (KNN) ===")
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train_binary)

        y_pred_knn = knn.predict(X_test)
        accuracy_knn = accuracy_score(y_test_binary, y_pred_knn)
        st.write(f"Accuracy: {accuracy_knn:.2%}\n")

        # Evaluation
        best_model = knn
        y_pred_best = y_pred_knn

        cm = confusion_matrix(y_test_binary, y_pred_best)
        figure = plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        st.pyplot(figure)

        # Modell speichern
        with open('best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
            st.session_state['best_model'] = best_model
            st.session_state['model_columns'] = X.columns.tolist()
            st.success("Modell wurde trainiert und gespeichert!")

with col2:
    st.header("Vorhersage")

    if 'best_model' in st.session_state:
        best_model = st.session_state['best_model']
        model_columns = st.session_state['model_columns']

        st.write("Bitte Werte für einen Studenten eingeben:")

        usage = st.number_input("Tägliche Nutzung (Stunden)", 0.0, 24.0, 2.0, 0.5)
        trust = st.slider("Vertrauen in AI (1-5)", 1, 5, 3)
        year = st.number_input("Studienjahr", 1, 10, 1)

        streams = ['Science', 'Commerce', 'Arts', 'Engineering', 'Medical', 'Agriculture', 'Law', 'Management',
                   'Hotel-management', 'Pharmacy']
        selected_stream = st.selectbox("Studiengang", streams)

        if st.button("Vorhersage starten"):
            input_data = pd.DataFrame({
                'Daily_Usage_Hours': [usage],
                'Trust_in_AI_Tools': [trust],
                'Year_of_Study': [year],
                'Stream': [selected_stream]
            })

            input_data = pd.get_dummies(input_data, columns=['Stream'])

            input_data_aligned = input_data.reindex(columns=model_columns, fill_value=0)

            # Vorhersage
            prediction = best_model.predict(input_data_aligned)
            probabilities = best_model.predict_proba(input_data_aligned)


            # Ergebnis anzeigen
            st.success("Berechnung abgeschlossen!")

            result = "Noten verbessern sich (1)" if prediction[0] == 1 else "Keine Verbesserung (0)"
            st.subheader(f"Ergebnis: {result}")

            st.write(f"Wahrscheinlichkeit (Nein / Ja): {probabilities[0]}")

    else:
        st.info("Bitte erst ein Modell links trainieren!")