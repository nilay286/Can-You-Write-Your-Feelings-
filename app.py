import streamlit as st
import numpy as np
import pandas as pd
import joblib
import altair as alt
import random

pipe_lr = joblib.load(open('emotion_classifier.pkl', 'rb'))

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {"kizgin":"😠😠", "korku":"😨😱", "mutlu":"🤗🤗", "uzgun":"😔😔", "surpriz":"😮😮"}
emotions_list = list(emotions_emoji_dict.keys())

def main():
    st.title('Duygularını Yazabilir Misin?')
    menu = ["Home"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        # Initialize counters and score dictionary
        attempts = 3
        correct_count_dict = {emotion: 0 for emotion in emotions_list}
        incorrect_count_dict = {emotion: 0 for emotion in emotions_list}
        score = 0  # Initialize overall score

        total_attempts = attempts * len(emotions_list)

        if 'current_emotion_index' not in st.session_state:
            st.session_state.current_emotion_index = 0
        if 'attempt_count' not in st.session_state:
            st.session_state.attempt_count = 0
        if 'form_counter' not in st.session_state:
            st.session_state.form_counter = 0

        # Ensure current_emotion_index is within valid range
        if st.session_state.current_emotion_index >= len(emotions_list):
            st.session_state.current_emotion_index = 0

        emotion = emotions_list[st.session_state.current_emotion_index]

        st.write("");st.write("");st.write("");st.write("");
        st.write("'{}' duyguyu metin olarak yazabilir misin?".format(emotion))
        st.write(emotions_emoji_dict[emotion])

        with st.form(key='emotion_clf_form_{}_{}_{}'.format(emotion, st.session_state.attempt_count, st.session_state.form_counter)):
            raw_text = st.text_area("Lütfen metninizi giriniz.")
            submit_text = st.form_submit_button(label="Gönder")

            if submit_text:
                try:
                    prediction = predict_emotions(raw_text)

                    # Check if the user's text matches the random emotion
                    if prediction[0] == emotion:
                        st.success("Doğru! {} Deneme hakkınız kaldı!".format(attempts - st.session_state.attempt_count - 1))
                        correct_count_dict[emotion] += 1
                        score += 3.33  # Update score by base score on each correct answer

                    else:
                        st.error("Yanlış! {} Deneme hakkınız kaldı!".format(attempts - st.session_state.attempt_count - 1))
                        incorrect_count_dict[emotion] += 1
                        score -= (3.33 / attempts)  # Decrease score proportionally on each incorrect answer

                    # Access prediction here since it's defined within the try block

                    col1, col2 = st.columns(2)

                    # Combine prediction text and emoji icon in col1
                    with col1:
                        st.success("Tahmin Sonucu:")
                        emoji_icon = emotions_emoji_dict[prediction[0]]
                        st.write("{}:{}".format(prediction[0], emoji_icon))

                    # Display probability table in col1 as well
                    with col1:
                        proba_df = pd.DataFrame(get_prediction_proba(raw_text), columns=pipe_lr.classes_)
                        st.write(proba_df.transpose())

                    # Display chart in col2
                    with col2:
                        st.success('Tahmin Grafiği:')
                        proba_df_clean = proba_df.transpose().reset_index()
                        proba_df_clean.columns = ["emotions", "probability"]
                        fig = alt.Chart(proba_df_clean).mark_bar().encode(
                            x='emotions', y='probability', color='emotions')
                        st.altair_chart(fig, use_container_width=True)

                    st.session_state.attempt_count += 1
                    st.session_state.form_counter += 1

                    if st.session_state.attempt_count == attempts:
                        st.session_state.current_emotion_index += 1
                        st.session_state.attempt_count = 0

                    if st.session_state.current_emotion_index == len(emotions_list):
                        st.write(f"Overall Score: {score}")

                except Exception as e:
                    st.error(f"Error: {e}")

    elif choice == "Monitor":
        st.subheader("Monitor App")
        # Implement monitoring functionality here

    else:
        st.subheader("About")
        # Implement about page content here


if __name__ == "__main__":
    main()