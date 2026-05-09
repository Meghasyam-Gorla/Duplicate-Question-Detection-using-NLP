import streamlit as st
import pickle
import pandas as pd

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Duplicate Question Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------
# Custom Styling
# ---------------------------------------------------

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 4rem;
    padding-right: 4rem;
}

textarea {
    font-size: 16px !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# App Title
# ---------------------------------------------------

st.title("Duplicate Question Pair Detection")

st.write("Enter two questions and check whether they are duplicates.")

# ---------------------------------------------------
# Load model and tfidf
# ---------------------------------------------------

@st.cache_resource
def load_files():

    with open('xgb.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    return model, tfidf


model, tfidf = load_files()

# ---------------------------------------------------
# Import feature engineering functions
# ---------------------------------------------------

from testing import feature_eng_1, feature_eng_2

# ---------------------------------------------------
# Input Section
# ---------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    q1 = st.text_area(
        "Enter Question 1",
        height=200,
        placeholder="Type first question here..."
    )

with col2:

    q2 = st.text_area(
        "Enter Question 2",
        height=200,
        placeholder="Type second question here..."
    )

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

st.write("")

if st.button("Predict Duplicate", use_container_width=True):

    if q1.strip() == "" or q2.strip() == "":

        st.warning("Please enter both questions")

    else:

        # Create dataframe
        df = pd.DataFrame({
            'question1': [q1],
            'question2': [q2]
        })

        # Feature Engineering
        x1 = feature_eng_1(df)

        x2 = feature_eng_2(x1)

        # Prediction
        result = model.predict(x2)[0]

        # Probability
        try:
            prob = model.predict_proba(x2)[0][1]
        except:
            prob = None

        st.write("---")

        # Result
        if result == 1:

            st.success("Duplicate Questions")

        else:

            st.error("Not Duplicate Questions")

        # Probability
        if prob is not None:

            st.subheader("Prediction Probability")

            st.progress(float(prob))

            st.write(f"Duplicate Probability: {round(prob * 100, 2)}%")