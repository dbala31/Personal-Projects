import os
import streamlit as st
import google.generativeai as genai
#from dotenv import load_dotenv
from PIL import Image

# Load environment variables from .env file
#load_dotenv()

# Get the Google API key from the environment variables
#api_key = os.getenv("GOOGLE_API_KEY")

# Configure the Google Generative AI with the API key
genai.configure(api_key="AIzaSyAY2xggCq0c4D09SVqlRn46Xto7MxCsOS0")

st.set_page_config(
    page_title="Movie Recommendation",
    page_icon="🎭"
)

# Check if the Google API key is provided in the sidebar
with st.sidebar:

    os.environ['GOOGLE_API_KEY'] = "AIzaSyDd-C1_qLdpV8qo_fBbGcwhjyWHGZxuX6c"
    "[Get a Google Gemini API key](https://ai.google.dev/)"
    "[View the source code](https://github.com/wms31/streamlit-gemini/blob/main/app.py)"

# Set the title and caption for the Streamlit app
st.title("Movie Recommendation Program")
st.caption("App to recommend movies using Google Gemini")

# Create tabs for the Streamlit app
tab1 = st.tabs(["Generate Recommendations for Movies"])

# Code for Gemini Pro model
with tab1:
    st.write("Using Gemini Pro - Text only model")
    st.subheader("Recommend Movies!")

    movie_genre = st.text_input("Enter Genre: \n\n", key="movie_genre",
                                     value="Drama")
    number_of_movies = st.text_input("How many movies would you like recommended to you? \n\n", key="number_of_movies", value="5")
    similar_movies = st.text_input("What are some similar movies from this genre you enjoy? \n\n",
                                         key="similar_movies", value="Oppenheimer")

    prompt = f"""Recommend me a {number_of_movies}-movie list, whose genre is {movie_genre} and is similar to {similar_movies}."""

    config = {
        "temperature": 0.8,
        "max_output_tokens": 2048,
    }

    generate_t2t = st.button("Generate Movie Recommendation", key="generate_t2t")
    model = genai.GenerativeModel("gemini-pro", generation_config=config)
    if generate_t2t and prompt:
        with st.spinner("Generating your movie recommendations..."):
            plan_tab, prompt_tab = st.tabs(["Movie Recommendations", "Prompt"])
            with plan_tab:
                response = model.generate_content(prompt)
                if response:
                    st.write("Your plan:")
                    st.write(response.text)
            with prompt_tab:
                st.text(prompt)


