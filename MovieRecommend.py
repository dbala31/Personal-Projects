import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key="AIzaSyAY2xggCq0c4D09SVqlRn46Xto7MxCsOS0")

st.set_page_config(
    page_title="Movie Recommendation",
    page_icon="🎭"
)

with st.sidebar:

    os.environ['GOOGLE_API_KEY'] = "AIzaSyDd-C1_qLdpV8qo_fBbGcwhjyWHGZxuX6c"
    "[Get a Google Gemini API key](https://ai.google.dev/)"
    "[View the source code](https://github.com/dbala31/Personal-Projects/blob/main/MovieRecommend.py)"

st.title("Movie Recommendation Program")
st.caption("App to recommend movies using Google Gemini")

tab1, = st.tabs(["Generate Recommendations for Movies"])

# Using the tab to add content
tab1.write("Using Gemini Pro - Text only model")
tab1.subheader("Recommend Movies!")

# Continue to use `tab1` to add other widgets or content to this tab
movie_genre = tab1.text_input("Enter Genre: \n\n", key="movie_genre", value="")
number_of_movies = tab1.text_input("How many movies would you like recommended to you? \n\n", key="number_of_movies", value="")
similar_movies = tab1.text_input("What are some similar movies from this genre you enjoy? \n\n", key="similar_movies", value="")

prompt = f"""Recommend me a {number_of_movies}-movie list, whose genre is {movie_genre} and is similar to {similar_movies}."""

config = {
    "temperature": 0.8,
    "max_output_tokens": 2048,
}

generate_t2t = tab1.button("Generate Movie Recommendation", key="generate_t2t")
model = genai.GenerativeModel("gemini-pro", generation_config=config)
if generate_t2t and prompt:
    with st.spinner("Generating your movie recommendations..."):
        response = model.generate_content(prompt)
        if response:
            plan_tab, prompt_tab = tab1.tabs(["Movie Recommendations", "Prompt"])
            with plan_tab:
                plan_tab.write("Your plan:")
                plan_tab.write(response.text)
            with prompt_tab:
                prompt_tab.text(prompt)

