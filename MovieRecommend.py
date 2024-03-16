import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

genai.configure(api_key="AIzaSyAY2xggCq0c4D09SVqlRn46Xto7MxCsOS0")

st.set_page_config(
    page_title="Movie Recommendation",
    page_icon="🎭"
)

with st.sidebar:

    os.environ['GOOGLE_API_KEY'] = "AIzaSyDd-C1_qLdpV8qo_fBbGcwhjyWHGZxuX6c"
    "[Get a Google Gemini API key](https://ai.google.dev/)"
    "[View the source code](https://github.com/wms31/streamlit-gemini/blob/main/app.py)"

st.title("Movie Recommendation Program")
st.caption("App to recommend movies using Google Gemini")

tab1, tab2 = st.tabs(["Generate Recommendations for Movies", "Visual Venture - Gemini Pro Vision"])

with tab1:
    st.write("Using Gemini Pro - Text only model")
    st.subheader("Recommend Movies!")

    movie_genre = st.text_input("Enter Genre: \n\n", key="movie_genre",
                                     value="")
    number_of_movies = st.text_input("How many movies would you like recommended to you? \n\n", key="number_of_movies", value="")
    similar_movies = st.text_input("What are some similar movies from this genre you enjoy? \n\n",
                                         key="similar_movies", value="")

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
with tab2:
    st.write("🖼️ Using Gemini Pro Vision - Multimodal model")
    st.subheader("🔮 Generate image to text responses")

    image_prompt = st.text_input("Ask any question about the image", placeholder="Prompt", label_visibility="visible",
                                 key="image_prompt")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    image = ""

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

    submit = st.button("Generate Response")

    if submit:
        model = genai.GenerativeModel('gemini-pro-vision')
        with st.spinner("Generating your response using Gemini..."):
            if image_prompt != "":
                response = model.generate_content([image_prompt, image])
            else:
                response = model.generate_content(image)
        response = response.text
        st.subheader("Gemini's response")
        st.write(response)