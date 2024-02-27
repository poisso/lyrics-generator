# Lyrics Generator App

This Streamlit app allows users to generate song lyrics inspired by a selected band's style. The app uses language models, vector stores, and retrievers to provide creative and thematic lyric generation.
Features

- Band Selection: Users can select a band from the provided list of subdirectories within the lyrics_data directory.

- Language Option: Choose between English (en) and French (fr) for lyric generation.

- User-defined Theme: Input a custom theme for the generated lyrics.

- Lyric Generation: Click the "Generate Lyrics" button to invoke the lyric generation process based on the selected band's style and the provided theme.


# Directory Structure

- lyrics_data/: Contains subdirectories, each named after a band, with their respective lyrics.

- scraper.ipynb: Jupyter Notebook for scraping lyrics using the AzLyrics website.

# Dependencies

- Streamlit
- langchain_community
- langchain_openai
- langchain_core

# Notes

    This app utilizes language models from OpenAI and other components from the LangChain community.

    The azlyrics_scraper.ipynb notebook can be used to retrieve lyrics from the AzLyrics website.

