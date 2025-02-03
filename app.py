import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# Load datasets
anime = pd.read_csv('anime.csv')
synopsis = pd.read_csv('anime_with_synopsis.csv')

# Merge datasets
anime = anime.merge(synopsis, on='Name')
anime.dropna(inplace=True)

# Data preprocessing
def preprocess_data(df):
    df["Genres"] = df["Genres"].apply(lambda x: [genre.strip() for genre in x.split(",")] if isinstance(x, str) else x)
    df["Description"] = df["Description"].apply(lambda x: x.split() if isinstance(x, str) else x)
    df["synopsis"] = df["synopsis"].apply(lambda x: x.split() if isinstance(x, str) else x)
    df["tags"] = df["Genres"] + df["Description"] + df["synopsis"]
    df["tags"] = df["tags"].apply(lambda x: " ".join(x))
    df["tags"] = df["tags"].apply(lambda x: x.lower())
    return df

anime = preprocess_data(anime)

# Stemming function
ps = PorterStemmer()
def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

anime["tags"] = anime["tags"].apply(stem)

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(anime["tags"]).toarray()

# Similarity matrix
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(anime_name):
    if anime_name not in anime["Name"].values:
        return ["Anime not found! Please try another name."]
    anime_index = anime[anime["Name"] == anime_name].index[0]
    distances = similarity[anime_index]
    anime_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [anime.iloc[i[0]].Name for i in anime_list]

# Streamlit Interface
st.title("Anime Recommender System")
st.markdown("Find similar anime based on your favorite!")

selected_anime = st.selectbox("Choose an anime:", anime["Name"].values)

if st.button("Recommend"):
    recommendations = recommend(selected_anime)
    st.write("### Recommended Anime:")
    for rec in recommendations:
        st.write(f"- {rec}")
