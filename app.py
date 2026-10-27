import streamlit as st
import pandas as pd
import pickle
import random
import plotly.express as px
from collections import Counter
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

movies = pickle.load(open("movie_list.pkl", "rb"))
if os.path.exists("similarities.pkl"):
    similarities=pickle.load(open("similarities.pkl", "rb"))
else:
    if st.button("Generate similarities"):
        cv = CountVectorizer(max_features=5000, stop_words='english')
        dtm = cv.fit_transform(movies['tags']).toarray()
        dtm_df=pd.DataFrame(data=dtm, columns=cv.get_feature_names_out())
        similarities = cosine_similarity(dtm_df)
        pickle.dump(similarities, open('similarities.pkl', 'wb'))

if "history" not in st.session_state:
    st.session_state.history = []

def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]

    distances = similarities[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended = []
    for i in movie_list:
        recommended.append(movies.iloc[i[0]].title)
    return recommended

st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.stApp{
background:linear-gradient(-45deg,#0f172a,#111827,#1e293b,#0f172a);
background-size:400% 400%;
animation:gradientBG 15s ease infinite;
color:white;
}

@keyframes gradientBG{
0%{
background-position:0% 50%;
}
50%{
background-position:100% 50%;
}
100%{
background-position:0% 50%;
}
}

section[data-testid="stSidebar"]{
background:#111827;
border-right:1px solid #2d3748;
}


.glass{
background:rgba(255,255,255,0.08);
backdrop-filter:blur(18px);
border-radius:18px;
padding:20px;
border:1px solid rgba(255,255,255,.15);
margin-bottom:20px;
transition:.4s;
}
.glass:hover{
transform:translateY(-6px);
box-shadow:0px 10px 30px rgba(0,255,255,.25);
}


.stButton>button{
width:100%;
height:50px;
border-radius:12px;
font-size:18px;
font-weight:bold;
background:#2563eb;
color:white;
border:none;
transition:.3s;
}

.stButton>button:hover{
background:#1d4ed8;
transform:scale(1.03);
}

.stSelectbox{
font-size:18px;
}

.footer{
text-align:center;
padding:25px;
color:#cbd5e1;
font-size:16px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "",
    [
        "Home",
        "Recommend Movies",
        "Explore Movies",
        "Surprise Me",
        "Statistics",
        "About"
    ]
)

if page == "Home":

    st.markdown("""
    <h1 style='text-align:center;font-size:48px;'>
        Movie Recommendation System
    </h1>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='glass'>
        <h2>Welcome!</h2>
        <p style='font-size:18px;line-height:1.8;'>
        Discover your next favorite movie using an AI-powered
        recommendation system based on <b>Content-Based Filtering</b>.
        Choose a movie you already like and receive five similar
        recommendations instantly. You can also explore movies using
        keywords like <b>Action</b>, <b>Horror</b>, <b>Comedy</b>,
        <b>Romance</b>, and more.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Movies",
            len(movies)
        )

    with col2:

        total_words = len(
            set(
                " ".join(movies["tags"]).split()
            )
        )

        st.metric(
            "Unique Keywords",
            total_words
        )

    with col3:
        st.metric(
            "Recommendation Model",
            "Ready"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("""
        <div class='glass'>
        <h3>Recommend Movies</h3>
        <p>
        Select your favorite movie and receive
        five similar recommendations using
        cosine similarities.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='glass'>
        <h3>Surprise Me</h3>
        <p>
        Don't know what to watch?
        Let the system randomly
        recommend a movie.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class='glass'>
        <h3>Explore Movies</h3>
        <p>
        Search movies using keywords like
        Action, Horror, Romance,
        Adventure, Comedy and many more.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='glass'>
        <h3>Search History</h3>
        <p>
        View your recent searches during
        the current session.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Popular Categories")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info("Action")
        st.info("Adventure")

    with col2:
        st.info("Comedy")
        st.info("Romance")

    with col3:
        st.info("Sci-Fi")
        st.info("Thriller")

    with col4:
        st.info("Fantasy")
        st.info("Animation")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='footer'>
    Developed by <b>Sakshi Anil Gharge</b>
    </div>
    """, unsafe_allow_html=True)

elif page == "Recommend Movies":

    st.title("Movie Recommendation")
    st.write(
        "Select a movie below and discover five similar movies based on content similarities."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    selected_movie = st.selectbox(
        "Choose a Movie",
        movies["title"].values
    )

    if st.button("Recommend Movies"):
        with st.spinner("Finding similar movies..."):
            recommended_movies = recommend(selected_movie)


        if selected_movie not in st.session_state.history:
            st.session_state.history.append(selected_movie)
        st.success("Recommendations Generated Successfully!")
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Top 5 Recommendations")

        for i, movie in enumerate(recommended_movies, start=1):
            movie_info = movies[movies["title"] == movie].iloc[0]
            tags = movie_info["tags"]
            short_tags = " ".join(tags.split()[:30])
            st.markdown(f"""
            <div class="glass">
            <h3>{i}. {movie}</h3>
            <hr>
            <p style="font-size:16px; line-height:1.8;">
            <b>Movie Information</b><br><br>
            {short_tags}...
            </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Recent Searches")
        recent = st.session_state.history[::-1][:5]
        cols = st.columns(len(recent))
        for col, movie in zip(cols, recent):
            with col:
                st.info(movie)

elif page == "Explore Movies":

    st.title("Explore Movies")

    st.write(
        "Search movies using keywords or choose a category."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Popular Categories")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Action"):
            st.session_state.keyword = "action"

    with col2:
        if st.button("Adventure"):
            st.session_state.keyword = "adventure"

    with col3:
        if st.button("Comedy"):
            st.session_state.keyword = "comedy"

    with col4:
        if st.button("Romance"):
            st.session_state.keyword = "romance"

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        if st.button("Horror"):
            st.session_state.keyword = "horror"

    with col6:
        if st.button("Thriller"):
            st.session_state.keyword = "thriller"

    with col7:
        if st.button("Sci-Fi"):
            st.session_state.keyword = "science fiction"

    with col8:
        if st.button("Animation"):
            st.session_state.keyword = "animation"

    st.markdown("<br>", unsafe_allow_html=True)

    search = st.text_input(
        "Search by Genre or Keyword",
        value=st.session_state.get("keyword", "")
    )

    if search:
        results = movies[
            movies["tags"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

        st.success(f"{len(results)} movies found")
        st.markdown("<br>", unsafe_allow_html=True)

        for _, row in results.head(30).iterrows():
            preview = " ".join(row["tags"].split()[:25])

            st.markdown(f"""
            <div class="glass">
            <h3>{row['title']}</h3>
            <hr>
            <p>
            <b>Keywords</b><br><br>
            {preview}...
            </p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info(
            "Search for Action, Horror, Comedy, Romance, Adventure, Sci-Fi and many more..."
        )

elif page == "Surprise Me":
    st.title("🎲 Surprise Me")
    st.write("Can't decide what to watch? Let the system pick a random movie for you.")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🎬 Pick a Random Movie"):
        random_movie = movies.sample(1).iloc[0]
        preview = " ".join(random_movie["tags"].split()[:35])
        st.markdown(f"""
        <div class="glass">
        <h2>{random_movie['title']}</h2>
        <hr>
        <p>
        <b>Movie Keywords</b><br><br>
        {preview}...
        </p>
        </div>
        """, unsafe_allow_html=True)

elif page == "Statistics":

    st.title("📊 Movie Statistics")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Movies", len(movies))

    with col2:
        total_keywords = len(set(" ".join(movies["tags"]).split()))
        st.metric("Unique Keywords", total_keywords)

    with col3:
        st.metric("Recent Searches", len(st.session_state.history))

    st.markdown("<br>", unsafe_allow_html=True)

    all_words = []
    for text in movies["tags"]:
        all_words.extend(text.split())

    common = Counter(all_words).most_common(10)

    keyword_df = pd.DataFrame(
        common,
        columns=["Keyword", "Count"]
    )

    fig = px.bar(
        keyword_df,
        x="Keyword",
        y="Count",
        title="Top 10 Most Common Keywords"
    )

    st.plotly_chart(fig, use_container_width=True)


    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Recent Searches")
    if len(st.session_state.history) == 0:
        st.info("No searches yet.")

    else:

        history_df = pd.DataFrame(
            {
                "Movie": st.session_state.history[::-1]
            }
        )
        st.dataframe(
            history_df,
            use_container_width=True
        )

elif page == "About":

    st.title("About Project")
    st.markdown("""
    <div class="glass">
    <h2>Movie Recommendation System</h2>
    <p style="font-size:18px;line-height:1.8;">
    This application is a Content-Based Movie Recommendation System
    developed using Machine Learning and Natural Language Processing.
    The system recommends movies based on their similarities using
    CountVectorizer and Cosine similarities.
    Users can also explore movies using keywords like Action,
    Horror, Romance, Adventure, Comedy and many more.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="glass">
        <h3>Technologies Used</h3>
        ✔ Python
        <br>✔ Streamlit
        <br>✔ Pandas
        <br>✔ Scikit-learn
        <br>✔ NLP
        <br>✔ CountVectorizer
        <br>✔ Cosine similarities
        <br>✔ Pickle
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="glass">
        <h3>Features</h3>
        ✔ Similar Movie Recommendation
        <br>✔ Explore Movies
        <br>✔ Keyword Search
        <br>✔ Surprise Me
        <br>✔ Statistics Dashboard
        <br>✔ Search History
        <br>✔ Modern Animated UI
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass">
    <h3>Recommendation Technique</h3>
    <p style="font-size:17px;line-height:1.8;">
    • Data Cleaning
    <br>• Text Preprocessing
    <br>• Lemmatization
    <br>• CountVectorizer
    <br>• Cosine similarities
    <br>• Content-Based Filtering
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass" style="text-align:center;">
    <h2>Developer</h2>
    <h3>Sakshi Anil Gharge</h3>
    <p>
    BCA Student | AI & Machine Learning Enthusiast
    </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
<hr>
Movie Recommendation System
<br>
Developed by <b>Diksha Dilip Patil</b>
</div>
""", unsafe_allow_html=True)
