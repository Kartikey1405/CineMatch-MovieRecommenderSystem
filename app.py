
import streamlit as st
import pandas as pd

from recommender import MovieRecommender


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CineMatch | Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# 2. CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 1.5rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #171923, #29354d);
        color: white;
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.3rem;
    }

    .hero p {
        margin-top: 0.5rem;
        color: #d7deea;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 12px 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. LOAD THE TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_recommender():
    return MovieRecommender("models")


try:
    recommender = load_recommender()

except Exception as error:
    st.error("Unable to load the recommendation model.")

    st.write(
        "Check the six artifact files inside your models folder."
    )

    st.exception(error)
    st.stop()


movies = recommender.movies
ratings = recommender.ratings
metadata = recommender.metadata


# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.caption("Personalized movie discovery")

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "For You",
            "Explore Similar Movies",
            "Predict a Rating",
            "Model Evaluation",
        ],
    )

    st.divider()

    st.caption(
        "Recommendation engine: SVD collaborative filtering"
    )


# --------------------------------------------------
# 5. COMMON HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🎬 CineMatch</h1>
        <p>
            Discover movies tailored to your taste
            using patterns in movie ratings.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 6. OVERVIEW PAGE
# --------------------------------------------------

if page == "Overview":

    st.subheader("Your movie discovery dashboard")

    st.write(
        "CineMatch uses a trained Singular Value Decomposition "
        "(SVD) collaborative-filtering model to estimate ratings "
        "and rank movies that a user has not rated in the "
        "available modelling sample."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Movies in catalogue",
        f"{len(movies):,}"
    )

    col2.metric(
        "Users in model sample",
        f"{ratings['userId'].nunique():,}"
    )

    col3.metric(
        "Ratings in model sample",
        f"{len(ratings):,}"
    )

    st.markdown("### How the system works")

    st.markdown(
        """
        1. **Data preparation:** Clean movie metadata and ratings.
        2. **Model training:** Train and compare recommendation algorithms.
        3. **Collaborative filtering:** Learn latent user and movie factors.
        4. **Recommendation:** Predict ratings for unseen movies.
        5. **Evaluation:** Compare prediction errors and ranking metrics.
        """
    )

    with st.expander("View model and dataset metadata"):
        st.json(metadata)

    st.info(
        "The available ratings file contains fewer records than "
        "the canonical MovieLens 20M dataset. Use the actual "
        "dataset counts when presenting this project."
    )


# --------------------------------------------------
# 7. PERSONALIZED RECOMMENDATIONS
# --------------------------------------------------

elif page == "For You":

    st.subheader("Personalized movie recommendations")

    valid_users = sorted(recommender.user_ids)

    default_user = (
        int(valid_users[0]) if valid_users else 1
    )

    user_id = st.number_input(
        "Enter a user ID from the training sample",
        min_value=1,
        value=default_user,
        step=1,
        key="recommend_user",
    )

    available_genres = sorted(
        {
            genre
            for value in movies["genres"].fillna("")
            for genre in str(value).split("|")
            if genre and genre != "(no genres listed)"
        }
    )

    selected_genre = st.selectbox(
        "Filter by genre",
        ["All genres"] + available_genres,
    )

    number_of_movies = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=30,
        value=10,
    )

    if st.button(
        "✨ Generate recommendations",
        type="primary",
        use_container_width=True,
    ):

        try:
            recommendations = recommender.recommend_movies(
                user_id=int(user_id),
                n=number_of_movies,
                genre=(
                    None
                    if selected_genre == "All genres"
                    else selected_genre
                ),
            )

            if recommendations.empty:
                st.warning(
                    "No unseen movies matched the selected filters."
                )

            else:
                st.success(
                    f"Found {len(recommendations)} recommendations."
                )

                for start in range(
                    0, len(recommendations), 3
                ):

                    columns = st.columns(3)

                    batch = recommendations.iloc[
                        start:start + 3
                    ]

                    for column, (_, movie) in zip(
                        columns, batch.iterrows()
                    ):

                        with column:
                            st.markdown(
                                "#### " + str(movie["title"])
                            )

                            st.caption(
                                str(movie.get("genres", ""))
                                .replace("|", " · ")
                            )

                            st.metric(
                                "Predicted rating",
                                f"{movie['predicted_rating']:.2f} / 5"
                            )

        except Exception as error:
            st.error(str(error))

    with st.expander("View this user's highest-rated movies"):

        history_user = st.number_input(
            "History user ID",
            min_value=1,
            value=default_user,
            step=1,
            key="history_user",
        )

        history = recommender.user_history(
            int(history_user)
        )

        if history.empty:
            st.info(
                "No ratings found for this user in the model sample."
            )

        else:
            st.dataframe(
                history[["title", "rating"]].head(20),
                use_container_width=True,
                hide_index=True,
            )


# --------------------------------------------------
# 8. SIMILAR MOVIES
# --------------------------------------------------

elif page == "Explore Similar Movies":

    st.subheader("Explore movies by genre")

    sorted_movies = (
        movies.sort_values("title")
        .reset_index(drop=True)
    )

    movie_titles = sorted_movies[
        "title"
    ].astype(str).tolist()

    selected_title = st.selectbox(
        "Choose a movie",
        movie_titles,
    )

    selected_movie = sorted_movies[
        sorted_movies["title"].astype(str)
        == selected_title
    ].iloc[0]

    st.write(
        "**Genres:** "
        + str(selected_movie["genres"]).replace("|", " · ")
    )

    number_of_similar = st.slider(
        "Number of similar movies",
        min_value=5,
        max_value=20,
        value=10,
    )

    if st.button(
        "Find similar movies",
        type="primary",
    ):

        similar_movies = (
            recommender.similar_movies_by_genre(
                int(selected_movie["movieId"]),
                number_of_similar,
            )
        )

        if similar_movies.empty:
            st.info(
                "No usable genre information is available."
            )

        else:
            display = similar_movies[
                ["title", "genres", "genre_similarity"]
            ].copy()

            display["genre_similarity"] = (
                display["genre_similarity"] * 100
            ).round(1)

            display = display.rename(
                columns={
                    "title": "Movie",
                    "genres": "Genres",
                    "genre_similarity": "Genre overlap (%)",
                }
            )

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
            )

    st.caption(
        "This feature uses Jaccard similarity between genre sets. "
        "It is a simple content-based feature, not a learned "
        "collaborative item-item model."
    )


# --------------------------------------------------
# 9. PREDICT A RATING
# --------------------------------------------------

elif page == "Predict a Rating":

    st.subheader("Estimate a user's movie rating")

    valid_users = sorted(recommender.user_ids)

    default_user = (
        int(valid_users[0]) if valid_users else 1
    )

    user_id = st.number_input(
        "User ID",
        min_value=1,
        value=default_user,
        step=1,
        key="prediction_user",
    )

    sorted_movies = (
        movies.sort_values("title")
        .reset_index(drop=True)
    )

    selected_title = st.selectbox(
        "Choose a movie",
        sorted_movies["title"].astype(str).tolist(),
        key="prediction_movie",
    )

    selected_movie = sorted_movies[
        sorted_movies["title"].astype(str)
        == selected_title
    ].iloc[0]

    if st.button(
        "Predict rating",
        type="primary",
    ):

        try:
            result = recommender.predict_rating(
                int(user_id),
                int(selected_movie["movieId"]),
            )

            st.metric(
                "Estimated rating",
                f"{result['estimated_rating']:.2f} / 5.0",
            )

            st.write(
                f"**Movie:** {result['title']}"
            )

            st.caption(
                "This is the model's estimate, not an actual user rating."
            )

        except Exception as error:
            st.error(str(error))


# --------------------------------------------------
# 10. MODEL EVALUATION
# --------------------------------------------------

elif page == "Model Evaluation":

    st.subheader("Model evaluation")

    comparison = recommender.get_model_comparison()

    st.markdown("### Prediction model comparison")

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

    rmse_column = next(
        (
            column
            for column in comparison.columns
            if column.lower() == "rmse"
        ),
        None,
    )

    model_column = next(
        (
            column
            for column in comparison.columns
            if column.lower()
            in ("model", "algorithm", "model_name")
        ),
        None,
    )

    if rmse_column and model_column:

        plot_data = comparison[
            [model_column, rmse_column]
        ].copy()

        plot_data[rmse_column] = pd.to_numeric(
            plot_data[rmse_column],
            errors="coerce",
        )

        plot_data = plot_data.dropna()

        if not plot_data.empty:
            st.markdown(
                "#### RMSE comparison (lower is better)"
            )

            st.bar_chart(
                plot_data.set_index(model_column)[rmse_column]
            )

    st.markdown("### Ranking metrics")

    ranking_metrics = (
        recommender.get_ranking_metrics()
    )

    st.dataframe(
        ranking_metrics,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Ranking metrics depend on the candidate set and relevance "
        "threshold used during evaluation. Interpret them alongside "
        "the evaluation methodology."
    )


# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "CineMatch | Academic project prototype | "
    "Predicted ratings are estimates, not guarantees."
)
