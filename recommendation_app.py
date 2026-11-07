import streamlit as st
import pandas as pd
import backend as backend
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

# Basic webpage setups
st.set_page_config(
   page_title="Course Recommender System",
   layout="wide",
   initial_sidebar_state="expanded",
)

@st.cache_data
def load_ratings():
    return backend.load_ratings()

@st.cache_data
def load_course_sims():
    return backend.load_course_sims()

@st.cache_data
def load_profile():
    return backend.load_profile()

@st.cache_data
def load_courses():
    return backend.load_courses()

@st.cache_data
def load_courses_genre():
    return backend.load_courses_genre()

@st.cache_data
def load_bow():
    return backend.load_bow()

def init_recommender_app():
    """
     function loads all datasets once, returns a DataFrame of the selected courses
    """
    with st.spinner('Loading datasets...'):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_bow_df = load_bow()
        profile_df = load_profile()
        course_genre_df = load_courses_genre()

    st.success('Datasets loaded successfully..')
    st.markdown("---")
    st.subheader("Select courses that you have audited or completed:")



    # Build an interactive table for `course_df`
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()



    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )

    results = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
    results = results[['COURSE_ID', 'TITLE']]
    st.subheader("Your courses:")
    st.table(results)
    return results



def predict(model_name, params):
    with st.spinner(f'Generating {model_name} recommendations...'):
        res = backend.predict(model_name, params)
    st.success('Recommendations generated!')
    return res

# --------------- MAIN UI ---------------
st.sidebar.title('Personalized Learning Recommender')

selected_courses_df = init_recommender_app()

st.sidebar.subheader('1. Select Recommendation Model')
model_selection = st.sidebar.selectbox(
    "Select model:",
    backend.models  
)

params = {}
st.sidebar.subheader('2. Tune Hyper-parameters:')

if model_selection == backend.COURSE_SIMILARITY:
    params['top_courses'] = st.sidebar.slider('Top courses', min_value=1, max_value=100, value=10, step=1)
    params['sim_threshold'] = st.sidebar.slider('Course Similarity Threshold %',
                                                min_value=0, max_value=100, value=50, step=10)

elif model_selection == backend.USER_PROFILE:
    params['profile_sim_threshold'] = st.sidebar.slider(
        'Minimum profile match score', min_value=0, max_value=100, value=10, step=5,
        help="A course's score is the sum of your profile weights for each of its genres.")

else:  # both clustering models
    params['cluster_no'] = st.sidebar.slider('Number of Clusters', min_value=2, max_value=50, value=5, step=1)
    if model_selection == backend.CLUSTERING_PCA:
        params['pca_components'] = st.sidebar.slider('Number of PCA Components',
                                                     min_value=2, max_value=10, value=2, step=1)

# every model except Course Similarity works from an existing user's profile
if model_selection != backend.COURSE_SIMILARITY:
    user_ids_list = load_profile()['user'].unique().tolist()
    params['user_id'] = st.sidebar.selectbox("Select your user ID", user_ids_list)


# 3. Prediction (the clustering models are fit fresh on each run)
st.sidebar.subheader('3. Prediction')
pred_button = st.sidebar.button("Recommend New Courses")
if pred_button:
    if model_selection == backend.COURSE_SIMILARITY:
        params['enrolled_courses'] = selected_courses_df['COURSE_ID'].tolist()

    if model_selection == backend.COURSE_SIMILARITY and not params['enrolled_courses']:
        st.warning("Select at least one course in the table first.")
    else:
        res_df = predict(model_selection, params)
        if res_df.empty:
            st.info("No results found. Try lowering the threshold.")
        elif model_selection == backend.COURSE_SIMILARITY:
            res_df = pd.merge(res_df, load_courses(), on="COURSE_ID").drop('COURSE_ID', axis=1)
            st.table(res_df)
        else:
            st.table(res_df)
