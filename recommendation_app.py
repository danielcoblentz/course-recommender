import streamlit as st
import pandas as pd
import time
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



def train(model_name, params):
    """
    Wrapper function that calls the 'train' function in 'backend.py' and handles any UI messages
    """
    with st.spinner(f'Training {model_name} model...'):
        time.sleep(0.5)
        backend.train(model_name, params)
    st.success('Done training!')

def predict(model_name, user_ids, params):
    with st.spinner(f'Generating {model_name} recommendations...'):
        time.sleep(0.5)
        res = backend.predict(model_name, user_ids, params)
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


# Course Similarity
if model_selection == backend.models[0]:
    top_courses = st.sidebar.slider('Top courses', min_value=1, max_value=100, value=10, step=1)
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=10)
    params['top_courses'] = top_courses
    params['sim_threshold'] = course_sim_threshold


# User Profile
elif model_selection == backend.models[1]:
    profile_sim_threshold = st.sidebar.slider('User Profile Similarity Threshold %',
                                              min_value=0, max_value=50,
                                              value=30, step=5)
    profile_df = load_profile()
    user_ids_list = profile_df['user'].unique().tolist()
    temp_user = st.sidebar.selectbox("Select your user ID", user_ids_list)

    params['profile_sim_threshold'] = profile_sim_threshold
    params['user_id'] = temp_user


# Clustering (no PCA)
elif model_selection == backend.models[2]:
    cluster_no = st.sidebar.slider('Number of Clusters', min_value=2, max_value=50, value=5, step=1)
    profile_df = load_profile()
    user_ids_list = profile_df['user'].unique().tolist()
    temp_user_two = st.sidebar.selectbox("Select your user ID to find others in the same cluster",
                                         user_ids_list)
    params['cluster_no'] = cluster_no
    params['temp_user_two'] = temp_user_two


# clustering w PCA
elif model_selection == backend.models[3]:
    cluster_no = st.sidebar.slider('Number of Clusters', min_value=2, max_value=50, value=5, step=1)
    pca_components = st.sidebar.slider('Number of PCA Components', min_value=2, max_value=10, value=2, step=1)
    profile_df = load_profile()
    user_ids_list = profile_df['user'].unique().tolist()
    temp_user_two = st.sidebar.selectbox("Select your user ID to find others in the same cluster (PCA)",user_ids_list)

    # Store these parameters for the backend
    params['cluster_no'] = cluster_no
    params['pca_components'] = pca_components
    params['temp_user_two'] = temp_user_two



# 3. Training -----------
st.sidebar.subheader('3. Training')
training_button = st.sidebar.button("Train Model")
if training_button:
    train(model_selection, params)

# 4. Prediction
st.sidebar.subheader('4. Prediction')
pred_button = st.sidebar.button("Recommend New Courses")
if pred_button:
    # Course sim
    if model_selection == backend.models[0]:
        params['enrolled_courses'] = selected_courses_df['COURSE_ID'].tolist()
        if not params['enrolled_courses']:
            st.warning("Select at least one course in the table first.")
        else:
            res_df = predict(model_selection, [None], params)
            if res_df is not None and not res_df.empty:
                res_df = res_df[['COURSE_ID', 'SCORE']]
                course_df = load_courses()
                res_df = pd.merge(res_df, course_df, on=["COURSE_ID"]).drop('COURSE_ID', axis=1)
                st.table(res_df)
            else:
                st.info("No recommendations found")



    elif model_selection == backend.models[1]:
        #user Profile
        user_ids = [params['user_id']]
        res_df = predict(model_selection, user_ids, params)
        if res_df is not None and not res_df.empty:
            st.table(res_df)
        else:
            st.info("No recommendations found check if user has data.")



    elif model_selection == backend.models[2]:
        # clustering
        user_ids = [params['temp_user_two']]
        res_df = predict(model_selection, user_ids, params)
        if res_df is not None and not res_df.empty:
            st.table(res_df)
        else:
            st.info("No cluster results found are you sure this user exists?")



    elif model_selection == backend.models[3]:
        #clustering w PCA
        user_ids = [params['temp_user_two']]
        res_df = predict(model_selection, user_ids, params)
        if res_df is not None and not res_df.empty:
            st.table(res_df)
        else:
            st.info("No cluster results found are you sure this user exists?")
