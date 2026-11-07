import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

COURSE_SIMILARITY = "Course Similarity"
USER_PROFILE = "User Profile"
CLUSTERING = "Clustering"
CLUSTERING_PCA = "Clustering with PCA"

# Four of the nine notebook techniques are wired into the app. KNN, NMF, the
# neural network, and the embedding based regression and classification models
# only exist in the notebooks so far.
models = (COURSE_SIMILARITY, USER_PROFILE, CLUSTERING, CLUSTERING_PCA)

def load_ratings():
    return pd.read_csv("ratings.csv")

def load_course_sims():
    return pd.read_csv("sim.csv")

def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df

def load_profile():
    return pd.read_csv("user_profile.csv")
    
def load_courses_genre():
    return pd.read_csv("course_genre.csv")

def load_bow():
    """
    Loads the Bag-of-Words text features for each course.
    """
    return pd.read_csv("courses_bows.csv")

def get_doc_dicts():
    """
    Creates dictionaries mapping doc indices to doc IDs and vice versa for  Bag-of-Words data
    """
    bow_df = load_bow()
    # group by doc_index, doc_id to ensure there is a unique row
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    
    del grouped_df
    return idx_id_dict, id_idx_dict



def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    # the courses the user hasn't enrolled in yet
    unselected_course_ids = all_courses.difference(enrolled_course_ids)

    # This dict will store course_id -> best similarity
    res = {}

    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                # Keep track of the highest similarity for each unselect course
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim



    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res



def user_profile_recommendations(course_ids, course_matrix, user_vector, threshold):
    """
    Scores candidate courses against a user profile vector.

    Each course row is dotted with the user's genre weights, so a course whose
    genres line up with what the user already likes scores higher. Only courses
    at or above `threshold` are kept.

    Returns two parallel lists: the surviving course ids and their scores.
    """
    if len(course_ids) == 0:
        return [], []

    course_matrix = np.asarray(course_matrix, dtype=float)
    user_vector = np.asarray(user_vector, dtype=float)

    recommendation_scores = np.dot(course_matrix, user_vector)

    courses, scores = [], []
    for i in range(len(course_ids)):
        score = recommendation_scores[i]
        if score >= threshold:
            courses.append(course_ids[i])
            scores.append(score)
    return courses, scores


def combine_cluster_labels(user_ids, labels):
    """
    Pairs each user ID with its K-Means cluster label.

    KMeans returns labels positionally, so the join is on row position rather
    than on a key, which is why the user IDs must arrive in their original order.
    """
    labels_df = pd.DataFrame(labels)
    cluster_df = pd.merge(user_ids, labels_df, left_index=True, right_index=True)
    cluster_df.columns = ['user', 'cluster']
    return cluster_df

def cluster_users(profile_df, n_clusters, pca_components=None):
    """
    Groups users with similar genre interests using K-Means.

    1 standardize every feature column (all columns except `user`)
    2 optionally reduce them to `pca_components` dimensions with PCA
    3 fit K-Means and return a DF mapping each user to a cluster label
    """
    features = StandardScaler().fit_transform(profile_df.drop(columns='user'))
    if pca_components is not None:
        features = PCA(n_components=pca_components, random_state=42).fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=42).fit(features)
    return combine_cluster_labels(profile_df[['user']], labels=km.labels_)


def predict_course_similarity(enrolled_course_ids, sim_threshold, top_courses):
    """
    Ranks the courses most similar to the ones selected, best first.
    """
    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)

    # res is sorted best first, so stop at the threshold or at top_courses
    courses, scores = [], []
    for course_id, score in res.items():
        if score < sim_threshold or len(courses) >= top_courses:
            break
        courses.append(course_id)
        scores.append(score)
    return pd.DataFrame({'COURSE_ID': courses, 'SCORE': scores})


def predict_user_profile(user_id, threshold):
    """
    Scores every course the user has not taken against their genre profile.
    """
    profile_df = load_profile()
    user_profile = profile_df[profile_df['user'] == user_id]
    if user_profile.empty:
        return pd.DataFrame(columns=['COURSE_ID', 'SCORE'])
    user_vector = user_profile.iloc[0, 1:].values

    ratings_df = load_ratings()
    enrolled_courses = ratings_df.loc[ratings_df['user'] == user_id, 'item']
    course_genres_df = load_courses_genre()
    unknown_course_df = course_genres_df[~course_genres_df['COURSE_ID'].isin(enrolled_courses)]
    course_matrix = unknown_course_df.iloc[:, 2:].values  # genre columns only, skipping COURSE_ID and TITLE

    courses, scores = user_profile_recommendations(
        unknown_course_df['COURSE_ID'].values, course_matrix, user_vector, threshold
    )
    return pd.DataFrame({'COURSE_ID': courses, 'SCORE': scores})


def predict_cluster_peers(model_name, user_id, params):
    """
    Returns every user that lands in the same cluster as `user_id`.
    """
    pca_components = params.get('pca_components', 2) if model_name == CLUSTERING_PCA else None
    cluster_df = cluster_users(load_profile(), params.get('cluster_no', 5), pca_components)
    user_rows = cluster_df[cluster_df['user'] == user_id]
    if user_rows.empty:
        return cluster_df.iloc[0:0]
    return cluster_df[cluster_df['cluster'] == user_rows['cluster'].iloc[0]]


def predict(model_name, params):
    """
    Runs the selected model with the hyperparameters chosen in the sidebar.

    Course Similarity and User Profile return COURSE_ID and SCORE columns; the
    clustering models return the user and cluster of everyone who shares a
    cluster with params['user_id'].
    """
    if model_name == COURSE_SIMILARITY:
        return predict_course_similarity(
            list(params.get('enrolled_courses', [])),
            params.get('sim_threshold', 60) / 100.0,
            params.get('top_courses', 10),
        )
    if model_name == USER_PROFILE:
        return predict_user_profile(int(params['user_id']), params.get('profile_sim_threshold', 10.0))
    if model_name in (CLUSTERING, CLUSTERING_PCA):
        return predict_cluster_peers(model_name, int(params['user_id']), params)
    raise ValueError(f"Unknown model: {model_name}")
