import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# not all models complete only first 4
models = (
    "Course Similarity",  # models[0]
    "User Profile",       # models[1]
    "Clustering",         # models[2]
    "Clustering with PCA" # models[3] 
    #  "KNN",
    #  "NMF",
    #  "Neural Network",
    #  "Regression with Embedding Features",
    #  "Classification with Embedding Features"
)
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


def train(model_name, params):
    """
    Trains the clustering models; the content based models have nothing to fit.
    """
    cluster_no = params.get("cluster_no", 5)

    if model_name == models[2]:
        return cluster_users(load_profile(), cluster_no)
    elif model_name == models[3]:
        return cluster_users(load_profile(), cluster_no, params.get("pca_components", 2))

def predict(model_name, user_ids, params):
   
    # primary prediction function for all models
    # 1 for course Sim: find similar courses
    # 2 for User Profile: uses a dot-product approach with user and course vectors
    # 3 for clustering: returns the cluster of a specific user or set of users
    # 4 for clustering with PCA: same as above but after training with PCA
  
    sim_threshold = 0.6
    profile_sim_threshold = 10.0
    # convert relevant params
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0
    if "profile_sim_threshold" in params:
        profile_sim_threshold = params["profile_sim_threshold"]

    cluster_no = params.get("cluster_no", 5)
    temp_user_two = params.get("temp_user_two", None)

    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    res_df = None

    # Loop over each user in user_ids 
    for user_id in user_ids:
        if model_name == models[0]:
            # --- course similarity ---
            enrolled_course_ids = list(params.get('enrolled_courses', []))

            # dict of course->similarity
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)

            # filter  similarity threshold
            users, courses, scores = [], [], []
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)

            
            res_dict = {
                'USER': users,
                'COURSE_ID': courses,
                'SCORE': scores
            }
            res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])

        elif model_name == models[1]:
            # --- User Profile ---
            # Use user_id from params if needed
            temp_user = params.get('user_id', user_id)
            temp_user = int(temp_user)
            ratings_df = load_ratings()
            profile_df = load_profile()
            course_genres_df = load_courses_genre()

            all_courses = set(course_genres_df['COURSE_ID'].values)

            # get the feature vector for the user
            test_user_profile = profile_df[profile_df['user'] == temp_user]
            test_user_vector = test_user_profile.iloc[0, 1:].values

            # courses the user hasn't taken
            enrolled_courses = ratings_df[ratings_df['user'] == temp_user]['item'].to_list()
            unknown_courses = all_courses.difference(enrolled_courses)
            unknown_course_df = course_genres_df[course_genres_df['COURSE_ID'].isin(unknown_courses)]
            unknown_course_ids = unknown_course_df['COURSE_ID'].values
            course_matrix = unknown_course_df.iloc[:, 2:].values  # genre columns only, skipping COURSE_ID and TITLE

            # score by dot product and keep what clears profile_sim_threshold
            courses, scores = user_profile_recommendations(
                unknown_course_ids, course_matrix, test_user_vector, profile_sim_threshold
            )

            res_dict = {
                'COURSE_ID': courses,
                'SCORE': scores
            }
            res_df = pd.DataFrame(res_dict, columns=['COURSE_ID', 'SCORE'])


           # --- Clustering (No PCA) ---
        elif model_name == models[2]:
            #  re-train the model in the train function
            trained_df = train(model_name, params)  # returns user->cluster

            # temp_user_two is the user ID we want to see the cluster for
            if temp_user_two is not None:
                temp_user_two = int(temp_user_two)
                filt = trained_df['user'] == temp_user_two
                cluster_value = int(trained_df[filt]['cluster'].iloc[0])
                #  all users in that cluster
                res_df = trained_df[trained_df['cluster'] == cluster_value]


# --- clustering w PCA ---
        elif model_name == models[3]:
            trained_df = train(model_name, params)

            if temp_user_two is not None:
                temp_user_two = int(temp_user_two)
                filt = trained_df['user'] == temp_user_two
                cluster_value = int(trained_df[filt]['cluster'].iloc[0])
                # get all users in that cluster
                res_df = trained_df[trained_df['cluster'] == cluster_value]
        break

    return res_df
