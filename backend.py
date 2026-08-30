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

def add_new_ratings(new_courses):
    """
    Records the courses a visitor selected under a brand new user ID.

    1. Assigns a new user ID (max existing user + 1)
    2. Attaches the selected courses to that user with a default rating
    3. Saves back to ratings.csv
    """
    res_dict = {}
    if len(new_courses) > 0:
        ratings_df = load_ratings()
        # next user ID
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)

        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings

        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("ratings.csv", index=False)
        return new_id
    


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

def train(model_name, params):
    """
     training function for all models.
    Checks the model_name and hyperparameters in params then trains the appropriate model.

     clustering and c lustering with PCA:
     1 Load user_profile.csv
     2 standardize feature columns
     3 PCA -> optional
     4 fit a K-Means model
     5 return a DF mapping each user to a cluster label
    """
    if "cluster_no" in params:
        cluster_no = params["cluster_no"]
    else:
        cluster_no = 5  # fallback if not provided (may be worth trying 3 or 4 for better results)

    if model_name == models[2]:
        # Basic clustering (no PCA)
        user_profile_df = load_profile()
        scaler = StandardScaler()


        # every column except user is a feature
        feature_names = list(user_profile_df.columns[1:])
        features = user_profile_df.loc[:, user_profile_df.columns != 'user']

        # scale the features in place
        user_profile_df[feature_names] = scaler.fit_transform(user_profile_df[feature_names])
        user_ids = user_profile_df[['user']]

        # fit Kmeans
        km = KMeans(n_clusters=cluster_no, random_state=42)
        km = km.fit(features)
        cluster_labels = km.labels_

        res_df = combine_cluster_labels(user_ids, labels=cluster_labels)
        return res_df



    elif model_name == models[3]:
        user_profile_df = load_profile()
        scaler = StandardScaler()
        pca_components = params.get("pca_components", 2)

        # collect feature names
        feature_names = list(user_profile_df.columns[1:])
        features = user_profile_df.loc[:, user_profile_df.columns != 'user']

        # Scale  features
        user_profile_df[feature_names] = scaler.fit_transform(user_profile_df[feature_names])
        user_ids = user_profile_df[['user']]

        # PCA for dimensionality reduction
        pca = PCA(n_components=pca_components, random_state=42)
        reduced_features = pca.fit_transform(features)

        #  cluster the PCA-reduced features
        km = KMeans(n_clusters=cluster_no, random_state=42)
        km = km.fit(reduced_features)
        cluster_labels = km.labels_

        res_df = combine_cluster_labels(user_ids, labels=cluster_labels)
        return res_df

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
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            
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
