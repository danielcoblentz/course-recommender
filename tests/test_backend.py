"""
Unit tests for the scoring helpers in backend.py.

backend.py does all of its file reading inside functions, so importing it here
is side effect free and the scoring helpers can be exercised on small hand
built fixtures instead of the shipped CSVs.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend  # noqa: E402


# The 14 genre columns, in the order used by course_genre.csv and user_profile.csv.
GENRES = [
    "Database", "Python", "CloudComputing", "DataAnalysis", "Containers",
    "MachineLearning", "ComputerVision", "DataScience", "BigData", "Chatbot",
    "R", "BackendDev", "FrontendDev", "Blockchain",
]

PYTHON = GENRES.index("Python")
MACHINE_LEARNING = GENRES.index("MachineLearning")
BLOCKCHAIN = GENRES.index("Blockchain")


def genre_vector(**weights):
    """Builds a 14 slot genre row, defaulting every unnamed genre to zero."""
    vec = np.zeros(len(GENRES))
    for name, value in weights.items():
        vec[GENRES.index(name)] = value
    return vec


class TestUserProfileRecommendations:
    """The dot product scoring behind the User Profile model."""

    def test_matching_course_outscores_unrelated_course(self):
        # A user who only ever takes Python and machine learning courses.
        user = genre_vector(Python=40.0, MachineLearning=35.0)

        matching = genre_vector(Python=1, MachineLearning=1)
        unrelated = genre_vector(Blockchain=1)

        course_ids = ["MATCH", "UNRELATED"]
        matrix = np.vstack([matching, unrelated])

        courses, scores = backend.user_profile_recommendations(
            course_ids, matrix, user, threshold=0.0
        )

        by_id = dict(zip(courses, scores))
        assert by_id["MATCH"] == pytest.approx(75.0)
        assert by_id["UNRELATED"] == pytest.approx(0.0)
        assert by_id["MATCH"] > by_id["UNRELATED"]

    def test_threshold_filters_out_weak_matches(self):
        user = genre_vector(Python=40.0)
        matrix = np.vstack([
            genre_vector(Python=1),      # scores 40
            genre_vector(Blockchain=1),  # scores 0
        ])

        courses, scores = backend.user_profile_recommendations(
            ["STRONG", "WEAK"], matrix, user, threshold=10.0
        )

        assert courses == ["STRONG"]
        assert scores == pytest.approx([40.0])

    def test_scoring_is_deterministic_for_identical_input(self):
        user = genre_vector(Python=12.0, DataScience=7.0, R=3.0)
        matrix = np.vstack([
            genre_vector(Python=1, DataScience=1),
            genre_vector(R=1),
            genre_vector(Blockchain=1, Containers=1),
        ])
        course_ids = ["A", "B", "C"]

        first = backend.user_profile_recommendations(course_ids, matrix, user, 0.0)
        second = backend.user_profile_recommendations(course_ids, matrix, user, 0.0)

        assert first[0] == second[0]
        assert first[1] == pytest.approx(second[1])

    def test_all_zero_profile_does_not_crash(self):
        # A brand new user with no history yet: every genre weight is zero.
        user = genre_vector()
        matrix = np.vstack([genre_vector(Python=1), genre_vector(Blockchain=1)])

        courses, scores = backend.user_profile_recommendations(
            ["A", "B"], matrix, user, threshold=10.0
        )

        # Nothing clears a positive threshold, but the call still returns cleanly.
        assert courses == []
        assert scores == []

    def test_empty_candidate_set_does_not_crash(self):
        user = genre_vector(Python=40.0)
        courses, scores = backend.user_profile_recommendations([], [], user, 0.0)
        assert courses == []
        assert scores == []


class TestCourseSimilarityRecommendations:
    """The similarity lookup behind the Course Similarity model."""

    @staticmethod
    def fixture():
        # Three courses; index 0 is very similar to index 1 and unlike index 2.
        idx_id_dict = {0: "PY0101EN", 1: "PY0201EN", 2: "BC0101EN"}
        id_idx_dict = {v: k for k, v in idx_id_dict.items()}
        sim_matrix = np.array([
            [1.00, 0.90, 0.05],
            [0.90, 1.00, 0.02],
            [0.05, 0.02, 1.00],
        ])
        return idx_id_dict, id_idx_dict, sim_matrix

    def test_similar_course_outscores_unrelated_course(self):
        idx_id_dict, id_idx_dict, sim_matrix = self.fixture()

        res = backend.course_similarity_recommendations(
            idx_id_dict, id_idx_dict, ["PY0101EN"], sim_matrix
        )

        assert res["PY0201EN"] > res["BC0101EN"]
        # Results come back sorted by descending similarity.
        assert list(res.keys()) == ["PY0201EN", "BC0101EN"]

    def test_enrolled_courses_are_not_recommended_back(self):
        idx_id_dict, id_idx_dict, sim_matrix = self.fixture()

        res = backend.course_similarity_recommendations(
            idx_id_dict, id_idx_dict, ["PY0101EN"], sim_matrix
        )

        assert "PY0101EN" not in res

    def test_is_deterministic_for_identical_input(self):
        idx_id_dict, id_idx_dict, sim_matrix = self.fixture()

        first = backend.course_similarity_recommendations(
            idx_id_dict, id_idx_dict, ["PY0101EN"], sim_matrix
        )
        second = backend.course_similarity_recommendations(
            idx_id_dict, id_idx_dict, ["PY0101EN"], sim_matrix
        )

        assert list(first.keys()) == list(second.keys())
        assert list(first.values()) == pytest.approx(list(second.values()))

    def test_no_enrolled_courses_returns_nothing(self):
        idx_id_dict, id_idx_dict, sim_matrix = self.fixture()

        res = backend.course_similarity_recommendations(
            idx_id_dict, id_idx_dict, [], sim_matrix
        )

        assert res == {}


class TestCombineClusterLabels:
    """Cluster assignment bookkeeping shared by both clustering models."""

    def test_maps_each_user_to_its_label(self):
        user_ids = pd.DataFrame({"user": [2, 4, 7, 9]})
        labels = np.array([0, 1, 0, 2])

        res = backend.combine_cluster_labels(user_ids, labels)

        assert list(res.columns) == ["user", "cluster"]
        assert res["user"].tolist() == [2, 4, 7, 9]
        assert res["cluster"].tolist() == [0, 1, 0, 2]

    def test_handles_a_single_user(self):
        res = backend.combine_cluster_labels(pd.DataFrame({"user": [2]}), np.array([3]))

        assert len(res) == 1
        assert res.iloc[0]["cluster"] == 3


class TestClusterUsers:
    """K-Means grouping behind both clustering models."""

    @staticmethod
    def profiles():
        # Two clear interest groups split on Python, plus a Database column with
        # far larger raw values that is pure noise and carries no group signal.
        rng = np.random.default_rng(0)
        n = 40
        python = np.r_[np.full(n // 2, 1.0), np.full(n // 2, 3.0)] + rng.normal(0, 0.05, n)
        database = rng.normal(500, 60, n)
        return pd.DataFrame({"user": range(n), "Python": python, "Database": database})

    @staticmethod
    def recovers_python_split(res):
        first, second = res["cluster"].iloc[:20], res["cluster"].iloc[20:]
        return first.nunique() == 1 and second.nunique() == 1 and first.iloc[0] != second.iloc[0]

    def test_features_are_standardized_before_clustering(self):
        # Unscaled, the noisy Database column would swamp the Python split.
        res = backend.cluster_users(self.profiles(), n_clusters=2)
        assert self.recovers_python_split(res)

    def test_pca_variant_returns_one_label_per_user(self):
        df = self.profiles()
        res = backend.cluster_users(df, n_clusters=2, pca_components=2)

        assert res["user"].tolist() == df["user"].tolist()
        assert set(res["cluster"]) == {0, 1}

    def test_is_deterministic(self):
        df = self.profiles()
        first = backend.cluster_users(df, n_clusters=3, pca_components=2)
        second = backend.cluster_users(df, n_clusters=3, pca_components=2)
        assert first["cluster"].tolist() == second["cluster"].tolist()


def test_predict_rejects_an_unknown_model():
    with pytest.raises(ValueError):
        backend.predict("Not A Model", {})
