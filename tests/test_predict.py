"""
End to end checks of backend.predict() against the CSVs shipped in the repo.

The loaders read by relative path, so every test runs from the repository root.
"""

import hashlib
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import backend  # noqa: E402


@pytest.fixture(autouse=True)
def repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)


SELECTED = ["ML0122EN", "RP0105EN"]


def enrolled_courses(user_id):
    ratings = backend.load_ratings()
    return set(ratings.loc[ratings["user"] == user_id, "item"])


class TestCourseSimilarity:
    def predict(self, **params):
        params = {"enrolled_courses": SELECTED, "sim_threshold": 30, "top_courses": 10, **params}
        return backend.predict(backend.COURSE_SIMILARITY, params)

    def test_returns_ranked_courses_above_the_threshold(self):
        res = self.predict()

        assert list(res.columns) == ["COURSE_ID", "SCORE"]
        assert not res.empty
        assert res["SCORE"].is_monotonic_decreasing
        assert (res["SCORE"] >= 0.30).all()

    def test_never_recommends_a_selected_course(self):
        assert not set(self.predict()["COURSE_ID"]) & set(SELECTED)

    @pytest.mark.parametrize("top", [1, 3, 10])
    def test_top_courses_caps_the_result(self, top):
        assert len(self.predict(top_courses=top)) == top

    def test_does_not_write_to_ratings_csv(self):
        def digest():
            with open("ratings.csv", "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()

        before = digest()
        self.predict()
        assert digest() == before


class TestUserProfile:
    USER = 2

    def predict(self, **params):
        params = {"user_id": self.USER, "profile_sim_threshold": 10, "top_courses": 10, **params}
        return backend.predict(backend.USER_PROFILE, params)

    def test_returns_ranked_courses_capped_at_top_courses(self):
        res = self.predict(top_courses=5)

        assert list(res.columns) == ["COURSE_ID", "SCORE"]
        assert len(res) == 5
        assert res["SCORE"].is_monotonic_decreasing
        assert (res["SCORE"] >= 10).all()

    def test_skips_courses_the_user_already_took(self):
        res = self.predict(top_courses=1000)
        assert not set(res["COURSE_ID"]) & enrolled_courses(self.USER)

    def test_unknown_user_returns_empty(self):
        assert self.predict(user_id=-1).empty


@pytest.mark.parametrize("model, extra", [
    (backend.CLUSTERING, {}),
    (backend.CLUSTERING_PCA, {"pca_components": 3}),
])
class TestClusteringModels:
    def test_returns_the_users_own_cluster(self, model, extra):
        res = backend.predict(model, {"user_id": 2, "cluster_no": 5, **extra})

        assert list(res.columns) == ["user", "cluster"]
        assert 2 in set(res["user"])
        assert res["cluster"].nunique() == 1

    def test_unknown_user_returns_empty(self, model, extra):
        assert backend.predict(model, {"user_id": -1, "cluster_no": 5, **extra}).empty
