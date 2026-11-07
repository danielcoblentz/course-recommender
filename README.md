# Course Recommender

A Streamlit web app that suggests courses from your enrollment history, using content similarity, user profile scoring, and K-Means clustering.

**Live app:** https://ibm-machine-learning-capstone.onrender.com

The Render URL still carries the project's original name.
Free Render instances sleep when idle, so the first load can take around 30 seconds to wake up.

![demo](images/app_demo.png)
<p align="center">Figure 1: Application demo</p>

## Attribution

This project started from the [Coursera IBM Machine Learning Capstone](https://www.coursera.org/learn/machine-learning-capstone/home/module/2).

**Provided by IBM:** all 11 lab notebooks in `notebooks/`, every dataset in the repository root (`ratings.csv`, `course_genre.csv`, `course_processed.csv`, `courses_bows.csv`, `sim.csv`, `user_profile.csv`), and the Streamlit starter scaffolding the app was built on.

**My own work:** the model implementations in `backend.py` and the app in `recommendation_app.py`, specifically:

- User profile scoring, which ranks unseen courses by the dot product of the course genre vector and the user's genre weights.
- Course similarity recommendation over the Bag-of-Words similarity matrix, with a tunable threshold and result count.
- K-Means clustering over user profiles, and a second variant that runs PCA before clustering, both exposed as selectable models with tunable cluster counts.
- The Streamlit sidebar wiring: per-model hyperparameter controls, AgGrid course selection, and the prediction flow.
- The tests in `tests/`.
- The Render deployment.

The notebooks are the IBM coursework and are kept here as reference.
The app does not import them.

## Models

Four of the nine techniques covered in the notebooks are wired into the app:

| Model | Method |
| --- | --- |
| Course Similarity | Bag-of-Words similarity matrix lookup |
| User Profile | Genre vector dot product against the user profile |
| Clustering | K-Means over standardized user profiles |
| Clustering with PCA | PCA dimensionality reduction, then K-Means |

Course Similarity recommends from the courses you tick in the table.
The other three work from an existing user ID in `user_profile.csv`, picked in the sidebar.
The clustering models list the other users in that user's cluster.

The remaining notebook techniques (KNN, NMF, neural network embeddings, regression and classification on embedding features) are implemented in the notebooks but not yet exposed in the app.

## Tech stack

- **Frontend:** Streamlit, [Streamlit-AgGrid](https://github.com/PablocFonseca/streamlit-aggrid)
- **Backend:** Python, Pandas, NumPy
- **Machine learning:** [scikit-learn](https://scikit-learn.org/stable/), [SciPy](https://docs.scipy.org/doc/scipy/)
- **Data:** CSV files (ratings, courses, similarity matrices)
- **Deployment:** [Render](https://render.com)

## Running locally

Requires Python 3.11.
The pinned `numpy==1.23.5` and `pandas==1.5.3` have no wheels for newer versions.

1. Clone the repository:

   ```bash
   git clone https://github.com/danielcoblentz/course-recommender
   cd course-recommender
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run recommendation_app.py
   ```

Streamlit serves the app at http://localhost:8501.
Run it from the repository root, since the loaders in `backend.py` read the CSV files by relative path.

## Tests

`tests/test_backend.py` unit tests the scoring and clustering helpers on small fixtures, and `tests/test_predict.py` runs every model end to end on the shipped CSVs.

```bash
pip install -r requirements-dev.txt
pytest
```

## Notebooks

The `notebooks/` folder holds the IBM lab notebooks with the model definitions, experiments, and summaries.
They are easiest to read in Google Colab.
Some of them use TensorFlow, which is not in `requirements.txt` because the app does not need it.
