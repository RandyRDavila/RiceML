import seaborn as sns
import numpy as np

__all__ = ['iris_data']

def iris_data(features=None, species=None, labels=None):
    """
    Loads the Iris dataset and filters it based on specified features and species, with
    custom labels for species classes.

    Parameters
    ----------
    features : list of str, optional
        List of feature names to include in the returned data. If None, all features
        are included. Valid feature names are: "sepal_length", "sepal_width",
        "petal_length", "petal_width".

    species : list of str, optional
        List of species to filter by. If None, all species are included.
        Valid species are: "setosa", "versicolor", "virginica".

    labels : list of int, optional
        List of integer labels corresponding to the species provided in the `species`
        argument. If None, the function will assign default integer labels starting
        from 0 based on the order of species. The length of `labels` must match
        the length of `species`.

    Returns
    -------
    X : numpy.ndarray
        A NumPy array of shape (n_samples, n_features) containing the filtered feature
        values based on the provided `features` parameter.

    y : numpy.ndarray
        A NumPy array of shape (n_samples,) containing the species labels based on the
        filtered data with custom or default labels.

    Examples
    --------
    >>> X, y = load_iris_data(features=["sepal_length", "sepal_width"], species=["setosa", "versicolor"], labels=[-1, 1])
    >>> X.shape
    (100, 2)
    >>> y
    array([-1, -1, ..., 1, 1])
    """
    # Load the Iris dataset
    iris = sns.load_dataset("iris")

    # Filter by species if specified
    if species:
        iris = iris[iris["species"].isin(species)]
    else:
        species = iris["species"].unique().tolist()

    # Assign default labels if not provided
    if labels is None:
        labels = list(range(len(species)))
    elif len(labels) != len(species):
        raise ValueError("Length of `labels` must match the length of `species`.")

    # Map species to custom labels
    species_to_label = dict(zip(species, labels))
    iris["species"] = iris["species"].map(species_to_label)

    # Select features if specified, otherwise use all feature columns
    if features is None:
        features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # Extract feature data and labels
    X = iris[features].values
    y = iris["species"].values

    return X, y
