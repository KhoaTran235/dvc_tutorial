from sklearn.model_selection import train_test_split

def split(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
    X (array-like): The input features.
    y (array-like): The target variable.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Controls the randomness of the split.

    Returns:
    X_train, X_test, y_train, y_test: The split datasets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)