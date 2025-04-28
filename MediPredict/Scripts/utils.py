def encode_ordinal(X, mapping, columns):
    """Encode ordinal features using a mapping."""
    X = X.copy()
    for col in columns:
        X[col] = X[col].map(mapping)
    return X