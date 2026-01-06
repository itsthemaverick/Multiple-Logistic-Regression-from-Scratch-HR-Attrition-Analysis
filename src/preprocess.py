import pandas as pd
from sklearn.preprocessing import StandardScaler
def preprocess(df):
    df = df.copy()

    categorical_cols = ['Department','salary']
    df = pd.get_dummies(df,columns=categorical_cols,drop_first=True)
    
    X_unscaled = df.drop('left',axis=1)
    y = df['left']

    scalar = StandardScaler()
    X = scalar.fit_transform(X_unscaled)

    X = pd.DataFrame(X,columns=X_unscaled.columns)

    return X,y