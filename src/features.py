import re
import numpy as np
import pandas as pd


DECK_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
TITLE_MAP = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
GENDER_MAP = {"male": 0, "female": 1}
PORT_MAP = {"S": 0, "C": 1, "Q": 2}

RARE_TITLES = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
               'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']

FEATURE_COLUMNS = [
    'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
    'Sex', 'Title', 'Deck', 'relatives', 'not_alone', 'Age_Class', 'Fare_Per_Person'
]


def engineer_features(df: pd.DataFrame, train_age_mean: float = None, train_age_std: float = None) -> pd.DataFrame:
    df = df.copy()

    # Drop unused columns
    for col in ['PassengerId', 'Ticket']:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Deck from Cabin
    df['Cabin'] = df['Cabin'].fillna("U0")
    df['Deck'] = df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    df['Deck'] = df['Deck'].map(DECK_MAP).fillna(0).astype(int)
    df = df.drop('Cabin', axis=1)

    # Age imputation
    mean = train_age_mean if train_age_mean is not None else df['Age'].mean()
    std = train_age_std if train_age_std is not None else df['Age'].std()
    is_null = df['Age'].isnull().sum()
    if is_null > 0:
        rand_age = np.random.randint(mean - std, mean + std, size=is_null)
        age_slice = df['Age'].copy()
        age_slice[np.isnan(age_slice)] = rand_age
        df['Age'] = age_slice
    df['Age'] = df['Age'].astype(int)

    # Embarked
    df['Embarked'] = df['Embarked'].fillna('S').map(PORT_MAP)

    # Fare
    df['Fare'] = df['Fare'].fillna(0).astype(int)

    # Title
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(RARE_TITLES, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    df['Title'] = df['Title'].map(TITLE_MAP).fillna(0)
    df = df.drop('Name', axis=1)

    # Sex
    df['Sex'] = df['Sex'].map(GENDER_MAP)

    # Age bins
    df.loc[df['Age'] <= 11, 'Age'] = 0
    df.loc[(df['Age'] > 11) & (df['Age'] <= 18), 'Age'] = 1
    df.loc[(df['Age'] > 18) & (df['Age'] <= 22), 'Age'] = 2
    df.loc[(df['Age'] > 22) & (df['Age'] <= 27), 'Age'] = 3
    df.loc[(df['Age'] > 27) & (df['Age'] <= 33), 'Age'] = 4
    df.loc[(df['Age'] > 33) & (df['Age'] <= 40), 'Age'] = 5
    df.loc[df['Age'] > 40, 'Age'] = 6
    df['Age'] = df['Age'].astype(int)

    # Fare bins
    df.loc[df['Fare'] <= 7, 'Fare'] = 0
    df.loc[(df['Fare'] > 7) & (df['Fare'] <= 14), 'Fare'] = 1
    df.loc[(df['Fare'] > 14) & (df['Fare'] <= 31), 'Fare'] = 2
    df.loc[(df['Fare'] > 31) & (df['Fare'] <= 99), 'Fare'] = 3
    df.loc[(df['Fare'] > 99) & (df['Fare'] <= 250), 'Fare'] = 4
    df.loc[df['Fare'] > 250, 'Fare'] = 5
    df['Fare'] = df['Fare'].astype(int)

    # Family features
    df['relatives'] = df['SibSp'] + df['Parch']
    df['not_alone'] = (df['relatives'] == 0).astype(int)

    # Interaction features
    df['Age_Class'] = df['Age'] * df['Pclass']
    df['Fare_Per_Person'] = (df['Fare'] / (df['relatives'] + 1)).astype(int)

    return df[FEATURE_COLUMNS]
