import pandas as pd
import os
import requests
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

# Check for ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"

# write your code here
def clean_data(path):
    data = pd.read_csv(path)
    df = pd.DataFrame(data)

    df['b_day'] = [pd.to_datetime(d) for d in df['b_day']]
    df['draft_year'] = [pd.to_datetime(f'{d}-01-01') for d in df['draft_year']]
    df['team'] = df['team'].fillna('No Team')
    df['height'] = [float(s[s.index('/') + 1:]) for s in df['height']]
    df['weight'] = [float(s[s.index('/') + 1:s.index('kg')]) for s in df['weight']]
    df['country'] = [s if s == 'USA' else 'Not-USA' for s in df['country']]
    df['salary'] = [float(s[1:]) for s in df['salary']]
    df['draft_round'] = ['0' if s == 'Undrafted' else s for s in df['draft_round']]

    return df

def feature_data(df):
    df['version'] = [pd.to_datetime('20'+d[-2:]) for d in df.version]
    df['age'] = df.version.dt.year - df.b_day.dt.year
    df['experience'] = df.version.dt.year - df.draft_year.dt.year
    df['bmi'] = df.weight / df.height / df.height
    df = df.drop(columns=['version', 'b_day', 'draft_year', 'weight', 'height'])

    high_card_list = []
    for col in df.columns:
        if df[col].dtype != 'float64' and df[col].unique().size >= 50:
            high_card_list.append(col)
    df = df.drop(columns=high_card_list)

    return df

def multicol_data(df):
    X = df.drop(columns='salary')
    y = df.salary
    m = X.corr(numeric_only=True)
    pairs = []
    for i in range(m.shape[0]):
        for j in range(i + 1, m.shape[0]):
            if abs(m.iloc[i][j]) > 0.5:
                pairs.append((i, j))
    for p in pairs:
        col0 = m.columns[p[0]]
        col1 = m.columns[p[1]]
        r0 = y.corr(X[col0])
        r1 = y.corr(X[col1])
        if r0 < r1:
            df = df.drop(columns=col0)
        else:
            df = df.drop(columns=col1)
    return df

def transform_data(df):
    X = df.drop(columns='salary')
    y = df.salary
    ss = StandardScaler()

    num_feat_df = X.select_dtypes('number')  # numerical features
    num_scale_df = ss.fit_transform(num_feat_df)

    cat_feat_df = X.select_dtypes('object')
    ohe = OneHotEncoder()
    cat_ohe = ohe.fit_transform(cat_feat_df)
    cat = ohe.categories_
    cat_lst = [cat[i][j] for i in range(len(cat)) for j in range(cat[i].size)]

    X = pd.DataFrame(num_scale_df, columns=num_feat_df.columns)
    X[cat_lst] = cat_ohe.toarray()

    return X, y

pd.options.display.max_columns = None
df = clean_data(data_path)
df = feature_data(df)
df = multicol_data(df)
X, y = transform_data(df)

answer = {
    'shape': [X.shape, y.shape],
    'features': list(X.columns),
    }
print(answer)
