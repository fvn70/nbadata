import pandas as pd
import os
import requests

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


pd.options.display.max_columns = None
df = clean_data(data_path)
print(df[['b_day', 'team', 'height', 'weight', 'country', 'draft_round', 'draft_year', 'salary']].head())
