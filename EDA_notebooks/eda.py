import pandas as pd

df = pd.read_csv("datasets/ncaa_d1_player_deltas_cleaned.csv")    

df['conferences'] = df.apply(
    lambda row: 'Big Ten' if row['team'] in ['Penn State University Nittany Lions'] else row['conference'],
    axis=1
)

print(df.head(25))