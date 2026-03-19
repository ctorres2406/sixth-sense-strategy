import pandas as pd

df = pd.read_csv('MMasseyOrdinals.csv')

# Keep only the most respected systems
SYSTEMS = ['SAG','BPI','NET','POM','MOR','RPI','ESPN','KPI','AP','COL','DES','DOK','TRK','RTH','WLK']
df = df[df['SystemName'].isin(SYSTEMS)]

# Aggregate to one row per team per season
agg = df.groupby(['Season','TeamID'])['OrdinalRank'].agg(
    avg_rank='mean',
    min_rank='min', 
    std_rank='std',
    num_systems='count'
).reset_index()

agg.to_csv('MMasseyOrdinals_condensed.csv', index=False)
print(f"Done: {len(agg)} rows, {agg['Season'].nunique()} seasons")
