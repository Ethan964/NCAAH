import streamlit as st
import pandas as pd

st.set_page_config(page_title='NCAA Hockey Analytics', layout='wide')
st.title("NCAA Men's Hockey Player Analytics")

@st.cache_data
def load_data():
    df = pd.read_csv("datasets/master_player.csv")    
    return df

df = load_data()

st.sidebar.header('Filters')

max_toi = float(df['TOI (sec)'].max() / 60) if 'TOI (sec)' in df.columns else 600.0
min_toi = st.sidebar.slider(
    "Mininum Time on Ice (Minutes)",
    min_value=0.0,
    max_value=max_toi,
    value=50.0
)

filtered_df = df[df['TOI (sec)'] / 60 >= min_toi] if 'TOI (sec)' in df.columns else df

if 'conference' in filtered_df.columns:
    conferences = filtered_df['conference'].dropna().unique().tolist()
    selected_confs = st.sidebar.multiselect('Select Conferences', options=conferences, default=conferences)
    filtered_df = filtered_df[filtered_df['conference'].isin(selected_confs)]

teams = filtered_df['team'].dropna().unique().tolist()
selected_teams = st.sidebar.multiselect('Select Teams', options=teams, default=teams)
filtered_df = filtered_df[filtered_df['team'].isin(selected_teams)]

st.title('NCAA Hockey Player Comparison Dashboard')

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
stats_cols = [col for col in numeric_cols if col not in ['player', 'team', 'conference', 'position']]

col1, col2 = st.columns(2)
with col1:
    x_stat = st.selectbox("X-Axis Statistic", options=stats_cols, index=stats_cols.index('Successful Offensive Touches') if 'Successful Offensive Touches' in stats_cols else 0)
with col2:
    y_stat = st.selectbox("Y-Axis Statistic", options=stats_cols, index=stats_cols.index('Expected Goals') if 'Expected Goals' in stats_cols else 1)

if not filtered_df.empty:
    fig = px.scatter(
        filtered_df
        x=x_stat,
        y=y_stat,
        color='team',
        hover_name='player',
        hover_data=['team', 'position', 'toi_(min)'],
        title=f'{y_stat} vs {x_stat}',
        template='plotly_white'
    )
    fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))                      )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning('No Players match the selected filters. Please adjust your criteria.')
