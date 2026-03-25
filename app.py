# -----------------------------------------#
#             File Setup Sec.              #
# -----------------------------------------#
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity



# init x and y axes for starting up website
if 'x_stat_state' not in st.session_state:
    st.session_state.x_stat_state = 'successful_offensive_touches'
if 'y_stat_state' not in st.session_state:
    st.session_state.y_stat_state = 'defensive_dump_in_recoveries'

# button for setting new variables when recommendation button is clicked
def set_axes(new_x, new_y):
    st.session_state.x_stat_state = new_x
    st.session_state.y_stat_state = new_y

st.set_page_config(page_title='NCAA Hockey Analytics', layout='wide')
st.title("NCAA Men's Hockey Player Analytics")

# -----------------------------------------#
#             Intro/Manifest Sec.          #
# -----------------------------------------#

st.markdown('''
            Welcome to the interactive player evaluation dashboard. This tool 
            visualizes advanced player tracking and micro events for NCAA
            Division I Men's Hockey Hockey, enabling an apples-to-apples 
            comparisons across the country.

            **The Problem: Conference Bias** Evaluating NCAA hockey players 
            using raw data is inherently flawed due to the massive tactical 
            disparities between conferences. A skater competing in a high-event,
            transition-heavy conference will naturally accumulate more offensive
            touches and controlled entries than an equally talented player 
            operating in a rigid, defensively structured conference. If 
            evaluated purely on raw volume, the environment masks the true talent.
            
            **The Solution: Delta Metrics** To isolate individual skill from 
            team and conference systems, this dashboard utilizes standardized 
            metrics. By calculating the expected baseline for every micro-statistic
            within a specific conference and subtracting it from the player's 
            actual output, we generate **Delta Metrics**.

            These adjusted values represent a player's true performance *relative to their environment*. 
            A positive value indicates the player is generating more offense 
            (or preventing more chances) than the average skater operating within 
            that exact same conference style.
            ''')

with st.expander("Deeper Dive: Methodology & Dataset"):
    st.markdown("""
    * **Data Source:** Proprietary micro-event and tracking data provided by SportLogiq.
    * **Baseline Calculation:** Team and conference baselines were established by aggregating play-by-play tendencies, neutralizing outlier systems to find the true conference mean.
    * **Analytical Application:** This standardized approach is designed to aid in scouting, transfer portal evaluations, and predicting professional translation by stripping away the "noise" of NCAA conference disparity. 
    """)

st.divider()

# -----------------------------------------#
#             Metric Pairing               #
#            Recommender Sec.              #
# -----------------------------------------#


st.markdown('### Suggested Metric Pairings')
st.markdown('Not sure where to start? Try configuring the X and Y axes with these metric pairings to uncover specific player archetypes:')

colA, colB = st.columns(2)

with colA:
    st.info("**The Transition Engine (Zone-to-Zone)**\n*Finds players who act as a one-man breakout.*")
    st.button(
        "View Transition Engine", 
        on_click=set_axes, 
        args=('controlled_exits', 'controlled_entries'), 
        use_container_width=True
    )
    
    st.success("**The Dual-Threat Creator**\n*Identifies offensive catalysts who create chances for themselves and others.*")
    st.button(
        "View Dual-Threat Creator", 
        on_click=set_axes, 
        args=('total_pass_to_slot_attempts', 'expected_goals'), 
        use_container_width=True
    )

with colB:
    st.warning("**The Puck Manager (Risk vs. Reward)**\n*Highlights efficiency: high possession drivers who rarely turn it over.*")
    st.button(
        "View Puck Manager", 
        on_click=set_axes, 
        args=('possession_driving', 'failed_possessions'), 
        use_container_width=True
    )
    
    st.error("**The Entry Style Clash**\n*Separates the gritty, dump-and-chase players from rush-entry specialists.*")
    st.button(
        "View Entry Style Clash", 
        on_click=set_axes, 
        args=('dump_in_rate', 'total_carry_ins'), 
        use_container_width=True
    )
st.divider()

# -----------------------------------------#
#             Data Loader Sec.             #
# -----------------------------------------#

@st.cache_data
def load_data():
    df = pd.read_csv("datasets/ncaa_d1_player_deltas.csv")    
    return df

df = load_data()

df = df.iloc[:, 1:]


# -----------------------------------------#
#             Filter/Sidebar Sec.          #
# -----------------------------------------#

st.sidebar.header('Filters')

time_parts = df['toi_(min)'].str.split(':', expand=True)
df['toi_(min)'] = time_parts[0].astype(float) + (time_parts[1].astype(float) / 60)
df['toi_(min)'] = df['toi_(min)'].round(2)

filtered_df = df.copy()
max_toi = float(df['toi_(min)'].max()) if 'toi_(min)' in df.columns else 1000.0
min_toi = st.sidebar.slider(
    "Mininum Time on Ice (Minutes)",
    min_value=0.0,
    max_value=max_toi,
    value=50.0
)

filtered_df = filtered_df[filtered_df['toi_(min)'] >= min_toi]

if 'conference' in filtered_df.columns:
    st.sidebar.markdown("**Select Conferences**")
    conferences = sorted(filtered_df['conference'].dropna().unique().tolist())
    
    selected_confs = []
    for conf in conferences:
        if st.sidebar.checkbox(conf, value=True):
            selected_confs.append(conf)
            
    # Apply the filter based on which boxes are checked
    filtered_df = filtered_df[filtered_df['conference'].isin(selected_confs)]
teams = sorted(filtered_df['team'].dropna().unique().tolist())
selected_teams = st.sidebar.multiselect("Select Teams", options=teams, default=[])
if selected_teams:
    filtered_df = filtered_df[filtered_df['team'].isin(selected_teams)]
else:
    filtered_df = filtered_df.iloc[0:0]

# -----------------------------------------#
#              Visualization Sec.          #
#              Split into two tabs         # 
# -----------------------------------------#


# tab setup for two pages
tab1, tab2 = st.tabs(['Scatterplot Explorer', 'Player Similarity Engine'])

# -----------------------------------------#
#       Tab 1: Scatterplot Explorer        # 
# -----------------------------------------#

with tab1:
    st.markdown('## Metric Comparison')

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    stats_cols = [col for col in numeric_cols if col not in ['player', 'team', 'conference', 'position']]

    col1, col2 = st.columns(2)
    with col1:
        x_stat = st.selectbox(
            "X-Axis Statistic", 
            options=stats_cols, 
            key='x_stat_state' 
        )
    y_stat = st.selectbox(
            "Y-Axis Statistic", 
            options=stats_cols, 
            key='y_stat_state' 
        )

    if not selected_teams:
        st.info('Select one or more teams from the sidebar to populate the graph.')
    elif not filtered_df.empty:
        fig = px.scatter(
            filtered_df,
            x=x_stat,
            y=y_stat,
            color="team",
            hover_name="player",
            custom_data=["team", "position", "toi_(min)", 'conference', x_stat, y_stat]    
        )

        fig.update_traces(
                marker=dict(size=12, opacity=0.7, line=dict(width=0.5, color='white')),
                hovertemplate=(
                    "<b>%{hovertext}</b><br><br>" +
                    "<b>Team:</b> %{customdata[0]}<br>" +
                    "<b>Position:</b> %{customdata[1]}<br>" +
                    "<b>TOI:</b> %{customdata[2]} min<br><br>" +
                    "<b>Conference:</b> %{customdata[3]}<br><br>" +

                    "<b>" + x_stat + ":</b> %{x}<br>" +
                    "<b>" + y_stat + ":</b> %{y}<br>" +
                    "<extra></extra>"
            )
        )    
        fig.update_layout(
                title=f"{y_stat} vs {x_stat}",
                template="plotly_white",
                legend=dict(title="Teams"),
                margin=dict(l=20, r=20, t=50, b=20),
                hoverlabel=dict(
                    bgcolor="white",   
                    font_size=14,
                    font_family="Helvetica, Arial, sans-serif",
                    font_color="black",  
                    bordercolor="black"  
                )
            )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning('No Players match the selected filters. Please adjust your criteria.')



# -----------------------------------------#
#       Tab 2: Similarity Engine           # 
# -----------------------------------------#

with tab2:
    st.markdown("## Scout's Engine: Player Comparables")
    st.markdown('Select a target player to identify the 5 most similar skaters in the country based on their entire delta metric profile. ' \
    '*Note: Comparisons evaluate the entire NCAA landscape, restricted by only the Minimum TOI slider.*')

    sim_pool = df[df['toi_(min)'] >= min_toi].copy()

    player_list = sim_pool['player'].dropna().unique().tolist()
    target_player = st.selectbox('Select Target Player', options=player_list, index=None, placeholder="Type in player's name: ")

    if target_player:
        features = sim_pool[stats_cols].fillna(0)

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        sim_pool_reset = sim_pool.reset_index(drop=True)
        target_idx = sim_pool_reset[sim_pool_reset['player'] == target_player].index[0]

        target_vector = scaled_features[target_idx].reshape(1, -1)
        sim_scores = cosine_similarity(target_vector, scaled_features).flatten()

        sim_pool_reset['Similarity Score'] = sim_scores
        top_matches = sim_pool_reset[sim_pool_reset['player'] != target_player].sort_values(by='Similarity Score', ascending=False).head(5)

        st.divider()
        st.subheader(f'Top 5 Comparables for {target_player}')

        display_cols = ['player', 'team', 'position', 'conference', 'toi_(min)', 'Similarity Score']
        display_df = top_matches[display_cols].copy()

        display_df['Similarity Score'] = (display_df['Similarity Score'] * 100).round(1).astype(str) + '%'        
        
        display_df.rename(columns={
            'player': 'Player', 
            'team': 'Team', 
            'position': 'Position', 
            'conference': 'Conference', 
            'toi_(min)': 'TOI (Min)'
        }, inplace=True)
        
        st.dataframe(display_df, hide_index=True, use_container_width=True)