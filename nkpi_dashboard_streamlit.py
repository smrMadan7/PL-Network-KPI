import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine
from pyvis.network import Network
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import calendar
import numpy as np
import os
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
load_dotenv()

# Database connection
def get_database_connection():
    DATABASE_URL = os.getenv("DB_URL")
    engine = create_engine(DATABASE_URL)
    return engine

def get_members_with_office_hours(engine):
    query = 'SELECT COUNT(*) FROM public."Member" WHERE "officeHours" IS NOT NULL;'
    result = pd.read_sql(query, engine)
    return result.iloc[0, 0] 

def get_total_members(engine):
    query = 'SELECT COUNT(*) FROM public."Member";'
    result = pd.read_sql(query, engine)
    return result.iloc[0, 0] 

def get_teams_with_office_hours(engine):
    query = 'SELECT COUNT(*) FROM public."Team" WHERE "officeHours" IS NOT NULL;'
    result = pd.read_sql(query, engine)
    return result.iloc[0, 0]  

def get_total_teams(engine):
    query = 'SELECT COUNT(*) FROM public."Team";'
    result = pd.read_sql(query, engine)
    return result.iloc[0, 0] 

def fetch_team_data(engine):
    query = """
       SELECT 
    COALESCE(
        pe.properties->'teamName',
        pe.properties->'name'
    ) AS team,
    COALESCE(
        pe.properties->>'loggedInUserName', 
        pe.properties->'user'->>'name',
        'Untracked User'
    ) AS Viewed_By,
    pe.timestamp,
    EXTRACT(MONTH FROM pe.timestamp) AS month,  -- Extract as number
    EXTRACT(DAY FROM pe.timestamp) AS day,
    EXTRACT(YEAR FROM pe.timestamp) AS year,
    CASE
        WHEN pe."event" = 'irl-guest-list-table-team-clicked' THEN 'IRL Page'
        WHEN pe."event" = 'team-clicked' THEN 'Teams Landing Page'
        WHEN pe."event" = 'memeber-detail-team-clicked' THEN 'Member Profile Page'
        WHEN (pe."event" = 'project-detail-maintainer-team-clicked' or pe."event" = 'project-detail-contributing-team-clicked') THEN 'Project Page'
        WHEN pe."event" = 'featured-team-card-clicked' THEN 'Home Page'
        ELSE 'Other'
    END AS source_type,
    CASE
        WHEN (pe.properties->>'loggedInUserEmail' IS NOT NULL OR pe.properties->'user'->>'email' IS NOT NULL) THEN 'LoggedIn'
        ELSE 'LoggedOut'
    END AS user_status,
    -- Combine logged-in and logged-out users into a single row for each user
    CASE
        WHEN (pe.properties->>'loggedInUserEmail' IS NULL AND pe.properties->'user'->>'email' IS NULL) THEN 'Untracked'
        ELSE COALESCE(
            pe.properties->>'loggedInUserName', 
            pe.properties->'user'->>'name'
        )
    END AS user_label,
    COUNT(*) AS clicks
FROM 
    posthogevents pe
WHERE 
    pe."event" IN ('irl-guest-list-table-team-clicked', 'team-clicked', 'memeber-detail-team-clicked', 'project-detail-maintainer-team-clicked', 'project-detail-contributing-team-clicked', 'featured-team-card-clicked')
    AND (
        -- Exclude the specific users by name (whether logged-in or logged-out)
        (
            COALESCE(
                pe.properties->>'loggedInUserName', 
                pe.properties->'user'->>'name'
            ) NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
        )
        -- Include logged-out users (where loggedInUserName or user->name is NULL)
        OR 
        (pe.properties->>'loggedInUserName' IS NULL AND pe.properties->'user'->>'name' IS NULL)
    )
GROUP BY 
    pe.properties->'teamName',           
    pe.properties->'name',               
    pe.properties->>'loggedInUserName',  
    pe.properties->'user'->>'name',      
    pe.properties->>'loggedInUserEmail',
    pe.properties->'user'->>'email',     
    pe.timestamp,                        
    pe."event"                           
ORDER BY 
    pe.timestamp DESC,                   
    EXTRACT(MONTH FROM pe.timestamp) Desc;
    """
    return pd.read_sql(query, engine)


def calculate_week(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')  
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['week_of_month'] = ((df['day'] - 1) // 7 + 1).astype(int)  
    df['month-week'] = (
        df['month'].apply(lambda x: calendar.month_name[x]) + " " + 
        df['week_of_month'].apply(lambda x: f"{x}{'st' if x == 1 else 'nd' if x == 2 else 'rd' if x == 3 else 'th'}") + " Week"
    )
    periods = df['timestamp'].dt.to_period('W-SUN')
    df['week_start_date'] = periods.apply(lambda p: p.start_time)
    df['week_end_date'] = periods.apply(lambda p: p.end_time)
    df['week_range'] = df['week_start_date'].dt.strftime('%b %d') + " - " + df['week_end_date'].dt.strftime('%b %d')
    
    df['week_of_month'] = df['timestamp'].dt.isocalendar().week
    return df

def plot_bar_chart_for_team(df, time_aggregation):
    if time_aggregation == "Month":
        df['month_name'] = df['month'].apply(lambda x: calendar.month_name[x])
        
        month_order = list(calendar.month_name[1:])  
        df['month_name'] = pd.Categorical(df['month_name'], categories=month_order, ordered=True)
        
        aggregated_df = df.groupby(['year', 'month_name', 'source_type']).agg({'clicks': 'sum'}).reset_index()
        
    else:
        df['week_start'] = df['timestamp'] - pd.to_timedelta(df['timestamp'].dt.weekday, unit='D')
        df['week_end'] = df['week_start'] + pd.to_timedelta(6, unit='D')
        df['week_range'] = df['week_start'].dt.strftime('%b %d') + ' - ' + df['week_end'].dt.strftime('%b %d')
        aggregated_df = df.groupby(['year', 'week_range', 'source_type']).agg({'clicks': 'sum'}).reset_index()
    
    title = 'Breakdown of user engagement and activity on Directory team profiles'
    
    fig = go.Figure()

    for source_type in aggregated_df['source_type'].unique():
        event_data = aggregated_df[aggregated_df['source_type'] == source_type]
        x_values = event_data['week_range'] if time_aggregation == "Week" else event_data['month_name']
        fig.add_trace(go.Bar(
            x=x_values,
            y=event_data['clicks'],
            name=source_type
        ))

    fig.update_layout(
        title=title,
        xaxis_title=time_aggregation,
        yaxis_title="Views",
        barmode='stack',
        template="plotly_dark",
        xaxis_tickangle=45 
    )
    
    return fig

def plot_pie_chart_for_team_focus(focus_area_data):
    fig = go.Figure(data=[go.Pie(labels=focus_area_data['Focus Area'], values=focus_area_data['Teams'], hole=0.3)])
    fig.update_layout(
        template="plotly_dark"
    )
    return fig


def fetch_focus_area_data(engine):
    query = """
       SELECT 
    COALESCE(parentFA.title, fa.title, 'Undefined') AS "Focus Area",  -- Renaming FocusArea to "Focus Area" for better consistency
    COUNT(t.uid) AS "Teams"  -- Renaming "TeamCount" to "Teams" for better consistency
FROM 
    public."Team" t
LEFT JOIN 
    public."TeamFocusArea" tfa ON t.uid = tfa."teamUid"
LEFT JOIN 
    public."FocusArea" fa ON fa.uid = tfa."focusAreaUid"
LEFT JOIN 
    public."FocusAreaHierarchy" fah ON fa.uid = fah."subFocusAreaUid"
LEFT JOIN 
    public."FocusArea" parentFA ON fah."focusAreaUid" = parentFA.uid
GROUP BY 
    COALESCE(parentFA.title, fa.title, 'Undefined')
ORDER BY 
    "Teams" DESC;  -- Renamed "TeamCount" to "Teams" here for consistency
    """
    return pd.read_sql(query, engine)


def fetch_team_search_data(engine, year=None, month=None, user_status=None):
    query = """
        SELECT 
            COALESCE(
                NULLIF(properties->>'loggedInUserName', ''),  
                NULLIF(properties->'user'->>'name', ''),  
                'Untracked User'  
            ) AS searched_by,
            (properties->>'value') AS team_search_value,
            TO_CHAR("timestamp", 'YYYY-MM-DD') AS date,
            CASE 
                WHEN COALESCE(NULLIF(properties->>'loggedInUserName', ''), NULLIF(properties->'user'->>'name', '')) IS NOT NULL 
                THEN 'loggedin'
                ELSE 'loggedout'
            END AS user_status,
            COUNT(*) AS event_count  
        
        FROM posthogevents
        WHERE 
            "event" = 'team-search'
    """

    conditions = []
    params = []

    if year:
        conditions.append("EXTRACT(YEAR FROM \"timestamp\") = %s")
        params.append(year)
    
    if month:
        conditions.append("EXTRACT(MONTH FROM \"timestamp\") = %s")
        params.append(month)
    
    if user_status:
        if user_status == "loggedin":
            conditions.append("""
                COALESCE(
                    NULLIF(properties->>'loggedInUserName', ''), 
                    NULLIF(properties->'user'->>'name', '')
                ) IS NOT NULL
            """)
        elif user_status == "loggedout":
            conditions.append("""
                COALESCE(
                    NULLIF(properties->>'loggedInUserName', ''), 
                    NULLIF(properties->'user'->>'name', '')
                ) IS NULL
            """)

    if conditions:
        query += " AND " + " AND ".join(conditions)
    
    query += """
        GROUP BY 
            searched_by, team_search_value, date, user_status  
        ORDER BY 
            event_count DESC;
    """

    return pd.read_sql(query, engine, params=tuple(params))


def fetch_total_teams_count(engine):
    query = """
        SELECT COUNT(*) AS total_teams
        FROM public."Team"
    """
    result = pd.read_sql(query, engine)
    return result['total_teams'][0]

def fetch_teams_per_month(engine, year=None, month=None):
    query = """
        SELECT 
            TO_CHAR("createdAt", 'YYYY-MM') AS month, 
            COUNT(*) AS team_count
        FROM public."Team"
    """
    
    conditions = []
    params = []

    if year:
        conditions.append("EXTRACT(YEAR FROM \"createdAt\") = %s")
        params.append(year)
    
    if month:
        conditions.append("EXTRACT(MONTH FROM \"createdAt\") = %s")
        params.append(month)
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += """
       GROUP BY TO_CHAR("createdAt", 'YYYY-MM')
       ORDER BY month;
    """
    
    result = pd.read_sql(query, engine, params=tuple(params))
    return result


def visualize_teams_per_month(data):
    if not pd.api.types.is_datetime64_any_dtype(data['month']):
        data['month'] = pd.to_datetime(data['month'], format='%Y-%m')

    data_grouped = data.groupby(data['month'].dt.to_period('M')).agg({'team_count': 'sum'}).reset_index()
    
    data_grouped['month'] = data_grouped['month'].dt.strftime('%b %Y')  # Format as 'Jan 2023'

    fig = px.bar(
        data_grouped,
        x="month", 
        y="team_count",
        labels={"month": "Month", "team_count": "No. of Teams"}, 
        text="team_count"  
    )

    fig.update_traces(
        texttemplate='%{text}',  
        textposition='outside',  
        marker=dict(
            color='skyblue', 
            line=dict(color='black', width=1)  
        )
    )

    fig.update_layout(
        # title="New Teams by Month",  # Set the title
        xaxis_title="Month", 
        yaxis_title="No. of Teams",  
        xaxis_tickangle=45,  
        barmode='group', 
        template="plotly_dark", 
        xaxis=dict(tickmode='array', tickvals=data_grouped['month'], ticktext=data_grouped['month']),
        plot_bgcolor="rgba(0, 0, 0, 0)", 
        margin=dict(l=40, r=40, t=40, b=80)  
    )

    st.plotly_chart(fig)

def fetch_total_interaction_count(engine):
    query = """
      SELECT 
    COUNT(*) AS total_interaction_count
FROM 
    public.posthogevents p
LEFT JOIN 
    public."Member" sm 
    ON COALESCE(
        (p.properties->>'userUid'),
        (p.properties->>'loggedInUserUid')
    ) = sm.uid
LEFT JOIN 
    public."Member" tm 
    ON COALESCE(
        (p.properties->>'memberUid'),
        substring(p.properties->>'$current_url' FROM '/members/([^/]+)')
    ) = tm.uid
LEFT JOIN 
    public."Team" t 
    ON substring(p.properties->>'$pathname' FROM '/teams/([^/]+)') = t.uid
WHERE 
    p."event" IN (
        'irl-guest-list-table-office-hours-link-clicked', 
        'member-officehours-clicked',
        'team-officehours-clicked'
    )
    AND (
        COALESCE(
            p.properties->>'loggedInUserName', 
            p.properties->'user'->>'name'
        ) NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
    )
    """
    result = pd.read_sql(query, engine)
    return result['total_interaction_count'][0]


def fetch_member_interactions_data(engine, selected_year, selected_month):
    query = """
    SELECT 
        m.name AS office_hours_initiated_by,
        m2.name AS office_hours_initiated_to,
        TO_CHAR(mi."createdAt", 'DD-MM-YYYY') AS date,  -- Combined date in DD-MM-YYYY format
        EXTRACT(YEAR FROM mi."createdAt") AS year,
        CASE 
            WHEN mfu.status = 'COMPLETED' THEN 
                CASE 
                    WHEN mf.response = 'POSITIVE' THEN 'Responded as Yes'
                    WHEN mf.response = 'NEGATIVE' THEN 'Responded as No'
                    ELSE 'Completed with unknown response'
                END
            WHEN mfu.status = 'CLOSED' THEN 'Pop-up Dismissed'
            WHEN mfu.status = 'PENDING' THEN 'Did not respond'
            ELSE 'Unknown status'
        END AS feedback_response_status,
        -- Implement the CASE to check if comments match dictionary values
        CASE 
            WHEN 'IFR0001' = ANY(mf."comments") THEN 'Link is broken'
            WHEN 'IFR0002' = ANY(mf."comments") THEN 'I plan to schedule soon'
            WHEN 'IFR0003' = ANY(mf."comments") THEN 'Preferred slot is not available'
            WHEN 'IFR0005' = ANY(mf."comments") THEN 'Got rescheduled'
            WHEN 'IFR0006' = ANY(mf."comments") THEN 'Got cancelled'
            WHEN 'IFR0007' = ANY(mf."comments") THEN 'Member didn’t show up'
            WHEN 'IFR0008' = ANY(mf."comments") THEN 'I could not make it'
            WHEN 'IFR0009' = ANY(mf."comments") THEN 'Call quality issues'
            ELSE '-'  -- Default case when no match is found
        END AS feedback_comments,
        COALESCE(mf.rating::TEXT, '-') AS rating
    FROM 
        public."MemberInteraction" mi
    LEFT JOIN 
        public."Member" m ON mi."sourceMemberUid" = m.uid 
    LEFT JOIN 
        public."Member" m2 ON mi."targetMemberUid" = m2.uid 
    LEFT JOIN 
        public."MemberFollowUp" mfu ON mfu."interactionUid" = mi.uid 
    LEFT JOIN 
        public."MemberFeedback" mf ON mfu.uid = mf."followUpUid" 
    WHERE 
        mfu."type" = 'MEETING_INITIATED'
        AND (
            COALESCE(m."name") NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
        )
    """

    # Apply the filters for year
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM mi.\"createdAt\") = {selected_year}"

    # Apply the filters for month
    if selected_month != "All":
        month_number = month_mapping[selected_month]
        query += f" AND EXTRACT(MONTH FROM mi.\"createdAt\") = {month_number}"

    query += " ORDER BY mi.\"createdAt\" DESC;"

    return pd.read_sql(query, engine)


@st.cache_data
def load_data_from_db(query):
    """Load data from the database based on the provided query."""
    df = pd.read_sql_query(query, get_database_connection())
    return df


def create_interaction_table(df, direction="Source → Target"):
    st.subheader(f"{direction} Interaction Data by Month and Year")
    st.dataframe(df)  

    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"Download {direction} Interaction Data as CSV",
        data=csv_data,
        file_name=f"{direction}_interactions_by_month_year.csv",
        mime="text/csv"
    )

def plot_top_10_interaction_count(df, title, color, source):
    member_interaction_counts = df.groupby('source_member_name')['interaction_count'].sum().reset_index()

    top_10_member_interactions = member_interaction_counts.sort_values(by='interaction_count', ascending=False).head(10)
    
    st.subheader(f"Top 10 {title}")
    
    fig = px.bar(
        top_10_member_interactions,
        x='source_member_name',
        y='interaction_count',
        labels={"source_member_name": f"{source} Member", "interaction_count": "No. of Interaction"}, 
        # title=f"Top 10 {title} Interaction Count",
        color_discrete_sequence=[color],  
        template="plotly_dark", 
    )

    if source == 'Source':
      source = 'Initiated By'
    elif source == 'Target':
      source = 'Initiated To'
    fig.update_traces(texttemplate='%{y}', textposition='outside', textfont=dict(size=12, color='white'))

    fig.update_layout(
        barmode='group',  
        plot_bgcolor="rgba(0, 0, 0, 0)",  
        margin=dict(l=40, r=40, t=100, b=80),  
        xaxis_title=source, 
        yaxis_title="No. of Interaction",  
        xaxis=dict(tickangle=45),  
        showlegend=False 
    )

    st.plotly_chart(fig)


def plot_monthwise_interaction(df):
    """Creates a bar chart for month-wise interaction count."""
    
    df['month'] = pd.to_datetime(df['month'])
    
    fig = px.bar(df, 
                 x='month', 
                 y='interaction_count',
                 labels={'month': 'Month', 'interaction_count': 'Interaction Count'},
                 title='Monthly Interaction Counts',
                 template="plotly_dark",
                 color='interaction_count', 
                 color_continuous_scale="Viridis"
                )
    
    fig.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=40, r=40, t=40, b=80), 
        xaxis_title='Month',
        yaxis_title='Interaction Count',
        barmode='group', 
        xaxis_tickformat="%b %Y", 
        xaxis_tickangle=-45, 
        xaxis=dict(tickmode='array', tickvals=df['month'])  
    )

    st.plotly_chart(fig, use_container_width=True)

month_mapping = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}


def generate_filtered_query_of_oh(selected_year, selected_month):
    query = """
    SELECT 
        'Member' AS source_type,
        "officeHours",
        EXTRACT(MONTH FROM "createdAt") AS month,
        EXTRACT(YEAR FROM "createdAt") AS year
    FROM 
        public."Member" m
    WHERE 
        "officeHours" IS NOT NULL 
        AND TRIM("officeHours") != ''
        AND 
        (
        'name' NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
        )
    """
    
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM m.\"createdAt\") = {selected_year}"
    
    if selected_month != "All":
        month_num = month_mapping.get(selected_month) 
        query += f" AND EXTRACT(MONTH FROM m.\"createdAt\") = {month_num}"

    query += """
    UNION ALL

    SELECT 
        'Team' AS source_type,
        "officeHours",
        EXTRACT(MONTH FROM "createdAt") AS month,
        EXTRACT(YEAR FROM "createdAt") AS year
    FROM 
        public."Team" t
    WHERE 
        "officeHours" IS NOT NULL 
        AND TRIM("officeHours") != ''
    """
    
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM t.\"createdAt\") = {selected_year}"
    
    if selected_month != "All":
        month_num = month_mapping.get(selected_month)  
        query += f" AND EXTRACT(MONTH FROM t.\"createdAt\") = {month_num}"

    return query

def plot_mom_growth(df):    
    profile_count = df.groupby(['year', 'month', 'source_type']).size().reset_index(name='count')
    profile_count['Month_Year'] = pd.to_datetime(profile_count[['year', 'month']].assign(day=1))
    profile_count = profile_count.sort_values(by=['Month_Year'])
    profile_count['Previous_Month_Count'] = profile_count.groupby('source_type')['count'].shift(1)
    profile_count['MoM_Growth_Percentage'] = (
        (profile_count['count'] - profile_count['Previous_Month_Count']) /
        profile_count['Previous_Month_Count'].replace(0, np.nan)  
    ) * 100

    profile_count['MoM_Growth_Percentage'] = profile_count['MoM_Growth_Percentage'].fillna(0)
    first_month = profile_count['Month_Year'].min()
    profile_count.loc[profile_count['Month_Year'] == first_month, 'MoM_Growth_Percentage'] = 0

    fig1 = px.bar(profile_count, 
                 x='Month_Year', 
                 y='count', 
                 color='source_type',
                 labels={'Month_Year': 'Month-Year', 'count': 'Profile Count'},
                 title="Monthly Profile Count (Team and Member)",
                 barmode='group',
                 template="plotly_dark",
                 color_discrete_map={'Member': 'rgb(0, 123, 255)', 'Team': 'rgb(255, 159, 64)'})

    fig1.update_layout(
        xaxis_title="Month-Year",
        yaxis_title="Profile Count",
        xaxis=dict(
            tickformat="%b %Y", 
            tickangle=45, 
            showticklabels=True,
            tickfont=dict(size=10),  
            showgrid=False,
        ),
        yaxis=dict(
            showgrid=True, 
            zeroline=True,  
            title_standoff=10,  
            tickangle=0,  
            tickfont=dict(size=12),  
            ticks="outside", 
        ),
        margin=dict(l=40, r=40, t=40, b=120),  
        plot_bgcolor="rgba(0, 0, 0, 0)",  
        hovermode="closest", 
        legend_title="Source Type", 
        height=600,  
    )

    fig1.update_traces(marker_line_width=0)
    st.plotly_chart(fig1)

    fig2 = px.bar(profile_count, 
                 x='Month_Year', 
                 y='MoM_Growth_Percentage', 
                 color='source_type',
                 labels={'Month_Year': 'Month-Year', 'MoM_Growth_Percentage': 'Month-over-Month Growth (%)'},
                 title="Month-over-Month Growth in Profiles with Office Hours (Team and Member)",
                 barmode='group',
                 template="plotly_dark",
                 color_discrete_map={'Member': 'rgb(0, 123, 255)', 'Team': 'rgb(255, 159, 64)'})

    fig2.update_layout(
        xaxis_title="Month-Year",
        yaxis_title="MoM Growth (%)",
        xaxis=dict(
            tickformat="%b %Y", 
            tickangle=45, 
            showticklabels=True,
            tickfont=dict(size=10),  
            showgrid=False,
        ),
        yaxis=dict(
            showgrid=True, 
            zeroline=True,  
            title_standoff=10,  
            tickangle=0,  
            tickfont=dict(size=12),  
            ticks="outside", 
        ),
        margin=dict(l=40, r=40, t=40, b=120),  
        plot_bgcolor="rgba(0, 0, 0, 0)",  
        hovermode="closest", 
        legend_title="Source Type", 
        height=600,  
    )

    fig2.add_annotation(
        x=first_month, 
        y=0, 
        text="Starting Point (0%)", 
        showarrow=True, 
        arrowhead=2, 
        arrowsize=1, 
        arrowcolor="red", 
        font=dict(size=12, color="white"),
        bgcolor="rgba(255, 0, 0, 0.3)",  
        borderpad=4,
    )

    fig2.update_traces(marker_line_width=0)
    st.plotly_chart(fig2)

    st.subheader("Full Raw Data")
    st.write(profile_count)


# def plot_stacked_bar_chart_of_oh(df):
    
#     profile_count = df.groupby(['year', 'month', 'source_type']).size().reset_index(name='count')
#     profile_count['Month_Year'] = pd.to_datetime(profile_count[['year', 'month']].assign(day=1))

#     profile_count = profile_count.sort_values(by=['Month_Year', 'count'], ascending=[True, False])

#     month_names = profile_count['Month_Year'].dt.strftime('%B %Y').unique()

#     fig = px.bar(profile_count, 
#                  x='Month_Year', 
#                  y='count', 
#                  color='source_type',
#                  labels={'Month_Year': 'Month-Year', 'count': 'Profiles with Office Hours'},
#                  title="Profiles with Listed Office Hours by Month and Year (Team and Member)",
#                  barmode='stack',
#                  template="plotly_dark",
#                  color_discrete_map={'Member': 'rgb(0, 123, 255)', 'Team': 'rgb(255, 159, 64)'})

#     fig.update_layout(
#         xaxis_title="Month-Year",
#         yaxis_title="Number of Profiles",
#         xaxis=dict(
#             tickformat="%b %Y", 
#             tickangle=45, 
#             showticklabels=True,
#             tickfont=dict(size=10),  
#             showgrid=False,
#             tickvals=profile_count['Month_Year'].unique(),  
#             ticktext=month_names  
#         ),
#         yaxis=dict(
#             showgrid=True, 
#             zeroline=False,  
#             title_standoff=10,  
#             type='linear',  
#             autorange=True, 
#             tickangle=0,  
#             tickfont=dict(size=12),  
#             ticks="outside", 
#         ),
#         margin=dict(l=40, r=40, t=40, b=120),  
#         plot_bgcolor="rgba(0, 0, 0, 0)",  
#         hovermode="closest", 
#         legend_title="Source Type", 
#         height=600,  
#     )

#     fig.update_traces(marker_line_width=0)

#     st.plotly_chart(fig)


# query = """
# SELECT 
#     sm.name AS source_member_name,
#     tm.name AS target_member_name,
#     EXTRACT(MONTH FROM mi."createdAt") AS month,
#     EXTRACT(YEAR FROM mi."createdAt") AS year,
#     CASE 
#         WHEN p."event" = 'irl-guest-list-table-office-hours-link-clicked' THEN 'IRL'
#         WHEN p."event" = 'member-officehours-clicked' THEN 'Member'
#         ELSE 'Unknown'
#     END AS page_type,
#     COUNT(*) AS interaction_count
# FROM 
#     public."MemberInteraction" mi 
# INNER JOIN 
#     public.posthogevents p 
# ON 
#     mi."sourceMemberUid" = COALESCE(
#         (p.properties->>'userUid'),
#         (p.properties->>'loggedInUserUid'))
#     AND mi."targetMemberUid" = COALESCE(
#         (p.properties->>'memberUid'),
#         substring(p.properties->>'$current_url' FROM '/members/([^/]+)'))
# LEFT JOIN 
#     public."Member" sm 
# ON 
#     mi."sourceMemberUid" = sm.uid
# LEFT JOIN 
#     public."Member" tm 
# ON 
#     mi."targetMemberUid" = tm.uid
# WHERE 
#     p."event" IN (
#         'irl-guest-list-table-office-hours-link-clicked', 
#         'member-officehours-clicked'
#     )
# GROUP BY 
#     sm.name, 
#     tm.name, 
#     EXTRACT(MONTH FROM mi."createdAt"), 
#     EXTRACT(YEAR FROM mi."createdAt"), 
#     p."event"
# ORDER BY 
#     year, 
#     month, 
#     interaction_count DESC;
# """

    
def fetch_oh_data(selected_year, selected_month, month_mapping):
    query = """
    SELECT 
        CASE 
            WHEN COALESCE(
                (p.properties->>'userUid'),
                (p.properties->>'loggedInUserUid')
            ) IS NULL THEN 
                COALESCE(
                    p.properties->>'loggedInUserName', 
                    p.properties->'user'->>'name'
                )
            ELSE sm.name 
        END AS initiated_by, 
        CASE 
            WHEN p."event" = 'irl-guest-list-table-office-hours-link-clicked' THEN tm.name  -- Regular member
            WHEN p."event" = 'member-officehours-clicked' THEN tm.name                        -- Member
            WHEN p."event" = 'team-officehours-clicked' THEN t.name                          -- New event for team
            ELSE 'Unknown'
        END AS Initiated_Team__or_Member,
        EXTRACT(MONTH FROM COALESCE(NULLIF((p.properties ->> '$sent_at')::text, ''), '1970-01-01')::timestamp) AS month,
        EXTRACT(YEAR FROM COALESCE(NULLIF((p.properties ->> '$sent_at')::text, ''), '1970-01-01')::timestamp) AS year,
        CASE 
            WHEN p."event" = 'irl-guest-list-table-office-hours-link-clicked' THEN 'IRL Page'
            WHEN p."event" = 'member-officehours-clicked' THEN 'Member Page'
            WHEN p."event" = 'team-officehours-clicked' THEN 'Team Page'   -- Adding "Team" for new event
            ELSE 'Unknown'
        END AS page_type,
        CASE
            WHEN p."event" IN ('irl-guest-list-table-office-hours-link-clicked', 'member-officehours-clicked') THEN 'Member'
            WHEN p."event" = 'team-officehours-clicked' THEN 'Team'
            ELSE 'Unknown'
        END AS target_type,
        COUNT(*) AS interaction_count,
        TO_DATE(SUBSTRING(p.properties ->> '$sent_at' FROM 1 FOR 10), 'YYYY-MM-DD') AS date
    FROM 
        public.posthogevents p
    LEFT JOIN 
        public."Member" sm 
        ON COALESCE(
            (p.properties->>'userUid'),
            (p.properties->>'loggedInUserUid')
        ) = sm.uid
    LEFT JOIN 
        public."Member" tm 
        ON COALESCE(
            (p.properties->>'memberUid'),
            substring(p.properties->>'$current_url' FROM '/members/([^/]+)')
        ) = tm.uid
    LEFT JOIN 
        public."Team" t 
        ON substring(p.properties->>'$pathname' FROM '/teams/([^/]+)') = t.uid  -- Join with the Team table using the team UID
    WHERE 
        p."event" IN (
            'irl-guest-list-table-office-hours-link-clicked', 
            'member-officehours-clicked',
            'team-officehours-clicked'  -- Including the new event
        )
        AND (
            COALESCE(
                p.properties->>'loggedInUserName', 
                p.properties->'user'->>'name'
            ) NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
        )
    """

    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM COALESCE(NULLIF((p.properties ->> '$sent_at')::text, ''), '1970-01-01')::timestamp) = {selected_year}"

    if selected_month != "All":
        # Map the selected month name to its numeric value using the month_mapping dictionary
        month_num = month_mapping.get(selected_month)
        if month_num:
            query += f" AND EXTRACT(MONTH FROM COALESCE(NULLIF((p.properties ->> '$sent_at')::text, ''), '1970-01-01')::timestamp) = {month_num}"

    query += """
    GROUP BY 
        sm.name, 
        tm.name, 
        t.name,   -- Include the team name in the GROUP BY
        EXTRACT(MONTH FROM COALESCE(NULLIF((p.properties ->> '$sent_at')::text, ''), '1970-01-01')::timestamp), 
        EXTRACT(YEAR FROM COALESCE(NULLIF((p.properties ->> '$sent_at')::text, ''), '1970-01-01')::timestamp),
        (p.properties ->> '$sent_at'), 
        COALESCE(
                (p.properties->>'userUid'),
                (p.properties->>'loggedInUserUid')
            ),
        COALESCE(
                    p.properties->>'loggedInUserName', 
                    p.properties->'user'->>'name'
                ),
        p."event"
    ORDER BY 
        year, 
        month, 
        interaction_count;
    """

    return query


def plot_bar_chart_of_OH(df, breakdown_type):
    fig = go.Figure()

    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    df['month_order'] = df['month_name'].apply(lambda x: month_order.index(x) + 1)

    # Depending on the breakdown type, either show the Team Breakdown or Member Breakdown
    if breakdown_type == "Team Breakdown":
        # Filter data to only include Team Page interactions
        filtered_df = df[df['page_type'] == 'Team Page']
        aggregated_data = filtered_df.groupby('month_name').agg(
            interaction_count=('interaction_count', 'sum')
        ).reset_index()

        fig.add_trace(go.Bar(
            x=aggregated_data['month_name'], 
            y=aggregated_data['interaction_count'],
            name='Team Page',
            hovertemplate=
                '<b>Month:</b> %{x}<br>' +  
                '<b>Breakdown Type:</b> Team Page<br>' +  
                '<b>Interactions:</b> %{y}', 
            marker=dict(
                line=dict(width=0) 
            )
        ))
    else:
        # Filter data to include both IRL and Member Pages under Member Breakdown
        filtered_df = df[df['page_type'].isin(['IRL Page', 'Member Page'])]
        aggregated_data_irl = filtered_df[filtered_df['page_type'] == 'IRL Page'].groupby('month_name').agg(
            interaction_count=('interaction_count', 'sum')
        ).reset_index()

        aggregated_data_member = filtered_df[filtered_df['page_type'] == 'Member Page'].groupby('month_name').agg(
            interaction_count=('interaction_count', 'sum')
        ).reset_index()

        # Add IRL Page trace
        fig.add_trace(go.Bar(
            x=aggregated_data_irl['month_name'], 
            y=aggregated_data_irl['interaction_count'],
            name='IRL Page',
            hovertemplate=
                '<b>Month:</b> %{x}<br>' +  
                '<b>Breakdown Type:</b> IRL Page<br>' +  
                '<b>Interactions:</b> %{y}', 
            marker=dict(
                line=dict(width=0) 
            )
        ))

        # Add Member Page trace
        fig.add_trace(go.Bar(
            x=aggregated_data_member['month_name'], 
            y=aggregated_data_member['interaction_count'],
            name='Member Page',
            hovertemplate=
                '<b>Month:</b> %{x}<br>' +  
                '<b>Breakdown Type:</b> Member Page<br>' +  
                '<b>Interactions:</b> %{y}', 
            marker=dict(
                line=dict(width=0) 
            )
        ))

    # Update layout
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="No. of Interaction",
        barmode='stack',
        template="plotly_dark",
        xaxis=dict(
            tickmode='array',
            tickvals=month_order,  
            ticktext=month_order  
        ),
        xaxis_tickangle=45  
    )

    return fig

def build_source_to_target_query(selected_year, selected_month_num):
    base_query = """
    SELECT 
        CASE 
            WHEN COALESCE(
                (p.properties->>'userUid'),
                (p.properties->>'loggedInUserUid')
            ) IS NULL THEN 
                COALESCE(
                    p.properties->>'loggedInUserName', 
                    p.properties->'user'->>'name'
                )
            ELSE sm.name 
        END AS source_member_name, 
        CASE 
            WHEN p."event" = 'irl-guest-list-table-office-hours-link-clicked' THEN tm.name  
            WHEN p."event" = 'member-officehours-clicked' THEN tm.name                       
            ELSE 'Unknown'
        END AS target_name,
        EXTRACT(MONTH FROM (p.properties ->> '$sent_at')::timestamp) AS month,
        EXTRACT(YEAR FROM (p.properties ->> '$sent_at')::timestamp) AS year,
        COUNT(*) AS interaction_count,
        TO_DATE(SUBSTRING(p.properties ->> '$sent_at' FROM 1 FOR 10), 'YYYY-MM-DD') AS date
    FROM 
        public.posthogevents p
    LEFT JOIN 
        public."Member" sm 
        ON COALESCE(
            (p.properties->>'userUid'),
            (p.properties->>'loggedInUserUid')
        ) = sm.uid
    LEFT JOIN 
        public."Member" tm 
        ON COALESCE(
            (p.properties->>'memberUid'),
            substring(p.properties->>'$current_url' FROM '/members/([^/]+)')
        ) = tm.uid
    WHERE 
        p."event" IN (
            'irl-guest-list-table-office-hours-link-clicked', 
            'member-officehours-clicked'
        )
        AND (
            COALESCE(
                p.properties->>'loggedInUserName', 
                p.properties->'user'->>'name'
            ) NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
        )
    """
    
    if selected_year != "All":
        base_query += f" AND EXTRACT(YEAR FROM (p.properties ->> '$sent_at')::timestamp) = {selected_year}"

    if selected_month_num:
        base_query += f" AND EXTRACT(MONTH FROM (p.properties ->> '$sent_at')::timestamp) = {selected_month_num}"

    base_query += """
    GROUP BY 
        sm.name, 
        tm.name, 
        EXTRACT(MONTH FROM (p.properties ->> '$sent_at')::timestamp), 
        EXTRACT(YEAR FROM (p.properties ->> '$sent_at')::timestamp), 
         COALESCE(
            (p.properties->>'userUid'),
            (p.properties->>'loggedInUserUid')
        ), 
        COALESCE(
                p.properties->>'loggedInUserName', 
                p.properties->'user'->>'name'
            ),
        (p.properties ->> '$sent_at'),
        p."event"
    ORDER BY 
        year, 
        month, 
        interaction_count DESC;
    """

    return base_query

def build_source_to_team_query(selected_year, selected_month_num):
    base_query = """
    SELECT 
    CASE 
        WHEN COALESCE(
            (p.properties->>'userUid'),
            (p.properties->>'loggedInUserUid')
        ) IS NULL THEN 
            COALESCE(
                p.properties->>'loggedInUserName', 
                p.properties->'user'->>'name'
            )
        ELSE sm.name 
    END AS source_member_name,
    CASE 
        WHEN p."event" = 'team-officehours-clicked' THEN t.name  
        ELSE 'Unknown'
    END AS target_name,
    EXTRACT(MONTH FROM (p.properties ->> '$sent_at')::timestamp) AS month,
    EXTRACT(YEAR FROM (p.properties ->> '$sent_at')::timestamp) AS year,
    COUNT(*) AS interaction_count,
    TO_DATE(SUBSTRING(p.properties ->> '$sent_at' FROM 1 FOR 10), 'YYYY-MM-DD') AS date
FROM 
    public.posthogevents p
LEFT JOIN 
    public."Team" t 
    ON substring(p.properties->>'$pathname' FROM '/teams/([^/]+)') = t.uid 
LEFT JOIN 
    public."Member" sm 
    ON COALESCE(
        (p.properties->>'userUid'),
        (p.properties->>'loggedInUserUid')    
    ) = sm.uid
WHERE 
    p."event" IN (
        'team-officehours-clicked'
    )
    AND (
        COALESCE(
            p.properties->>'loggedInUserName', 
            p.properties->'user'->>'name'
        ) NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
    )
    """
    
    if selected_year != "All":
        base_query += f" AND EXTRACT(YEAR FROM (p.properties ->> '$sent_at')::timestamp) = {selected_year}"

    if selected_month_num:
        base_query += f" AND EXTRACT(MONTH FROM (p.properties ->> '$sent_at')::timestamp) = {selected_month_num}"

    base_query += """
    GROUP BY 
        sm.name, 
        t.name,
        EXTRACT(MONTH FROM (p.properties ->> '$sent_at')::timestamp), 
        EXTRACT(YEAR FROM (p.properties ->> '$sent_at')::timestamp),
        COALESCE(
            (p.properties->>'userUid'),
            (p.properties->>'loggedInUserUid')
        ), 
        COALESCE(
                p.properties->>'loggedInUserName', 
                p.properties->'user'->>'name'
            ),
        p."event",
        TO_DATE(SUBSTRING(p.properties ->> '$sent_at' FROM 1 FOR 10), 'YYYY-MM-DD')  -- Added this line to group by date expression
    ORDER BY 
        year, 
        month, 
        interaction_count DESC;
    """

    return base_query



def build_target_to_source_query(selected_year, selected_month_num):
    base_query = """
    SELECT 
        tm.name AS target_member_name,  
        CASE 
        WHEN COALESCE(
            (p.properties->>'userUid'),
            (p.properties->>'loggedInUserUid')
        ) IS NULL THEN 
            COALESCE(
                p.properties->>'loggedInUserName', 
                p.properties->'user'->>'name'
            )
        ELSE sm.name 
        END AS source_member_name,
        EXTRACT(MONTH FROM (p.properties ->> '$sent_at')::timestamp) AS month,
        EXTRACT(YEAR FROM (p.properties ->> '$sent_at')::timestamp) AS year,
        COUNT(*) AS interaction_count,
        TO_DATE(SUBSTRING(p.properties ->> '$sent_at' FROM 1 FOR 10), 'YYYY-MM-DD') AS date
    FROM 
        public.posthogevents p
    LEFT JOIN 
        public."Member" sm 
        ON COALESCE(
            (p.properties->>'userUid'),
            (p.properties->>'loggedInUserUid')
        ) = sm.uid
    LEFT JOIN 
        public."Member" tm 
        ON COALESCE(
            (p.properties->>'memberUid'),
            substring(p.properties->>'$current_url' FROM '/members/([^/]+)')
        ) = tm.uid
    WHERE 
        p."event" IN (
            'irl-guest-list-table-office-hours-link-clicked', 
            'member-officehours-clicked'
        )
        AND (
            COALESCE(
                p.properties->>'loggedInUserName', 
                p.properties->'user'->>'name'
            ) NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
        )
    """
    
    if selected_year != "All":
        base_query += f" AND EXTRACT(YEAR FROM (p.properties ->> '$sent_at')::timestamp) = {selected_year}"

    if selected_month_num:
        base_query += f" AND EXTRACT(MONTH FROM (p.properties ->> '$sent_at')::timestamp) = {selected_month_num}"

    base_query += """
    GROUP BY 
        tm.name,  
        sm.name, 
        EXTRACT(MONTH FROM (p.properties ->> '$sent_at')::timestamp), 
        EXTRACT(YEAR FROM (p.properties ->> '$sent_at')::timestamp), 
         COALESCE(
            (p.properties->>'userUid'),
            (p.properties->>'loggedInUserUid')
        ), 
        COALESCE(
                p.properties->>'loggedInUserName', 
                p.properties->'user'->>'name'
            ),
        (p.properties ->> '$sent_at'),
        p."event"
    ORDER BY 
        year, 
        month, 
        interaction_count DESC;
    """

    return base_query



def main():
    st.set_page_config(page_title="nKPI Dashboard", layout="wide")

    page = st.sidebar.radio(
        "nKPI Dashboard",
        [
            "Team Analysis",
            "Office Hours Analysis",
            "Usage/Activity",
            "Network Growth",
            "IRL Gathering",
            "Hackathon/Workshops",
            "Network Density",
            "Network Strength - LongTerm Connectivity", 
            "Network Strength - Conversation Based Connectivity"
        ]
    )

    engine = get_database_connection()

    if page == "Team Analysis":
        st.title("Activity -- Directory Teams")

        st.markdown("""Breakdown of user engagement and activity on Directory team profiles""")

        df = fetch_team_data(engine)
        df['year'] = df['year'].astype(int)
        

        total_teams = fetch_total_teams_count(engine)

        st.markdown(
            f"""
            <div style="background-color:#FFD700; padding:20px; border-radius:8px; text-align:center;">
                <h3>Number of Teams in the Network</h3>
                <p style="font-size:28px; font-weight:bold; color:#2b2b2b;">{total_teams}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Filters")
        years = [2022, 2023, 2024]

        month_mapping = {
            "January": 1,
            "February": 2,
            "March": 3,
            "April": 4,
            "May": 5,
            "June": 6,
            "July": 7,
            "August": 8,
            "September": 9,
            "October": 10,
            "November": 11,
            "December": 12
        }

        months = [
            'January', 'February', 'March', 'April', 'May', 'June', 
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
      
        selected_year = st.selectbox("Select Year", ["All"] + years, index=0)
        selected_month = st.selectbox("Select Month", ["All"] + months, index=0)

        user_status_options = ['All', 'LoggedIn', 'LoggedOut']
        selected_user_status = st.selectbox("Select User Status", user_status_options, index=0)

        df = calculate_week(df)

        if selected_year != "All":
            df = df[df['year'] == int(selected_year)]

        if selected_month != "All":
            month_num = month_mapping.get(selected_month)  
            df = df[df['month'] == month_num]

        if selected_user_status != "All":
            df = df[df['user_status'] == selected_user_status]

        st.subheader("Activity (MAUs, WAUs) - Directory Teams")

        col1, col2 = st.columns([1, 3])
        with col1:
            time_aggregation = st.radio(
                "View by -",
                ["Month", "Week"],
                index=0
            )

        with col2:
            fig = plot_bar_chart_for_team(df, time_aggregation)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Overall Data"):
            df_modified = df.copy()

            df_modified.columns = df_modified.columns.str.lower().str.replace(' ', '_')

            if 'id' in df_modified.columns:
                df_modified = df_modified.drop(columns=['id'])
            if 'timestamp' in df_modified.columns:
                df_modified['date'] = df_modified['timestamp'].dt.date 

            columns_to_drop = ['month', 'day', 'year', 'week_of_month', 'week_start_date', 'week_end_date', 'week_range', 
                            'user_status', 'month-week', 'timestamp', 'clicks', 'month_name','week_start', 'week_end', 'user_label' ]

            df_modified = df_modified.drop(columns=[col for col in columns_to_drop if col in df_modified.columns])
            df_modified.columns = df_modified.columns.str.replace('_', ' ').str.title().str.replace(' ', ' ')
            st.dataframe(df_modified, use_container_width=True)

        st.subheader("Monthly New Teams")
        teams_per_month = fetch_teams_per_month(
            engine,
            year=int(selected_year) if selected_year != "All" else None,
            month=month_mapping.get(selected_month, None) if selected_month != "All" else None,
        )

        visualize_teams_per_month(teams_per_month)

        st.subheader("Team Segmentation by Focus Areas")
        focus_area_data = fetch_focus_area_data(engine)
        pie_fig = plot_pie_chart_for_team_focus(focus_area_data)
        st.plotly_chart(pie_fig, use_container_width=True)

        # '''loggedin_filter = st.selectbox("Select User Status", ["All", "LoggedIn User(Active)", "LoggedOut User"], index=0)

        # team_search_data = fetch_team_search_data(
        #     engine,
        #     year=int(selected_year) if selected_year != "All" else None,
        #     month=month_mapping.get(selected_month, None) if selected_month != "All" else None,
        #     user_status="loggedin" if loggedin_filter == "LoggedIn User(Active)" else ("loggedout" if loggedin_filter == "LoggedOut User" else None)
        # )

        # st.subheader("Team Search Data")
        # team_search_data = team_search_data.drop(columns=['event_count'])
        # team_search_data.columns = team_search_data.columns.str.replace('_', ' ').str.title().str.replace(' ', '')
        # st.dataframe(team_search_data, use_container_width=True)'''


    elif page == "Office Hours Analysis":
        st.title("Activity -- OH")
        
        st.markdown("""
            Breakdown of OH activity on Member and Team Profile
        """)

        total_interactions = fetch_total_interaction_count(engine)

        st.markdown(
            f"""
            <div style="background-color:#FFD700; padding:20px; border-radius:8px; text-align:center;">
                <h3>Number of OH initiated</h3>
                <p style="font-size:28px; font-weight:bold; color:#2b2b2b;">{total_interactions}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Filters")
        years = ["All", "2022","2023","2024"] 

        month_mapping = {
            "January": 1,
            "February": 2,
            "March": 3,
            "April": 4,
            "May": 5,
            "June": 6,
            "July": 7,
            "August": 8,
            "September": 9,
            "October": 10,
            "November": 11,
            "December": 12
        }

        months = [
            'January', 'February', 'March', 'April', 'May', 'June', 
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

        selected_year = st.selectbox("Select Year", years, index=0)
        selected_month = st.selectbox("Select Month", ["All"] + months, index=0)

        # Fetch data from the database based on selected filters
        query = fetch_oh_data(selected_year, selected_month, month_mapping)
        df = load_data_from_db(query)

        df['month_name'] = df['month'].apply(lambda x: list(month_mapping.keys())[list(month_mapping.values()).index(x)] if x in month_mapping.values() else 'Unknown')

        st.subheader(f"Breakdown of OH by Month, Year of Team/Member")

        breakdown_type = st.radio("Select Breakdown", ["Member Breakdown", "Team Breakdown"], index=0)

        fig = plot_bar_chart_of_OH(df, breakdown_type)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Overall OH"):

            df.columns = df.columns.str.lower().str.replace(' ', '_')

            if 'id' in df.columns:
                df = df.drop(columns=['id'])

            columns_to_drop = ['year', 'month', 'month_order', 'month_name', 'page_type']
            df['OH_Initiated_Source'] = df['page_type']

            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
            df.columns = df.columns.str.replace('_', ' ').str.title().str.replace(' ', ' ')

            st.dataframe(df, use_container_width=True)

        selected_month_num = month_mapping.get(selected_month, None) if selected_month != "All" else None

        st.subheader(f"Office Hours - LeaderBoard")

        source_to_target_query = build_source_to_target_query(selected_year, selected_month_num)
        target_to_source_query = build_target_to_source_query(selected_year, selected_month_num)
        source_to_team_target_query = build_source_to_team_query(selected_year, selected_month_num)

        source_to_target_df = load_data_from_db(source_to_target_query)
        target_to_source_df = load_data_from_db(target_to_source_query)
        source_to_target_team_df = load_data_from_db(source_to_team_target_query)

        plot_top_10_interaction_count(source_to_target_df, "Source Members with the Most Interactions", "purple", "Source")

        with st.expander("Overall OH Breakdown of Source with Target members by Date"):

            source_to_target_df.columns = source_to_target_df.columns.str.lower().str.replace(' ', '_')

            if 'id' in source_to_target_df.columns:
                source_to_target_df = source_to_target_df.drop(columns=['id'])

            columns_to_drop = ['year', 'month', 'month_order', 'month_name', 'source_member_name', 'target_name', 'interaction_count', 'date']

            source_to_target_df['Initiated By'] = source_to_target_df['source_member_name']
            source_to_target_df['Initiated To'] = source_to_target_df['target_name']
            source_to_target_df['Interaction Count'] = source_to_target_df['interaction_count']
            source_to_target_df['Date'] = source_to_target_df['date']
            source_to_target_df = source_to_target_df.drop(columns=[col for col in columns_to_drop if col in source_to_target_df.columns])
            source_to_target_df.columns = source_to_target_df.columns.str.replace('_', ' ').str.title().str.replace(' ', ' ')

            st.dataframe(source_to_target_df, use_container_width=True)

        plot_top_10_interaction_count(target_to_source_df, "Target Members with the Most Interactions", "orange", "Target")

        with st.expander("Overall OH Breakdown of Target with Source members by Date"):

            target_to_source_df.columns = target_to_source_df.columns.str.lower().str.replace(' ', '_')

            if 'id' in target_to_source_df.columns:
                target_to_source_df = target_to_source_df.drop(columns=['id'])

            columns_to_drop = ['year', 'month', 'month_order', 'month_name','target_member_name', 'source_member_name', 'interaction_count', 'date']

            target_to_source_df['Initiated By'] = target_to_source_df['target_member_name']
            target_to_source_df['Initiated To'] = target_to_source_df['source_member_name']
            target_to_source_df['Interaction Count'] = target_to_source_df['interaction_count']
            target_to_source_df['Date'] = target_to_source_df['date']

            target_to_source_df = target_to_source_df.drop(columns=[col for col in columns_to_drop if col in target_to_source_df.columns])
            target_to_source_df.columns = target_to_source_df.columns.str.replace('_', ' ').str.title().str.replace(' ', ' ')

            st.dataframe(target_to_source_df, use_container_width=True)

        plot_top_10_interaction_count(source_to_target_team_df, "Source Members with the Most Team Interactions", "red", "Source")

        with st.expander("Overall OH Breakdown of Source members with Target Team by Date"):

            source_to_target_team_df.columns = source_to_target_team_df.columns.str.lower().str.replace(' ', '_')

            if 'id' in source_to_target_team_df.columns:
                source_to_target_team_df = source_to_target_team_df.drop(columns=['id'])

            columns_to_drop = ['year', 'month', 'month_order', 'month_name', 'source_member_name', 'target_name', 'interaction_count', 'date']

            source_to_target_team_df['Initiated By'] = source_to_target_team_df['source_member_name']
            source_to_target_team_df['Team Name'] = source_to_target_team_df['target_name']
            source_to_target_team_df['Interaction Count'] = source_to_target_team_df['interaction_count']
            source_to_target_team_df['Date'] = source_to_target_team_df['date']

            source_to_target_team_df = source_to_target_team_df.drop(columns=[col for col in columns_to_drop if col in source_to_target_team_df.columns])
            source_to_target_team_df.columns = source_to_target_team_df.columns.str.replace('_', ' ').str.title().str.replace(' ', ' ')

            st.dataframe(source_to_target_team_df, use_container_width=True)

        df = fetch_member_interactions_data(engine, selected_year, selected_month)

        st.subheader("OH Feedback Data")

        df.columns = df.columns.str.lower().str.replace(' ', '_')

        if 'id' in df.columns:
            df = df.drop(columns=['id'])

        columns_to_drop = ['year']

        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        df.columns = df.columns.str.replace('_', ' ').str.title().str.replace(' ', ' ')
        st.dataframe(df, use_container_width=True)

        members_with_office_hours = get_members_with_office_hours(engine)
        total_members = get_total_members(engine)
        teams_with_office_hours = get_teams_with_office_hours(engine)
        total_teams = get_total_teams(engine)

        # Calculate percentages
        members_percentage = (members_with_office_hours / total_members) * 100
        teams_percentage = (teams_with_office_hours / total_teams) * 100

        # Prepare data for pie chart
        labels = ['Members with Office Hours', 'Members without Office Hours']
        sizes = [members_percentage, 100 - members_percentage]

        labels_teams = ['Teams with Office Hours', 'Teams without Office Hours']
        sizes_teams = [teams_percentage, 100 - teams_percentage]

        # Create Pie Chart for Members using Plotly
        fig_members = go.Figure(data=[go.Pie(
            labels=labels,
            values=sizes,
            hole=0.3,  # Makes the pie chart donut-shaped for better readability
            hoverinfo="label+percent",
            textinfo="percent",
            marker=dict(colors=['#66b3ff', '#ff9999']),
            showlegend=True
        )])

        fig_members.update_layout(
            # title="Members with and without Office Hours",
            # title_x=0.5,
            width=600,  
            height=600, 
            margin=dict(t=50, b=50, l=50, r=50), 
            legend=dict(
                x=1,  
                y=1,
                traceorder='normal',
                orientation='v', 
                font=dict(size=12),
                borderwidth=1
            )
        )

        st.subheader("Percentage of Members with Office Hours")
        st.plotly_chart(fig_members)

        fig_teams = go.Figure(data=[go.Pie(
            labels=labels_teams,
            values=sizes_teams,
            hole=0.3,  
            hoverinfo="label+percent",
            textinfo="percent",
            marker=dict(colors=['#99ff99', '#ffcc99']),
            showlegend=True
        )])

        fig_teams.update_layout(
            # title="Teams with and without Office Hours",
            # title_x=0.5,
            width=600,  
            height=600,  
            margin=dict(t=50, b=50, l=50, r=50),  
            legend=dict(
                x=1,  
                y=1,
                traceorder='normal',
                orientation='v',  
                font=dict(size=12),
                borderwidth=1
            )
        )
        st.subheader("Precentage of Teams with Office Hours")
        st.plotly_chart(fig_teams)

    elif page == 'Usage/Activity':
        st.title("Activity -- Usage/Activity")

        st.markdown("""
            Breakdown of Usage/ctivity
        """)

        st.subheader("Filters")
        years = ["All", "2024"] 

        month_mapping = {
            "January": 1,
            "February": 2,
            "March": 3,
            "April": 4,
            "May": 5,
            "June": 6,
            "July": 7,
            "August": 8,
            "September": 9,
            "October": 10,
            "November": 11,
            "December": 12
        }

        months = [
            'January', 'February', 'March', 'April', 'May', 'June', 
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

        selected_year = st.selectbox("Select Year", years, index=0)
        selected_month = st.selectbox("Select Month", ["All"] + months, index=0)
        user_status_options = ['All', 'LoggedIn', 'LoggedOut']
        loggedin_filter = st.selectbox("Select User Status", ["All", "LoggedIn User", "LoggedOut User"], index=0)
        team_search_data = fetch_team_search_data(
            engine,
            year=int(selected_year) if selected_year != "All" else None,
            month=month_mapping.get(selected_month, None) if selected_month != "All" else None,
            user_status="loggedin" if loggedin_filter == "LoggedIn User" else ("loggedout" if loggedin_filter == "LoggedOut User" else None)
        )

        st.subheader("Team Search Data")
        team_search_data = team_search_data.drop(columns=['event_count'])
        team_search_data.columns = team_search_data.columns.str.replace('_', ' ').str.title().str.replace(' ', ' ')
        st.dataframe(team_search_data, use_container_width=True)

        st.subheader("Active User")
        dummy_image_url = "https://plabs-assets.s3.us-west-1.amazonaws.com/Coming+Soon.png"
        st.image(dummy_image_url, caption="Active User Overview", width=900)

        st.subheader("Interacted with husky")
        dummy_image_url = "https://plabs-assets.s3.us-west-1.amazonaws.com/Coming+Soon.png"
        st.image(dummy_image_url, caption="Husky Overview", width=900)

        st.subheader("HomePage Interaction")
        dummy_image_url = "https://plabs-assets.s3.us-west-1.amazonaws.com/Coming+Soon.png"
        st.image(dummy_image_url, caption="Homepage Overview", width=900)

        st.subheader("Husky Feedback Data")
        dummy_image_url = "https://plabs-assets.s3.us-west-1.amazonaws.com/Coming+Soon.png"
        st.image(dummy_image_url, caption="Husky Feedback Data", width=900)

    elif page == 'Network Density':
        st.title("Activity --  Network Density")

        st.markdown("""
            Breakdown of Network Density
        """)

        st.subheader("Filters")
        years = ["All", "2024"] 

        month_mapping = {
            "January": 1,
            "February": 2,
            "March": 3,
            "April": 4,
            "May": 5,
            "June": 6,
            "July": 7,
            "August": 8,
            "September": 9,
            "October": 10,
            "November": 11,
            "December": 12
        }

        months = [
            'January', 'February', 'March', 'April', 'May', 'June', 
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

        selected_year = st.selectbox("Select Year", years, index=0)
        selected_month = st.selectbox("Select Month", ["All"] + months, index=0)

        st.subheader("Network density by Month")
        dummy_image_url = "https://plabs-assets.s3.us-west-1.amazonaws.com/Coming+Soon.png"
        st.image(dummy_image_url, caption="Network density by Month", width=900)

        st.subheader("Telegram Network Density")
        dummy_image_url = "https://plabs-assets.s3.us-west-1.amazonaws.com/Coming+Soon.png"
        st.image(dummy_image_url, caption="Telegram Network Density", width=900)

        st.subheader("Contact Section Usage")
        dummy_image_url = "https://plabs-assets.s3.us-west-1.amazonaws.com/Coming+Soon.png"
        st.image(dummy_image_url, caption="Contact Section Usage", width=900)

        st.subheader("Telegram Activity Usage")
        dummy_image_url = "https://plabs-assets.s3.us-west-1.amazonaws.com/Coming+Soon.png"
        st.image(dummy_image_url, caption="Telegram Activity Usage", width=900)

    elif page == 'Network Strength - LongTerm Connectivity':
        st.title("Network Strength - LongTerm Connectivity")

        file_path = "./network-strength/followersfollowing.csv"
        networkStrength(file_path)

    elif page == 'Network Strength - Conversation Based Connectivity':
        st.title("Activity --  Network Strength - Conversation Based Connectivity")

        file_path = "./network-strength/Connections_TwitInteractions.csv"
        networkStrength(file_path)
       
    elif page == 'IRL Gathering':
        st.title("Activity --  IRL Gathering")

        st.markdown("""
            Breakdown of IRL Gathering
        """)

        st.subheader("Filters")
        years = ["All", "2024"] 

        month_mapping = {
            "January": 1,
            "February": 2,
            "March": 3,
            "April": 4,
            "May": 5,
            "June": 6,
            "July": 7,
            "August": 8,
            "September": 9,
            "October": 10,
            "November": 11,
            "December": 12
        }

        months = [
            'January', 'February', 'March', 'April', 'May', 'June', 
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

        selected_year = st.selectbox("Select Year", years, index=0)
        selected_month = st.selectbox("Select Month", ["All"] + months, index=0)

        st.subheader("Distribution of Events by Month")
        dummy_image_url = "https://plabs-assets.s3.us-west-1.amazonaws.com/Coming+Soon.png"
        st.image(dummy_image_url, caption="Distribution of Events by Month", width=900)

        st.subheader("Distribution of Attendees per event and month")
        dummy_image_url = "https://plabs-assets.s3.us-west-1.amazonaws.com/Coming+Soon.png"
        st.image(dummy_image_url, caption="Distribution of Attendees per event and month", width=900)

        st.subheader("Distribution of Events by Topic")
        dummy_image_url = "https://plabs-assets.s3.us-west-1.amazonaws.com/Coming+Soon.png"
        st.image(dummy_image_url, caption="Distribution of Events by Topic", width=900)

        st.subheader("Distribution of speakers by host and month")
        dummy_image_url = "https://plabs-assets.s3.us-west-1.amazonaws.com/Coming+Soon.png"
        st.image(dummy_image_url, caption="Distribution of speakers by host and month", width=900)

        st.subheader("Distribution of attendees skills, distribution of attendees topics of interest")
        dummy_image_url = "https://plabs-assets.s3.us-west-1.amazonaws.com/Coming+Soon.png"
        st.image(dummy_image_url, caption="Distribution of attendees skills, distribution of attendees topics of interest", width=900)

        st.subheader("Distribution of hosting teams by industry tags and focus areas")
        dummy_image_url = "https://plabs-assets.s3.us-west-1.amazonaws.com/Coming+Soon.png"
        st.image(dummy_image_url, caption="Distribution of hosting teams by industry tags and focus areas", width=900)

        st.subheader("Attendees with listed OH")
        dummy_image_url = "https://plabs-assets.s3.us-west-1.amazonaws.com/Coming+Soon.png"
        st.image(dummy_image_url, caption="Attendees with listed OH", width=900)

        st.subheader("Distribution of speakers and hosts")
        dummy_image_url = "https://plabs-assets.s3.us-west-1.amazonaws.com/Coming+Soon.png"
        st.image(dummy_image_url, caption="Distribution of speakers and hosts", width=900)

    elif page == 'Hackathon/Workshops':
        st.title("Activity --  Hackathon/Workshops")

        st.markdown("""
            Breakdown of Hackathon/Workshops
        """)

        st.subheader("Filters")
        years = ["All", "2024"] 

        month_mapping = {
            "January": 1,
            "February": 2,
            "March": 3,
            "April": 4,
            "May": 5,
            "June": 6,
            "July": 7,
            "August": 8,
            "September": 9,
            "October": 10,
            "November": 11,
            "December": 12
        }

        months = [
            'January', 'February', 'March', 'April', 'May', 'June', 
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

        selected_year = st.selectbox("Select Year", years, index=0)
        selected_month = st.selectbox("Select Month", ["All"] + months, index=0)

        st.subheader("Hackathon by month")
        dummy_image_url = "https://plabs-assets.s3.us-west-1.amazonaws.com/Coming+Soon.png"
        st.image(dummy_image_url, caption="Hackathon by Month", width=900)

        st.subheader("Members Contribution in Hackathon")
        dummy_image_url = "https://plabs-assets.s3.us-west-1.amazonaws.com/Coming+Soon.png"
        st.image(dummy_image_url, caption="Members Contribution in Hackathon", width=900)

    elif page == 'Network Growth':
        st.title("Activity --  Network Growth")

        st.markdown("""
            Breakdown of Network Growth
        """)

        st.subheader("Filters")
        years = ["All", "2024"] 

        month_mapping = {
            "January": 1,
            "February": 2,
            "March": 3,
            "April": 4,
            "May": 5,
            "June": 6,
            "July": 7,
            "August": 8,
            "September": 9,
            "October": 10,
            "November": 11,
            "December": 12
        }

        months = [
            'January', 'February', 'March', 'April', 'May', 'June', 
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

        selected_year = st.selectbox("Select Year", years, index=0)
        selected_month = st.selectbox("Select Month", ["All"] + months, index=0)

        st.subheader("MoM analysis on Network Growth by Month")
        dummy_image_url = "https://plabs-assets.s3.us-west-1.amazonaws.com/Coming+Soon.png"
        st.image(dummy_image_url, caption="MoM analysis on Network Growth by Month", width=900)

        st.subheader("Distribution graph of people by skills")
        dummy_image_url = "https://plabs-assets.s3.us-west-1.amazonaws.com/Coming+Soon.png"
        st.image(dummy_image_url, caption="Distribution graph of people by skills", width=900)


@st.cache_data
def load_data(file_path):
    """Loads the CSV file into a DataFrame."""
    return pd.read_csv(file_path)

def calculate_network_strength(df, total_members=2000):
    """Calculates the network strength as the percentage of unique connected members."""
    unique_members = set(df['Member']).union(set(', '.join(df['NetworkConnections']).split(', ')))
    unique_count = len(unique_members)
    network_strength = (unique_count / total_members) * 100
    return unique_count, network_strength

def calculate_member_statistics(df):
    """Calculates statistics for each member."""
    member_stats = df.groupby('Member')['NetworkConnections'].apply(
        lambda x: ', '.join(x) if x.size > 0 else '').reset_index()
    member_stats['ConnectionCount'] = member_stats['NetworkConnections'].str.split(', ').apply(len)
    return member_stats

def calculate_team_statistics(df):
    """Calculates interaction counts for each team and their statistics."""
    df['Network_Connections_Teams'] = df['Network_Connections_Teams'].fillna('').astype(str)
    team_interaction_counts = df.groupby('Member_Teams')['Network_Connections_Teams'].apply(
        lambda x: ', '.join(x) if x.size > 0 else ''
    ).reset_index()
    team_interaction_counts['Network_Connections_Teams'] = team_interaction_counts[
        'Network_Connections_Teams'].str.split(', ')
    exploded_counts = team_interaction_counts.explode('Network_Connections_Teams')
    team_stats = exploded_counts['Network_Connections_Teams'].value_counts().reset_index()
    team_stats.columns = ['Team', 'Count']
    return team_stats

def create_bubble_chart(team_stats):
    """Creates a bubble chart for team statistics."""
    # Calculate average counts
    average_count = team_stats['Count'].mean()
    max_count = team_stats['Count'].max()
    min_count = team_stats['Count'].min()

    # Prepare data for bubble chart
    stats_df = pd.DataFrame({
        'Category': ['Above Average', 'Below Average', 'Maximum', 'Minimum'],
        'Count': [
            len(team_stats[team_stats['Count'] > average_count]),
            len(team_stats[team_stats['Count'] < average_count]),
            len(team_stats[team_stats['Count'] == max_count]),
            len(team_stats[team_stats['Count'] == min_count])
        ]
    })

    fig = px.scatter(
        stats_df,
        x='Category',
        y='Count',
        size='Count',
        hover_name='Category',
        title='Team Interaction Statistics',
        labels={'Count': 'Number of Teams', 'Category': 'Interaction Category'},
        size_max=60  # Adjust the maximum size of the bubbles
    )

    st.plotly_chart(fig)

    return stats_df  # Return the stats DataFrame for further processing

def create_stacked_bar_chart(selected_category, team_stats):
    """Creates a stacked bar chart for team statistics based on the selected category."""
    if selected_category == "Above Average":
        filtered_teams = team_stats[team_stats['Count'] > team_stats['Count'].mean()]
    elif selected_category == "Below Average":
        filtered_teams = team_stats[team_stats['Count'] < team_stats['Count'].mean()]
    elif selected_category == "Maximum":
        filtered_teams = team_stats[team_stats['Count'] == team_stats['Count'].max()]
    elif selected_category == "Minimum":
        filtered_teams = team_stats[team_stats['Count'] == team_stats['Count'].min()]

    # Prepare data for stacked bar chart
    fig = px.bar(filtered_teams,
                 x='Team',
                 y='Count',
                 title=f'Teams with {selected_category} Interactions',
                 labels={'Team': 'Team', 'Count': 'Interaction Count'},
                 color='Team',
                 text=None)  # Remove text from the bar chart

    st.plotly_chart(fig)

def display_filters(df):
    """Displays the network filters and applies the selected filter."""
    avg_connections = df['ConnectionCount'].mean()
    max_connections = df['ConnectionCount'].max()
    min_connections = df['ConnectionCount'].min()

    filter_option = st.selectbox(
        "Filter by connection count",
        options=["None", "Above Average", "Below Average", "Minimum", "Maximum"]
    )

    if filter_option == "None":
        filtered_df = df
        return filter_option, filtered_df

    if filter_option == "Above Average":
        filtered_df = df[df['ConnectionCount'] > avg_connections].reset_index(drop=True)
        return filter_option,filtered_df

    elif filter_option == "Below Average":
        filtered_df = df[df['ConnectionCount'] < avg_connections].reset_index(drop=True)
        return filter_option,filtered_df

    elif filter_option == "Minimum":
        filtered_df = df[df['ConnectionCount'] == min_connections].reset_index(drop=True)
        return filter_option,filtered_df

    elif filter_option == "Maximum":
        filtered_df = df[df['ConnectionCount'] == max_connections].reset_index(drop=True)
        return filter_option,filtered_df

    return df  # Return the original dataframe if no filter is applied

def visualize_network(filtered_df):
    """Creates and displays the network graph using the filtered data."""
    net = Network(height='900px', width='100%', notebook=True)

    for _, row in filtered_df.iterrows():
        source = row['Member']
        target = row['NetworkConnections']
        if "AreaOfInterest" in filtered_df.columns:
            relationship=row['AreaOfInterest']
        else:
            relationship = row.get('Relationship', 'Twitter')

        net.add_node(source, label=source, title=f"Member: {source}", color='green', borderColor='green', size=10)
        net.add_node(target, label=target, title=f"Connection {target}", color='blue', borderColor='green', size=10)

        net.add_edge(source, target, title=f"Relationship: {relationship}", color='purple')

    options = """
    {
        "nodes": {
            "font": {
                "size": 10
            },
            "shape": "dot"
        },
        "edges": {
            "smooth": {
                "type": "continuous"
            },
            "background": {
                "color": "#ccffcc"
            }
        },
        "physics": {
            "enabled": true
        }
        
    }
    """
    net.set_options(options)
    net.show('network_graph.html')
    HtmlFile = open('network_graph.html', 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height=900, width=1100, scrolling=True)

def networkStrength(file_path):
            # Load the selected CSV file
        df = load_data(file_path)
        df['ConnectionCount'] = df['NetworkConnections'].apply(lambda x: len(x.split(',')))

        # Calculate unique members and network strength
        unique_count, network_strength = calculate_network_strength(df)

        # Display network strength
        st.metric(label="Member Network Strength", value=f"{network_strength:.2f}%")
        st.progress(network_strength / 100)

        # Team Strength Indicator
        total_teams = 600  # Assuming total teams is 600
        unique_teams_count = df['Member_Teams'].nunique()  # Calculate unique teams for strength
        team_strength_percentage = (unique_teams_count / total_teams) * 100
        st.metric(label="Team Network Strength", value=f"{team_strength_percentage:.2f}%")
        st.progress(team_strength_percentage / 100)

        # Dropdown for Members or Teams
        selection_type = st.selectbox("Select:", options=["Member", "Team"])

        if selection_type == "Member":
            # Dropdown for members
            unique_members = df['Member'].unique()
            selected_member = st.selectbox("Select a member:", options=['All'] + list(unique_members))

            if selected_member != 'All':
                filtered_df = df[(df['Member'] == selected_member) | (df['NetworkConnections'] == selected_member)]
                visualize_network(filtered_df)
            else:
                option, filtered_df = display_filters(df)
                if not filtered_df.empty:
                    st.dataframe(filtered_df, height=300)  # Display DataFrame with a scrollable view
                    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"Download Network Connectivity Data as CSV",
                        data=csv_data,
                        file_name=f"{option}_network_connectivity_data.csv",
                        mime='text/csv'
                    )
                    visualize_network(filtered_df)

        elif selection_type == "Team":
            # Show team interactions
            team_stats = calculate_team_statistics(df)
            bubble_stats = create_bubble_chart(team_stats)
            selected_category = st.selectbox("Select a category to view teams:", bubble_stats['Category'].unique())
            create_stacked_bar_chart(selected_category, team_stats)


if __name__ == "__main__":
    main()