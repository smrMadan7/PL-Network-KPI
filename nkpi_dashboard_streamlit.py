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
        'Guest User'
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
        WHEN (pe.properties->>'loggedInUserEmail' IS NULL AND pe.properties->'user'->>'email' IS NULL) THEN 'Guest'
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
                'Guest User'  
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

def fetch_events_by_month_location(engine, selected_year, selected_month):
    # Build the SQL query dynamically based on the selected year and month
    query = """
        SELECT 
            pel."location",  
            DATE_TRUNC('month', pe."startDate") AS event_month, 
            COUNT(*) AS event_count
        FROM public."PLEvent" pe
        INNER JOIN public."PLEventLocation" pel ON pe."locationUid" = pel."uid"
        WHERE pe."startDate" IS NOT NULL
    """
    
    # Apply year filter
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM pe.\"startDate\") = {selected_year}"
    
    # Apply month filter
    if selected_month != "All":
        month_num = month_mapping[selected_month]
        query += f" AND EXTRACT(MONTH FROM pe.\"startDate\") = {month_num}"

    query += """
        GROUP BY event_month, pel."location"
        ORDER BY event_month;
    """
    
    return pd.read_sql(query, engine)

# def fetch_events_by_month_by_user_and_team(engine, selected_year, selected_month, include_teams=False):
#     query = """
#         SELECT 
#             pel."location", 
#             DATE_TRUNC('month', pe."startDate") AS event_month, 
#             COUNT(DISTINCT eg."memberUid") AS event_user_count,  
#             COUNT(DISTINCT eg."teamUid") AS event_team_count  
#         FROM public."PLEvent" pe
#         INNER JOIN public."PLEventLocation" pel ON pe."locationUid" = pel."uid"
#         LEFT JOIN public."PLEventGuest" eg ON pe."uid" = eg."eventUid"
#         LEFT JOIN public."Member" m ON eg."memberUid" = m."uid"
#         LEFT JOIN public."Team" t ON eg."teamUid" = t."uid"
#         WHERE pe."startDate" IS NOT NULL
#     """
    
#     # Apply year filter
#     if selected_year != "All":
#         query += f" AND EXTRACT(YEAR FROM pe.\"startDate\") = {selected_year}"
    
#     # Apply month filter
#     if selected_month != "All":
#         month_num = month_mapping[selected_month]
#         query += f" AND EXTRACT(MONTH FROM pe.\"startDate\") = {month_num}"

#     query += """
#         GROUP BY event_month, pel."location"
#         ORDER BY event_month;
#     """
    
#     if include_teams:
#         query = query.replace("COUNT(DISTINCT eg.\"memberUid\") AS event_user_count", 
#                               "COUNT(DISTINCT eg.\"teamUid\") AS event_team_count")
    
#     return pd.read_sql(query, engine)

def fetch_events_by_month_by_user(engine, selected_year, selected_month):
    query = """
        SELECT 
            pel."location", 
            DATE_TRUNC('month', pe."startDate") AS event_month, 
            COUNT(DISTINCT eg."memberUid") AS event_user_count
        FROM public."PLEvent" pe
        INNER JOIN public."PLEventLocation" pel ON pe."locationUid" = pel."uid"
        LEFT JOIN public."PLEventGuest" eg ON pe."uid" = eg."eventUid"
        LEFT JOIN public."Member" m ON eg."memberUid" = m."uid"
        WHERE pe."startDate" IS NOT NULL
    """
    
    # Apply year filter
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM pe.\"startDate\") = {selected_year}"
    
    # Apply month filter
    if selected_month != "All":
        month_num = month_mapping[selected_month]
        query += f" AND EXTRACT(MONTH FROM pe.\"startDate\") = {month_num}"

    query += """
        GROUP BY event_month, pel."location"
        ORDER BY event_month;
    """
    
    return pd.read_sql(query, engine)   

def fetch_events_by_month_by_team(engine, selected_year, selected_month):
    query = """
        SELECT 
            pel."location", 
            DATE_TRUNC('month', pe."startDate") AS event_month, 
            COUNT(DISTINCT eg."teamUid") AS event_team_count
        FROM public."PLEvent" pe
        INNER JOIN public."PLEventLocation" pel ON pe."locationUid" = pel."uid"
        LEFT JOIN public."PLEventGuest" eg ON pe."uid" = eg."eventUid"
        LEFT JOIN public."Team" t ON eg."teamUid" = t."uid"
        WHERE pe."startDate" IS NOT NULL
    """
    
    # Apply year filter
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM pe.\"startDate\") = {selected_year}"
    
    # Apply month filter
    if selected_month != "All":
        month_num = month_mapping[selected_month]
        query += f" AND EXTRACT(MONTH FROM pe.\"startDate\") = {month_num}"

    query += """
        GROUP BY event_month, pel."location"
        ORDER BY event_month;
    """
    
    return pd.read_sql(query, engine)   


def plot_events_by_month(dataframe):
    # Convert 'event_month' to datetime format (if it's not already)
    dataframe['event_month'] = pd.to_datetime(dataframe['event_month'])
    
    # Sort the dataframe by 'event_month' to ensure chronological order
    dataframe = dataframe.sort_values(by='event_month')

    # Create the bar plot
    fig = px.bar(
        dataframe, 
        x="event_month",  # Use datetime for sorting the x-axis
        y="event_count", 
        color="location",  # Use the correct column name for location
        labels={"event_month": "Month", "event_count": "Number of Events"},
        hover_data={"location": True, "event_month": True, "event_count": True}  # Explicitly show location in hover data
    )
    
    # Customize hover information to display only relevant details
    fig.update_traces(
        hovertemplate="<b>Month:</b> %{x|%B %Y}<br><b>Location:</b> %{customdata[0]}<br><b>Number of Events:</b> %{y}<extra></extra>"
    )
    
    # Update layout settings
    fig.update_layout(
        xaxis_title="Month", 
        yaxis_title="No. of Events", 
        showlegend=True,  # Keep the legend to show locations
    )

    return fig

# def plot_events_by_month_by_user_and_team(dataframe, view_option):
#     # Convert 'event_month' to the desired format 'Month Year' (e.g., 'January 2024')
#     dataframe['event_month'] = pd.to_datetime(dataframe['event_month']).dt.strftime('%B %Y')
    
#     # Set the appropriate y-axis label and column based on the view_option
#     if view_option == "Teams":
#         y_label = "Number of Teams"
#         y_column = "event_team_count"
#     else:
#         y_label = "Number of Users"
#         y_column = "event_user_count"
    
#     # Ensure the DataFrame does not contain any missing values for the necessary columns
#     dataframe = dataframe.dropna(subset=['event_month', y_column])
    
#     # Create the bar chart using Plotly
#     fig = px.bar(
#         dataframe, 
#         x="event_month", 
#         y=y_column, 
#         color="location",  # Color by location
#         title=f"Number of {view_option} by Month",
#         labels={"event_month": "Month", y_column: y_label},
#         hover_data={"location": True, "event_month": True, y_column: True}  # Hover data to show location and the selected metric
#     )
    
#     # Update hovertemplate to show the correct information on hover
#     fig.update_traces(
#         hovertemplate="<b>Month:</b> %{x}<br><b>Location:</b> %{customdata[0]}<br><b>" +
#                       y_label + ":</b> %{y}<extra></extra>"  # Avoid f-string and use y_label directly
#     )

#     # Update layout settings
#     fig.update_layout(
#         xaxis_title="Month", 
#         yaxis_title=y_label, 
#         showlegend=True  # Keep the legend to show locations
#     )

#     return fig

def plot_events_by_month_by_user(dataframe):
    # Convert 'event_month' to the desired format 'Month Year' (e.g., 'January 2024')
    dataframe['event_month'] = pd.to_datetime(dataframe['event_month'])
    
    # Ensure the DataFrame does not contain any missing values for the necessary columns
    dataframe = dataframe.dropna(subset=['event_month', 'event_user_count'])

    # Check if 'event_user_count' column exists and is numeric
    if 'event_user_count' not in dataframe.columns:
        raise ValueError("Column 'event_user_count' does not exist in the DataFrame.")
    
    # Ensure 'event_user_count' is numeric
    dataframe['event_user_count'] = pd.to_numeric(dataframe['event_user_count'], errors='coerce')

    # Drop rows where 'event_user_count' becomes NaN after coercion
    dataframe = dataframe.dropna(subset=['event_user_count'])
    dataframe = dataframe.sort_values(by='event_month')

    # Create the bar chart using Plotly
    fig = px.bar(
        dataframe, 
        x="event_month", 
        y="event_user_count", 
        color="location",  # Color by location
        # title="Number of Users by Month",
        labels={"event_month": "Month", "event_user_count": "Number of Users"},
        hover_data={"location": True, "event_month": True, "event_user_count": True}  # Hover data to show location and the selected metric
    )
    
    # Update hovertemplate to show the correct information on hover
    fig.update_traces(
        hovertemplate="<b>Month:</b> %{x}<br><b>Location:</b> %{customdata[0]}<br><b>Number of Users:</b> %{y}<extra></extra>"
    )

    # Update layout settings
    fig.update_layout(
        xaxis_title="Month", 
        yaxis_title="No. of Users", 
        showlegend=True  # Keep the legend to show locations
    )

    return fig

def fetch_events_by_month_with_hosts_and_speakers(engine, selected_year, selected_month):
    query = """
        SELECT 
            DATE_TRUNC('month', pe."startDate") AS event_month, 
            COUNT(DISTINCT CASE WHEN eg."isHost" = TRUE THEN eg."memberUid" END) AS Host,
            COUNT(DISTINCT CASE WHEN eg."isSpeaker" = TRUE THEN eg."memberUid" END) AS Speaker
        FROM public."PLEvent" pe
        LEFT JOIN public."PLEventGuest" eg ON pe."uid" = eg."eventUid"
        WHERE pe."startDate" IS NOT NULL
    """
    
    # Apply year filter
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM pe.\"startDate\") = {selected_year}"
    
    # Apply month filter
    if selected_month != "All":
        month_num = month_mapping[selected_month]
        query += f" AND EXTRACT(MONTH FROM pe.\"startDate\") = {month_num}"

    query += """
        GROUP BY event_month
        ORDER BY event_month;
    """
    
    df = pd.read_sql(query, engine)

    # Format the event_month column to 'Month YYYY' or 'YYYY-MM'
    df['event_month'] = pd.to_datetime(df['event_month']).dt.strftime('%B %Y')  # Example: 'January 2024'
    # Or use this for YYYY-MM format: df['event_month'] = pd.to_datetime(df['event_month']).dt.strftime('%Y-%m')

    return df

def reshape_for_plotting(dataframe):
    # Melt the dataframe so that there are two columns: one for 'type' (host/speaker) and one for the 'count'
    long_df = pd.melt(
        dataframe, 
        id_vars=["event_month"],  # Keep event_month as the identifier
        value_vars=["host", "speaker"],  # Columns to reshape
        var_name="type",  # New column for type of count (host/speaker)
        value_name="count"  # New column for the actual count
    )
    return long_df

def plot_events_by_month_with_hosts_and_speakers(dataframe):
    # Reshape the dataframe for Plotly
    long_df = reshape_for_plotting(dataframe)
    
    # Ensure event_month is formatted
    long_df['event_month'] = pd.to_datetime(long_df['event_month']).dt.strftime('%B %Y')  # Example: 'January 2024'

    # Create the bar chart for hosts and speakers per month
    fig = px.bar(
        long_df, 
        x="event_month", 
        y="count",  # We are plotting counts
        color="type",  # Color by type (host/speaker)
        # title="Number of Hosts and Speakers by Month",
        labels={"event_month": "Month", "count": "Count", "type": "Role"},
        hover_data={"event_month": True, "count": True, "type": True}  # Hover with details
    )
    
    # Update layout settings for better presentation
    fig.update_layout(
        xaxis_title="Month", 
        yaxis_title="No. of Host / Speaker", 
        showlegend=True  # Keep the legend to differentiate between hosts and speakers
    )

    return fig

def fetch_attendee_data(engine, selected_year, selected_month):
    query = """
        SELECT 
            CASE 
                WHEN m."officeHours" IS NOT NULL THEN 'With Office Hours'
                ELSE 'Without Office Hours'
            END AS office_hours_status,
            COUNT(*) AS attendee_count
        FROM public."PLEventGuest" pg
        INNER JOIN public."Member" m ON pg."memberUid" = m."uid"
        INNER JOIN public."PLEvent" pe ON pg."eventUid" = pe."uid"
        WHERE pe."createdAt" IS NOT NULL
    """

    # Apply year filter if selected_year is not "All"
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM pe.\"createdAt\") = {selected_year}"

    # Apply month filter if selected_month is not "All"
    if selected_month != "All":
        month_num = month_mapping[selected_month]  # Assuming you have a month_mapping dictionary
        query += f" AND EXTRACT(MONTH FROM pe.\"createdAt\") = {month_num}"

    # Continue with the grouping and ordering
    query += """
        GROUP BY office_hours_status;
    """

    return pd.read_sql(query, engine)

def fetch_event_topic_distribution(engine, selected_year, selected_month):
    query = """
        SELECT 
            unnest("PLEventGuest"."topics") AS topic, 
            COUNT(DISTINCT "PLEventGuest"."eventUid") AS event_count
        FROM 
            public."PLEventGuest"
        JOIN 
            public."PLEvent"
        ON 
            "PLEventGuest"."eventUid" = "PLEvent"."uid"
        WHERE 
            "PLEventGuest"."topics" IS NOT NULL
    """
    
    # Apply year filter if selected_year is not "All"
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM \"PLEvent\".\"startDate\") = {selected_year}"

    # Apply month filter if selected_month is not "All"
    if selected_month != "All":
        month_num = month_mapping[selected_month]  # Convert month name to number
        query += f" AND EXTRACT(MONTH FROM \"PLEvent\".\"startDate\") = {month_num}"

    query += """
        GROUP BY 
            topic
        ORDER BY 
            event_count DESC
        LIMIT 10;
    """
    
    return pd.read_sql(query, engine)


def visualize_event_topic_distribution(engine, selected_year, selected_month):
    st.subheader("Top Events by Topic")
    st.markdown("Distribution of events by topic")
    
    # Fetch data based on selected filters
    df = fetch_event_topic_distribution(engine, selected_year, selected_month)

    # Create bar chart
    fig = px.bar(
        df,
        x="topic",
        y="event_count",
        # title="Distribution of Events by Topic",
        labels={"topic": "Topic", "event_count": "No. of Events"},
        text_auto=True,
        color="event_count",
        # color_continuous_scale="Viridis"
    )
    fig.update_layout(xaxis_title="Topic", yaxis_title="No. of Events")
    st.plotly_chart(fig)

def fetch_attendee_data_by_topic(engine, selected_year, selected_month):
    query = """
        SELECT 
            unnest("PLEventGuest"."topics") AS topic,
            COUNT(DISTINCT "PLEventGuest"."memberUid") AS attendee_count
        FROM 
            public."PLEventGuest"
        JOIN 
            public."Member"
        ON 
            "PLEventGuest"."memberUid" = "Member"."uid"
        JOIN 
            public."PLEvent"
        ON 
            "PLEventGuest"."eventUid" = "PLEvent"."uid"
        WHERE 
            "PLEventGuest"."topics" IS NOT NULL
    """
    
    # Apply year filter if selected_year is not "All"
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM \"PLEvent\".\"startDate\") = {selected_year}"

    # Apply month filter if selected_month is not "All"
    if selected_month != "All":
        month_num = month_mapping[selected_month]  # Convert month name to number
        query += f" AND EXTRACT(MONTH FROM \"PLEvent\".\"startDate\") = {month_num}"

    query += """
        GROUP BY 
            topic
        ORDER BY 
            attendee_count DESC
        LIMIT 10;
    """
    
    return pd.read_sql(query, engine)

# Function to visualize attendees by topic as a bar chart
def visualize_attendees_by_topic(engine, selected_year, selected_month):
    # st.subheader("Distribution of Attendees by Topic")
    
    # Fetch data from the database based on selected filters
    df = fetch_attendee_data_by_topic(engine, selected_year, selected_month)
    
    # Create a bar chart
    fig = px.bar(
        df, 
        x="topic", 
        y="attendee_count", 
        # title="Distribution of Attendees by Topic", 
        labels={"topic": "Topic", "attendee_count": "No. of Attendees"}, 
        text_auto=True,
        color="attendee_count",
        # color_continuous_scale=px.colors.sequential.Blues
    )
    fig.update_layout(xaxis_title="Topic", yaxis_title="No. of Attendees")
    st.plotly_chart(fig)

# Function to visualize attendee percentage by topic as a pie chart
def visualize_attendee_percentage_by_topic(engine, selected_year, selected_month):
    # st.title("Percentage of Attendees by Topic")
    
    # Fetch data from the database based on selected filters
    df = fetch_attendee_data_by_topic(engine, selected_year, selected_month)
    
    # Create a pie chart
    fig = px.pie(
        df, 
        names="topic", 
        values="attendee_count", 
        # title="Percentage of Attendees by Topic", 
        labels={"topic": "Topic", "attendee_count": "No. of Attendees"}, 
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig)

def filter_data_by_month_and_year(dataframe, selected_year, selected_month):
    # Convert the 'event_month' to datetime if it's not already in datetime format
    dataframe['event_month'] = pd.to_datetime(dataframe['event_month'], errors='coerce')
    
    # Filter by selected year
    if selected_year != "All":
        dataframe = dataframe[dataframe['event_month'].dt.year == int(selected_year)]
    
    # Filter by selected month
    if selected_month != "All":
        # Convert month name to number
        month_number = pd.to_datetime(selected_month, format='%B').month
        dataframe = dataframe[dataframe['event_month'].dt.month == month_number]
    
    return dataframe

# Function to visualize the distribution of attendees by skill
def visualize_attendees_by_skill(engine, selected_year, selected_month):
    month_mapping = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
        "All": None
    }

    # Convert selected_month to its numeric value
    selected_month_num = month_mapping.get(selected_month, None)

    # Construct the WHERE clause conditionally based on the selected year and month
    year_condition = ""
    month_condition = ""
    
    if selected_year != "All":
        year_condition = f"EXTRACT(YEAR FROM \"PLEvent\".\"startDate\") = {selected_year}"
    
    if selected_month_num is not None:
        month_condition = f"EXTRACT(MONTH FROM \"PLEvent\".\"startDate\") = {selected_month_num}"
    
    # Combine the conditions
    where_conditions = []
    if year_condition:
        where_conditions.append(year_condition)
    if month_condition:
        where_conditions.append(month_condition)
    
    where_clause = " AND ".join(where_conditions)
    if where_clause:
        where_clause = "WHERE " + where_clause
    
    # Query to fetch data for attendees by skill, considering month and year filters
    query = f"""
        SELECT 
            "Skill"."title" AS skill_name,
            COUNT(DISTINCT "Member"."uid") AS attendee_count
        FROM 
            public."Member"
        JOIN 
            public."_MemberToSkill" 
        ON 
            "Member"."id" = "_MemberToSkill"."A"
        JOIN 
            public."Skill" 
        ON
            "_MemberToSkill"."B" = "Skill"."id"
        JOIN 
            public."PLEventGuest"
        ON
            "PLEventGuest"."memberUid" = "Member"."uid"
        JOIN
            public."PLEvent"
        ON
            "PLEventGuest"."eventUid" = "PLEvent"."uid"
        {where_clause}
        GROUP BY 
            "Skill"."title"
        ORDER BY 
            attendee_count DESC
        LIMIT 10;
    """

    # Fetch data from the database with the selected year and month
    df = pd.read_sql(query, engine)

    # Create a bar chart for skill distribution
    fig = px.bar(
        df, 
        x="skill_name", 
        y="attendee_count", 
        labels={"skill_name": "Skill", "attendee_count": "Number of Attendees"},
        title=f"Distribution of Attendees by Skill in {selected_month}/{selected_year}" if selected_month != "All" and selected_year != "All" else "Distribution of Attendees by Skill",
        text_auto=True,
        color="attendee_count",
    )
    
    st.plotly_chart(fig)

# Function to visualize the percentage of attendees by skill
def visualize_attendee_percentage_by_skill(engine, selected_year, selected_month):

    # Query to fetch data for attendee percentage by skill, considering month and year filters
    query = """
        SELECT 
            "Skill"."title" AS skill_name,
            COUNT(DISTINCT "Member"."uid") AS attendee_count,
            EXTRACT(MONTH FROM "PLEvent"."startDate") AS event_month,
            EXTRACT(YEAR FROM "PLEvent"."startDate") AS event_year
        FROM 
            public."Member"
        JOIN 
            public."_MemberToSkill" 
        ON 
            "Member"."id" = "_MemberToSkill"."A"
        JOIN 
            public."Skill" 
        ON 
            "_MemberToSkill"."B" = "Skill"."id"
        JOIN 
            public."PLEventGuest"
        ON 
            "PLEventGuest"."memberUid" = "Member"."uid"
        JOIN
            public."PLEvent"
        ON
            "PLEventGuest"."eventUid" = "PLEvent"."uid"
        GROUP BY 
            "Skill"."title", event_month, event_year
        ORDER BY 
            attendee_count DESC;
    """
    # Fetch data from the database
    df = pd.read_sql(query, engine)

    # Filter the data based on selected month and year
    df = filter_data_by_month_and_year(df, selected_year, selected_month)

    # Create a pie chart
    fig = px.pie(
        df, 
        names="skill_name", 
        values="attendee_count", 
        title="Percentage of Attendees by Skill", 
        # color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig)

def fetch_hosting_teams_by_focus_area(engine, selected_year, selected_month):
    query = """
        SELECT 
            COALESCE(parentFA.title, fa.title, 'Undefined') AS focus_area, 
            COUNT(DISTINCT t.uid) AS "Teams"
        FROM 
            public."PLEventGuest" eg
        LEFT JOIN 
            public."Team" t ON eg."teamUid" = t.uid
        LEFT JOIN 
            public."TeamFocusArea" tfa ON t.uid = tfa."teamUid"
        LEFT JOIN 
            public."FocusArea" fa ON fa.uid = tfa."focusAreaUid"
        LEFT JOIN 
            public."FocusAreaHierarchy" fah ON fa.uid = fah."subFocusAreaUid"
        LEFT JOIN 
            public."FocusArea" parentFA ON fah."focusAreaUid" = parentFA.uid
        WHERE 
            eg."isHost" = TRUE
    """
    
    # Apply year and month filters if provided
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM eg.\"createdAt\") = {selected_year}"
    
    if selected_month != "All":
        month_num = month_mapping[selected_month]
        query += f" AND EXTRACT(MONTH FROM eg.\"createdAt\") = {month_num}"

    query += """
        GROUP BY 
            COALESCE(parentFA.title, fa.title, 'Undefined')
        ORDER BY "Teams" DESC;
    """
    
    return pd.read_sql(query, engine)

def plot_hosting_teams_by_focus_area(dataframe):
    # Ensure the DataFrame does not contain any missing values for the necessary columns
    dataframe = dataframe.dropna(subset=['focus_area', 'Teams'])

    # Check if 'Teams' column exists and is numeric
    if 'Teams' not in dataframe.columns:
        raise ValueError("Column 'Teams' does not exist in the DataFrame.")
    
    # Ensure 'Teams' is numeric
    dataframe['Teams'] = pd.to_numeric(dataframe['Teams'], errors='coerce')

    # Drop rows where 'Teams' becomes NaN after coercion
    dataframe = dataframe.dropna(subset=['Teams'])

    # Create the bar chart using Plotly
    fig = px.bar(
        dataframe, 
        x="focus_area", 
        y="Teams", 
        # title="Number of Hosting Teams by Focus Area",
        labels={"focus_area": "Focus Area", "Teams": "No. of Hosting Teams"},
        hover_data={"focus_area": True, "Teams": True}  # Hover data to show focus area and number of teams
    )
    
    # Update hovertemplate to show the correct information on hover
    fig.update_traces(
        hovertemplate="<b>Focus Area:</b> %{x}<br><b>Number of Hosting Teams:</b> %{y}<extra></extra>"
    )

    # Update layout settings
    fig.update_layout(
        xaxis_title="Focus Area", 
        yaxis_title="No. of Hosting Teams", 
        showlegend=False  # Hide the legend since we only have one bar category
    )

    return fig

def fetch_hosts_and_speakers_by_month(engine, selected_year, selected_month):
    # Begin the base SQL query
    query = """
        SELECT 
            DATE_TRUNC('month', pe."startDate") AS event_month, 
            COUNT(DISTINCT CASE WHEN eg."isHost" = TRUE AND eg."memberUid" IS NOT NULL THEN eg."memberUid" END) AS "host_members", 
            COUNT(DISTINCT CASE WHEN eg."isHost" = TRUE AND eg."teamUid" IS NOT NULL THEN eg."teamUid" END) AS "host_teams", 
            COUNT(DISTINCT CASE WHEN eg."isSpeaker" = TRUE AND eg."memberUid" IS NOT NULL THEN eg."memberUid" END) AS "speaker_members", 
            COUNT(DISTINCT CASE WHEN eg."isSpeaker" = TRUE AND eg."teamUid" IS NOT NULL THEN eg."teamUid" END) AS "speaker_teams"
        FROM 
            public."PLEvent" pe
        INNER JOIN 
            public."PLEventGuest" eg ON pe."uid" = eg."eventUid"
        LEFT JOIN 
            public."Member" m ON eg."memberUid" = m."uid"
        LEFT JOIN 
            public."Team" t ON eg."teamUid" = t."uid"
        WHERE 
            pe."startDate" IS NOT NULL
            AND (eg."isHost" = TRUE OR eg."isSpeaker" = TRUE)
    """

    # Apply year filter if not "All"
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM pe.\"startDate\") = {selected_year}"

    # Apply month filter if not "All"
    if selected_month != "All":
        month_num = month_mapping[selected_month]  # Assuming month_mapping is defined somewhere
        query += f" AND EXTRACT(MONTH FROM pe.\"startDate\") = {month_num}"

    # Continue with the grouping and ordering
    query += """
        GROUP BY 
            event_month
        ORDER BY 
            event_month;
    """
    
    # Execute the query and return the result as a DataFrame
    return pd.read_sql(query, engine)


def plot_hosts_and_speakers_distribution(dataframe):
    # Convert event_month to a readable string format
    dataframe['event_month'] = pd.to_datetime(dataframe['event_month']).dt.strftime('%B %Y')

    # Melt the dataframe to make it suitable for a stacked bar chart
    long_df = pd.melt(dataframe, id_vars=['event_month'], 
                      value_vars=['host_members', 'host_teams', 'speaker_members', 'speaker_teams'],
                      var_name='role_type', value_name='count')

    # Create a mapping to display role types in the desired format
    role_type_map = {
        'host_members': 'Hosts(Members)',
        'host_teams': 'Hosts(Teams)',
        'speaker_members': 'Speakers(Members)',
        'speaker_teams': 'Speakers(Teams)'
    }

    # Replace the role_type values using the mapping
    long_df['role_type'] = long_df['role_type'].map(role_type_map)

    # Create the bar chart using Plotly
    fig = px.bar(long_df, 
                 x="event_month", 
                 y="count", 
                 color="role_type", 
                #  title="Distribution of Hosts and Speakers by Month",
                 labels={"event_month": "Month", "count": "No. of Hosts/Speakers", "role_type": "Role Type"},
                 category_orders={"role_type": ["Hosts(Members)", "Hosts(Teams)", "Speakers(Members)", "Speakers(Teams)"]},
                 hover_data={"event_month": True, "role_type": True, "count": True})
    
    # Update layout settings for the chart
    fig.update_layout(
        xaxis_title="Month", 
        yaxis_title="No. of Hosts/Speakers", 
        showlegend=True
    )

    return fig


def fetch_active_users(engine):
    query = """
        SELECT
            EXTRACT(YEAR FROM timestamp) AS year,
            EXTRACT(MONTH FROM timestamp) AS month,
            COUNT(
                DISTINCT COALESCE(
                    properties->>'userName',
                    properties->>'loggedInUserName',
                    properties->'user'->>'name'
                )
            ) AS active_user_count
        FROM 
            public.posthogevents
        WHERE
            properties->>'$session_id' IS NOT NULL
        GROUP BY 
            year, month
        HAVING 
            COUNT(*) > 1
        ORDER BY 
            year, month;
    """
    return pd.read_sql(query, engine)

# def calculate_mom_growth(df):
#     df['month_year'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
#     df = df.sort_values(by='month_year')
    
#     df['prev_active_user_count'] = df['active_user_count'].shift(1)
#     df['mom_growth'] = (
#         (df['active_user_count'] - df['prev_active_user_count'])
#         / df['prev_active_user_count']
#     ) * 100
#     return df

def calculate_mom_growth(df):
    df['month_year'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df = df.sort_values(by='month_year')
    
    df['prev_active_user_count'] = df['active_user_count'].shift(1)
    
    df['mom_growth'] = (
        (df['active_user_count'] - df['prev_active_user_count'])
        / df['prev_active_user_count']
    ) * 100
    
    df['mom_growth'].fillna(0, inplace=True)
    
    return df

def fetch_average_session_time(engine):
    query = """
        WITH session_durations AS (
            SELECT 
                properties->>'$session_id' AS session_id,
                EXTRACT(EPOCH FROM MAX(timestamp) - MIN(timestamp)) AS session_duration_seconds
            FROM 
                public.posthogevents
            WHERE 
                properties->>'$session_id' IS NOT NULL
            GROUP BY 
                properties->>'$session_id'
        )
        SELECT 
            FLOOR(AVG(session_duration_seconds) / 60) AS average_duration_minutes,
            MOD(AVG(session_duration_seconds), 60) AS average_duration_seconds
        FROM 
            session_durations;
    """
    return pd.read_sql(query, engine)

def fetch_events_by_month(engine, selected_year, selected_month):
    # Build the SQL query dynamically based on the selected year and month
    query = """
    SELECT
        CASE
            WHEN properties->'$set'->>'$current_url' = 'https://directory.plnetwork.io/members'
            THEN 'Member Landing Page'
            
            WHEN properties->'$set'->>'$current_url' = 'https://directory.plnetwork.io/projects'
            THEN 'Project Landing Page'
            
            WHEN properties->'$set'->>'$current_url' = 'https://directory.plnetwork.io/irl'
            THEN 'IRL Landing Page'
            
            WHEN properties->'$set'->>'$current_url' = 'https://directory.plnetwork.io/teams'
            THEN 'Team Landing Page'
            
            WHEN properties->'$set'->>'$current_url' LIKE 'https://directory.plnetwork.io/members%' 
            THEN 'Member Details Page'
            
            WHEN properties->'$set'->>'$current_url' LIKE 'https://directory.plnetwork.io/projects%' 
            THEN 'Project Details Page'
            
            WHEN properties->'$set'->>'$current_url' LIKE 'https://directory.plnetwork.io/teams%' 
            THEN 'Team Details Page'
            
            WHEN properties->'$set'->>'$current_url' LIKE 'https://directory.plnetwork.io/irl%' 
            THEN 'IRL Details Page'

            ELSE 'Other'
        END AS page_type,
        COUNT(*) AS event_count
    FROM 
        public.posthogevents
    WHERE
        properties->'$set'->>'$current_url' IS NOT NULL
    """
    
    # Apply year filter
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM timestamp) = {selected_year}"
    
    # Apply month filter
    if selected_month != "All":
        month_num = month_mapping[selected_month]
        query += f" AND EXTRACT(MONTH FROM timestamp) = {month_num}"

    query += """
    GROUP BY page_type
    ORDER BY page_type;
    """
    
    # Execute the query and return the result as a DataFrame
    return pd.read_sql(query, engine)

# Function to plot the events by page type
def plot_events_by_page_type(df):
    # Plot a bar chart
    fig = px.bar(
        df, 
        x="page_type", 
        y="event_count", 
        # title="Event Count by Page Type",
        labels={"page_type": "Page Type", "event_count": "Event Count"},
        color="page_type",  # Color by page_type
        hover_data={"page_type": True, "event_count": True}
    )

    # Customize the layout
    fig.update_layout(
        xaxis_title="Page Type",
        yaxis_title="Number of Events",
        showlegend=False
    )

    return fig

import psycopg2

def fetch_data_from_db_pagetype(engine, selected_year, selected_month):
    # Define the query
    query = """
    SELECT
        CASE
            WHEN properties->'$set'->>'$current_url' = 'https://directory.plnetwork.io/members'
            THEN 'Member Landing Page'
            WHEN properties->'$set'->>'$current_url' = 'https://directory.plnetwork.io/projects'
            THEN 'Project Landing Page'
            WHEN properties->'$set'->>'$current_url' = 'https://directory.plnetwork.io/irl'
            THEN 'IRL Landing Page'
            WHEN properties->'$set'->>'$current_url' = 'https://directory.plnetwork.io/teams'
            THEN 'Team Landing Page'
            WHEN properties->'$set'->>'$current_url' LIKE 'https://directory.plnetwork.io/members%%' 
            THEN 'Member Details Page'
            WHEN properties->'$set'->>'$current_url' LIKE 'https://directory.plnetwork.io/projects%%' 
            THEN 'Project Details Page'
            WHEN properties->'$set'->>'$current_url' LIKE 'https://directory.plnetwork.io/teams%%' 
            THEN 'Team Details Page'
            WHEN properties->'$set'->>'$current_url' LIKE 'https://directory.plnetwork.io/irl%%' 
            THEN 'IRL Details Page'
            ELSE 'Other'
        END AS page_type,
        EXTRACT(MONTH FROM timestamp) AS month,
        COUNT(*) AS event_count
    FROM public.posthogevents
    WHERE properties->'$set'->>'$current_url' IS NOT NULL
    """

    # Convert selected_month to its numeric value
    selected_month_num = month_mapping.get(selected_month, None)

    # Apply filters based on the selected year and month
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM timestamp) = {selected_year}"
    
    if selected_month != "All":
        query += f" AND EXTRACT(MONTH FROM timestamp) = {selected_month_num}"

    query += """
    GROUP BY page_type, month
    ORDER BY month, page_type;
    """

    # Execute the query and load the result into a DataFrame
    return pd.read_sql(query, engine)


def plot_events_by_page_type(df):
    # Plot a bar chart with page_type as the legend
    fig = px.bar(
        df, 
        x="month", 
        y="event_count", 
        color="page_type",  # Color by page_type
        labels={"month": "Month", "event_count": "Event Count", "page_type": "Page Type"},
        barmode="group",  # Group bars by month
        hover_data={"page_type": True, "event_count": True, "month": True}
    )

    # Customize the layout
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Events",
        showlegend=True,
        title="Event Count by Page Type for Each Month"
    )

    # Update the x-axis to display month names instead of numbers
    fig.update_xaxes(
        tickmode='array',
        tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )

    return fig

def fetch_data_from_db(engine, selected_year, selected_month):
    # Base SQL query with page type and entity name extraction
    query = """
    SELECT
        CASE
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/members/%%' THEN 'Member Detail'
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/members?search%%' THEN 'Member Search'
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/projects/%%' THEN 'Project Detail'
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/projects?search%%' THEN 'Project Search'
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/teams/%%' THEN 'Team Detail'
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/teams?search%%' THEN 'Team Search'
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/irl/%%' THEN 'IRL Detail'
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/irl?search%%' THEN 'IRL Search'
            ELSE 'Other'
        END AS page_type,
        COUNT(*) AS event_count,
        -- Member Name (join with the 'members' table using the UID from the URL)
        CASE
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/members/%%' THEN m.name
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/projects/%%' THEN p.name
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/teams/%%' THEN t.name
            ELSE NULL
        END AS entity_name,
        -- For Project Search, extract the value after 'search='
        CASE
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/projects?search=%%' THEN
                SPLIT_PART(SUBSTRING(properties->'$set'->>'$current_url' FROM 'search=([^&]+)'), '=', 2)
            ELSE NULL
        END AS project_search_value,
        -- For Member Search, extract the value after 'search='
        CASE
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/members?search=%%' THEN
                SPLIT_PART(SUBSTRING(properties->'$set'->>'$current_url' FROM 'search=([^&]+)'), '=', 2)
            ELSE NULL
        END AS member_search_value,
        -- For Team Search, extract the value after 'search='
        CASE
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/teams?search=%%' THEN
                SPLIT_PART(SUBSTRING(properties->'$set'->>'$current_url' FROM 'search=([^&]+)'), '=', 2)
            ELSE NULL
        END AS team_search_value
    FROM public.posthogevents
    -- Join the members table based on the extracted member UID from the URL
    LEFT JOIN public."Member" m ON properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/members/%%' AND m.uid = SPLIT_PART(properties->'$set'->>'$current_url', '/', 5)
    -- Join the projects table based on the extracted project UID from the URL
    LEFT JOIN public."Project" p ON properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/projects/%%' AND p.uid = SPLIT_PART(properties->'$set'->>'$current_url', '/', 5)
    -- Join the teams table based on the extracted team UID from the URL
    LEFT JOIN public."Team" t ON properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/teams/%%' AND t.uid = SPLIT_PART(properties->'$set'->>'$current_url', '/', 5)
    WHERE properties->'$set'->>'$current_url' IS NOT NULL
    """

    # Convert selected_month to its numeric value
    selected_month_num = month_mapping.get(selected_month, None)

    # Apply filters based on the selected year and month
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM timestamp) = {selected_year}"
    
    if selected_month != "All":
        query += f" AND EXTRACT(MONTH FROM timestamp) = {selected_month_num}"

    query += """
    GROUP BY page_type, entity_name, project_search_value, member_search_value, team_search_value
    ORDER BY page_type, event_count DESC;
    """
    
    # Execute the query and return the results as a DataFrame
    return pd.read_sql(query, engine)

def create_bar_chart(data, selected_page_type):
    # Filter the data by the selected page type
    filtered_data = data[data['page_type'] == selected_page_type]
    
    # Create a new column combining entity name and search value to make the chart more informative
    # This helps to show a more meaningful label on the x-axis (e.g., member name or project name)
    if selected_page_type == 'Member Search':
        filtered_data['display_label'] = filtered_data['member_search_value'].fillna('No Search')
    elif selected_page_type == 'Project Search':
        filtered_data['display_label'] = filtered_data['project_search_value'].fillna('No Search')
    elif selected_page_type == 'Team Search':
        filtered_data['display_label'] = filtered_data['team_search_value'].fillna('No Search')
    else:
        # For details (non-search types), just use the entity name
        filtered_data['display_label'] = filtered_data['entity_name']

    # Create a bar chart using Plotly
    fig = px.bar(
        filtered_data,
        x='display_label',  # Use the new label for x-axis
        y='event_count',
        title=f'Event Count for {selected_page_type}',
        labels={'display_label': 'Entity Name / Search Term', 'event_count': 'Event Count', 'page_type': 'Page Type'},
        color='display_label',  # Optional: add color based on entity/search value for distinction
        category_orders={'display_label': filtered_data['display_label'].unique().tolist()},  # To order by label
        text='event_count',  # Show event count on bars
    )
    
    # Display the chart in the Streamlit app
    st.plotly_chart(fig)

def fetch_data_network_growth(engine, table_name):
    query = f"""
    SELECT
        DATE_TRUNC('month', "createdAt") AS month,
        COUNT(DISTINCT "uid") AS new_entries,
        LAG(COUNT(DISTINCT "uid")) OVER (ORDER BY DATE_TRUNC('month', "createdAt")) AS previous_month_entries,
        CASE 
            WHEN LAG(COUNT(DISTINCT "uid")) OVER (ORDER BY DATE_TRUNC('month', "createdAt")) > 0 THEN
                ((COUNT(DISTINCT "uid") - LAG(COUNT(DISTINCT "uid")) OVER (ORDER BY DATE_TRUNC('month', "createdAt")))::DECIMAL /
                 LAG(COUNT(DISTINCT "uid")) OVER (ORDER BY DATE_TRUNC('month', "createdAt"))) * 100
            ELSE 
                NULL
        END AS growth_percentage
    FROM 
        public."{table_name}"
    WHERE 
        "createdAt" IS NOT NULL
    GROUP BY 
        DATE_TRUNC('month', "createdAt")
    ORDER BY 
        month;
    """
    return pd.read_sql(query, engine)

# def visualize_mom_analysis(df, entity_name):
#     """
#     Visualize Month-over-Month analysis for a given entity.
    
#     Args:
#     - df: Pandas DataFrame with `month`, `new_entries`, and `growth_percentage`.
#     - entity_name: Name of the entity (e.g., 'Project' or 'Team').
#     """
#     # Ensure 'month' column is in datetime format
#     df['month'] = pd.to_datetime(df['month'])
    
#     # Set up the figure and axis
#     fig, ax1 = plt.subplots(figsize=(12, 6))

#     # Plot new entries as a line plot
#     sns.lineplot(data=df, x='month', y='new_entries', marker='o', label='New Entries', ax=ax1, color='blue')
#     ax1.set_ylabel('New Entries', color='blue')
#     ax1.tick_params(axis='y', labelcolor='blue')
#     ax1.set_xlabel('Month')
#     ax1.set_title(f"Month-over-Month Analysis: {entity_name}")

#     # Plot growth percentage as a bar chart on the secondary y-axis
#     ax2 = ax1.twinx()
#     sns.barplot(data=df, x='month', y='growth_percentage', alpha=0.6, ax=ax2, color='green')
#     ax2.set_ylabel('Growth Percentage (%)', color='green')
#     ax2.tick_params(axis='y', labelcolor='green')

#     # Rotate x-axis labels for better readability
#     plt.xticks(rotation=45)
#     plt.tight_layout()

#     # Add legend
#     ax1.legend(loc='upper left')
#     ax2.legend(['Growth Percentage'], loc='upper right')

#     # Show the plot
#     plt.show()

def plot_mom_analysis(df, title):
    """
    Function to plot Month-over-Month (MoM) growth analysis using Plotly.
    """
    # Format 'month' to a string 'YYYY-MM' for better visualization
    df['formatted_month'] = df['month'].dt.strftime('%Y-%m')

    # Ensure there are no missing values for growth_percentage
    df = df.dropna(subset=['growth_percentage'])

    # Create the plot
    fig = px.line(
        df,
        x='formatted_month',
        y='growth_percentage',
        title=title,
        labels={'formatted_month': 'Month-Year', 'growth_percentage': 'MoM Growth (%)'},
        markers=True
    )
    
    # Show the plot in Streamlit
    st.plotly_chart(fig)


def fetch_search_event_data(engine, event_name, selected_year, selected_month):
    query = f"""
    SELECT 
        COALESCE(
            properties->>'loggedInUserEmail', 
            properties->'user'->>'email'
        ) AS email,
        COALESCE(
            NULLIF(properties->>'loggedInUserName', ''),  
            NULLIF(properties->'user'->>'name', ''), 
            'Guest User'  
        ) AS name,
        COUNT(*) AS event_count
    """
    
    # Add the conditional search value based on the event_name
    if event_name == 'projects_search':
        query += ", COALESCE(properties->>'search', properties->>'name') AS search_value"
    elif event_name == 'team_search':
        query += ", properties->>'value' AS search_value"
    elif event_name == 'member-list-search-clicked':
        query += ", properties->>'searchValue' AS search_value"
    elif event_name == 'project_search':
        query += ", COALESCE(properties->>'searchQuery', properties->>'searchTerm') AS search_value"
    else:
        query += ", NULL AS search_value"  # Default case if event_name doesn't match any

    query += f"""
    FROM posthogevents
    WHERE "event" = '{event_name}' 
    AND (
        (
            COALESCE(
                properties->>'loggedInUserName', 
                properties->'user'->>'name'
            ) NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
        )
        OR 
        (properties->>'loggedInUserName' IS NULL AND properties->'user'->>'name' IS NULL)
    )
    """

    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM timestamp) = {selected_year}"

    if selected_month != "All":
        month_number = month_mapping[selected_month]
        query += f" AND EXTRACT(MONTH FROM timestamp) = {month_number}"

    query += """
    GROUP BY email, name, search_value
    ORDER BY event_count DESC
    LIMIT 10;
    """
    
    # Replace with your actual database engine
    return pd.read_sql(query, engine)


def fetch_search_event_data_by_month(engine, event_type, selected_year, selected_month):
    query = f"""
    SELECT 
        DATE_TRUNC('month', timestamp) AS month_start,  
        COUNT(*) AS event_count
    FROM posthogevents
    WHERE "event" = '{event_type}' 
    AND (
        -- Exclude the specific users by name (whether logged-in or logged-out)
        (
            COALESCE(
                properties->>'loggedInUserName', 
                properties->'user'->>'name'
            ) NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
        )
        -- Include logged-out users (where loggedInUserName or user->name is NULL)
        OR 
        (properties->>'loggedInUserName' IS NULL AND properties->'user'->>'name' IS NULL)
    )
    """

    # Apply year and month filters
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM timestamp) = {selected_year}"

    if selected_month != "All":
        month_number = month_mapping[selected_month]
        query += f" AND EXTRACT(MONTH FROM timestamp) = {month_number}"

    query += """
    GROUP BY month_start
    ORDER BY month_start;
    """
    return pd.read_sql(query, engine)

def fetch_data_by_contact_event(selected_type, selected_year, selected_month):
    query = f"""
    SELECT 
        COALESCE(
            NULLIF(TRIM(properties->>'loggedInUserName'), ''),  -- Treat empty strings as NULL and trim whitespace
            NULLIF(TRIM(properties->'user'->>'name'), ''), 
            'Guest User'  -- Default value if name is missing or empty
        ) AS name,
        COALESCE(properties->>'type', 'Unknown') AS type,
        COUNT(*) AS event_count
    FROM posthogevents
    WHERE "event" = 'member-detail-social-link_clicked'
      AND (
        -- Exclude the specific users by name (whether logged-in or logged-out)
        (
            COALESCE(
                properties->>'loggedInUserName', 
                properties->'user'->>'name'
            ) NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
        )
        -- Include logged-out users (where loggedInUserName or user->name is NULL)
        OR 
        (properties->>'loggedInUserName' IS NULL AND properties->'user'->>'name' IS NULL)
    )
    """
    
    # Apply type filter
    if selected_type != "All":
        query += f" AND COALESCE(properties->>'type', 'Unknown') = '{selected_type}'"

    # Apply year filter
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM timestamp) = {selected_year}"

    # Apply month filter
    if selected_month != "All":
        month_number = month_mapping[selected_month]
        query += f" AND EXTRACT(MONTH FROM timestamp) = {month_number}"

    query += """
    GROUP BY name, type
    ORDER BY event_count DESC
    LIMIT 10;
    """
    return query


def main():
    st.set_page_config(page_title="nKPI Dashboard", layout="wide")

    page = st.sidebar.radio(
        "nKPI Dashboard",
        [
           "Activity - Directory Teams",
            "Office Hours Usage",
            "Directory MAUs",
            "Network Density",
            "IRL Gatherings",
            "Network Growth",
            # "Hackathons",
            "Search Usage"
        ]
    )

    engine = get_database_connection()

    if page == "Activity - Directory Teams":
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
        selected_user_status = st.selectbox("Select User Status", user_status_options, index=0, key="user_status_selectbox")

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


    elif page == "Office Hours Usage":
        st.title("Office Hours Usage")
        
        st.markdown("""
            Breakdown of OH activity on Member and Team Profile
        """)

        df = load_data("./OH/OH-Data-Members.csv")

        # # Display the raw data
        # st.subheader("Raw Data")
        # st.dataframe(df)
# no_users_w_oh,precent_users_w_oh,month,Year,total_users,no_teams_w_oh,percent_teams_w_oh,total_teams

 
        # Combine `month` and `Year` columns to create a `month_year` column
        # df['month_year'] = df['month'] + " " + df['Year'].astype(str)

        try:
            # Convert `month` to datetime (assuming it has both month and year)
            df['month'] = pd.to_datetime(df['month'], format='%b %Y')
        except ValueError:
            st.error("Ensure the `month` column is in the format 'Nov 2022'.")
            st.stop()

        df = load_data("./OH/OH-Data-Members.csv")

        # Convert `month` to datetime for sorting
        df['month'] = pd.to_datetime(df['month'], format='%b %Y')
        df = df.sort_values(by='month')


        # ---- Bar Chart for Counts ----
        st.subheader("Bar Chart: Counts for Users and Teams with Office Hours")

        # Create the count bar chart
        fig_count = go.Figure()

        # Add bars for `no_users_w_oh`
        fig_count.add_trace(go.Bar(
            x=df['month'].dt.strftime('%b %Y'),
            y=df['no_users_w_oh'],
            name='Users with Office Hours (Count)',
            marker_color='skyblue',
            hoverinfo='text',
            text=[f"{val} Users" for val in df['no_users_w_oh']]
        ))

        # Add bars for `no_teams_w_oh`
        fig_count.add_trace(go.Bar(
            x=df['month'].dt.strftime('%b %Y'),
            y=df['no_teams_w_oh'],
            name='Teams with Office Hours (Count)',
            marker_color='red',
            hoverinfo='text',
            text=[f"{val} Teams" for val in df['no_teams_w_oh']]
        ))

        # Customize layout for the count chart
        fig_count.update_layout(
            title="Users and Teams with Office Hours (Count)",
            xaxis_title="Month",
            yaxis_title="Count",
            barmode='stack',  # Grouped bar chart
            xaxis_tickangle=-45,
            showlegend=True
        )

        # Display the count bar chart
        st.plotly_chart(fig_count)

        # ---- Bar Chart for Percentages ----
        st.subheader("Bar Chart: Percentages for Users and Teams with Office Hours")

        # Create the percentage bar chart
        fig_percent = go.Figure()

        # Add bars for `precent_users_w_oh`
        fig_percent.add_trace(go.Bar(
            x=df['month'].dt.strftime('%b %Y'),
            y=df['precent_users_w_oh'],
            name='Users with Office Hours (%)',
            marker_color='skyblue',
            hoverinfo='text',
            text=[f"{val}% Users" for val in df['precent_users_w_oh']]
        ))

        # Add bars for `percent_teams_w_oh`
        fig_percent.add_trace(go.Bar(
            x=df['month'].dt.strftime('%b %Y'),
            y=df['percent_teams_w_oh'],
            name='Teams with Office Hours (%)',
            marker_color='red',
            hoverinfo='text',
            text=[f"{val}% Teams" for val in df['percent_teams_w_oh']]
        ))

        # Customize layout for the percentage chart
        fig_percent.update_layout(
            title="Users and Teams with Office Hours (Percentage)",
            xaxis_title="Month",
            yaxis_title="Percentage",
            barmode='stack',  
            xaxis_tickangle=-45,
            showlegend=True
        )

        # Display the percentage bar chart
        st.plotly_chart(fig_percent)


        total_scheduled = 62
        confirmed_meetings = 19
        gave_feedback = 43
        did_not_give_feedback = 62 - gave_feedback  # Total scheduled - feedback given
        nps_from_feedback = "80%"

        # Pie chart for the number of confirmed meetings vs total scheduled
        fig_confirmed_meetings = go.Figure(data=[go.Pie(
            labels=["Confirmed Meetings", "Unconfirmed Meetings","Gave Feedback", "Did Not Give Feedback"],
            values=[confirmed_meetings, total_scheduled - confirmed_meetings,gave_feedback, did_not_give_feedback],
            marker=dict(colors=["#1f77b4", "#ff7f0e","#2ca02c", "#d62728"]),
            hoverinfo="label+percent",
            textinfo="value+percent"
        )])

    

        # Streamlit layout
        st.title("Meeting Feedback Analysis")

        # Display Pie charts
        st.plotly_chart(fig_confirmed_meetings)


        st.title("Aggregated OH Data")
        df = load_data("./OH/OH Data-AggregatedOHs.csv")

        # Check if the CSV has the necessary columns
        if "Date" in df.columns and "user-oh" in df.columns and "irl-user-oh" in df.columns and "team-oh" in df.columns and "combined-oh" in df.columns:
            
            # Convert the 'Date' column to datetime for better handling
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
            # Create the bar chart
            st.subheader("Bar Chart of Office Hours")

            # Create a bar chart using Plotly
            fig = go.Figure()

            # Add bars for each category
            fig.add_trace(go.Bar(
                x=df['Date'],
                y=df['user-oh'],
                name='User OH',
                marker_color='skyblue',
                hoverinfo='text',
                text=[f"{val} User OH" for val in df['user-oh']]
            ))

            fig.add_trace(go.Bar(
                x=df['Date'],
                y=df['irl-user-oh'],
                name='IRL User OH',
                marker_color='lightgreen',
                hoverinfo='text',
                text=[f"{val} IRL User OH" for val in df['irl-user-oh']]
            ))

            fig.add_trace(go.Bar(
                x=df['Date'],
                y=df['team-oh'],
                name='Team OH',
                marker_color='red',
                hoverinfo='text',
                text=[f"{val} Team OH" for val in df['team-oh']]
            ))

            fig.add_trace(go.Bar(
                x=df['Date'],
                y=df['combined-oh'],
                name='Combined OH',
                marker_color='lightcoral',
                hoverinfo='text',
                text=[f"{val} Combined OH" for val in df['combined-oh']]
            ))

            # Update layout for better visualization
            fig.update_layout(
                title="Office Hours for Different Categories Over Time",
                xaxis_title="Date",
                yaxis_title="Office Hours",
                barmode='stack',  # Stacked bar chart
                xaxis_tickangle=-45,  # Rotate x-axis labels for readability
                showlegend=True
            )

            # Display the Plotly chart in Streamlit
            st.plotly_chart(fig)

        else:
            st.error("The CSV file is missing some required columns. Please ensure the file contains 'Date', 'user-oh', 'irl-user-oh', 'team-oh', and 'combined-oh' columns.")



        
        st.title("Responses PMF-v1")
        df = load_data("./OH/OH Data-PMFv1.csv")
        # st.subheader("Directory in app survey - How would you feel if you could no longer use the PL Directory?")
            # Check if the CSV has the necessary columns
        if "Response Category" in df.columns and "Count" in df.columns and "Percentage" in df.columns:
            
        
            # Pie chart using Plotly
            fig = go.Figure(data=[go.Pie(
                labels=df['Response Category'],
                values=df['Count'],
                hoverinfo='label+percent',  # Show label and percentage
                textinfo='percent',  # Display percentage on the chart
                marker=dict(colors=["#1f77b4", "#ff7f0e", "#2ca02c"])  # Customize the colors
            )])

            # Update layout for better visualization
            fig.update_layout(
                title="Distribution of Response Categories",
                showlegend=True
            )

            # Display the Plotly pie chart
            st.plotly_chart(fig)
            
        else:
            st.error("The CSV file is missing some required columns. Please ensure the file contains 'Response Category', 'Count', and 'Percentage' columns.")

    elif page == "Directory MAUs":
        st.title("Active User Analysis")

        df = fetch_average_session_time(engine)
        avg_minutes = int(df['average_duration_minutes'][0])
        avg_seconds = int(df['average_duration_seconds'][0])

        # Display the results
        # st.header("Session Duration Summary")
        # st.metric("Average Session Duration", f"{avg_minutes} min {avg_seconds} sec")

        st.markdown(
            f"""
            <div style="background-color:#FFD700; padding:20px; border-radius:8px; text-align:center;">
                <h3>Average Session Duration</h3>
                <p style="font-size:28px; font-weight:bold; color:#2b2b2b;">{avg_minutes} min {avg_seconds} sec</p>
            </div>
            """,
            unsafe_allow_html=True
        )

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

        selected_year = st.selectbox("Select Year", years, index=0, key="year_selectbox")
        selected_month = st.selectbox("Select Month", ["All"] + months, index=0,  key="month_selectbox")

        df = fetch_active_users(engine)
        df = calculate_mom_growth(df)

        if selected_year != "All":
            df = df[df['year'] == int(selected_year)]

        if selected_month != "All":
            month_number = month_mapping[selected_month]
            df = df[df['month'] == month_number]

        st.subheader("Monthly Active Users")

        st.markdown("Active users are those who have engaged in more than one activity during their login session.")
        df['formatted_month'] = df['month_year'].dt.strftime('%b %Y')  # Format X-axis labels

        fig_active_users = px.bar(
            df,
            x='formatted_month',
            y='active_user_count',
            # title="Monthly Active User Count",
            labels={'formatted_month': 'Month-Year', 'active_user_count': 'Active User Count'},
            text_auto=True
        )
        st.plotly_chart(fig_active_users)

        st.subheader("MoM Active User Growth Analysis")
        # fig_mom_growth = px.area(
        #     df,
        #     x='month_year',  # X-axis: Month-Year   
        #     y='mom_growth',  # Y-axis: MoM Growth
        #     title="MoM Growth Analysis",  # Title of the chart
        #     labels={'month_year': 'Month-Year', 'mom_growth': 'MoM Growth (%)'},
        #     markers=True
        # )
        fig = px.area(df, x="formatted_month", y="active_user_count", 
              title="MOM Analysis: User Count", 
              labels={'formatted_month': 'Month-Year', 'active_user_count': 'Active User Count'},)
        st.plotly_chart(fig)
        # st.plotly_chart(fig_mom_growth)

        st.subheader("Page Type Analytics")

        df = fetch_data_from_db_pagetype(engine, selected_year, selected_month)

        fig = plot_events_by_page_type(df)
        st.plotly_chart(fig, use_container_width=True)

        df = fetch_data_from_db(engine, selected_year, selected_month)

        st.subheader("Page Path Analytics")

        page_type = st.selectbox(
            'Page Type',
            df['page_type'].unique()
        )

        top_10_df = (
            df[df['page_type'] == page_type]
            .sort_values(by='event_count', ascending=False)
            .head(10)
        )

        # Create bar chart based on the selected page type and top 10 records
        create_bar_chart(top_10_df, page_type)

    elif page == 'Network Density':
        st.title("Network Density")


        # Member Connection Strength
        st.subheader("Member Connection Strength")
        file_path = "./network-strength/followersfollowing.csv"
        networkStrength(file_path, "member-strength")
       
        # Team Connection Strength
        st.subheader("Team Connection Strength")
        file_path1 = "./network-strength/Connections_TwitInteractions.csv"
        networkStrength(file_path1, "team-strength")

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

        selected_year = st.selectbox("Select Year", years, index=0, key="year_selectbox")
        selected_month = st.selectbox("Select Month", ["All"] + months, index=0,  key="month_selectbox")

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
       
    elif page == 'IRL Gatherings':
        st.title("IRL Gatherings")

        # st.markdown("""
        #     Breakdown of IRL Gathering
        # """)

        st.subheader("Filters")
        years = ["All", "2024"]  # Example, add more years if needed

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

        engine = get_database_connection()

        # Fetch the data based on the selected filters
        df = fetch_events_by_month_location(engine, selected_year, selected_month)
        df['event_month'] = pd.to_datetime(df['event_month'])

        # Plot the graph
        st.subheader("Events by Month")
        st.markdown("Monthly event distribution by geography")
        fig = plot_events_by_month(df)
        st.plotly_chart(fig, use_container_width=True)

        # view_option = "Users"  # Since we're only focusing on Users, this is hardcoded

        # # Fetch the data based on the selected year and month for users
        # df = fetch_events_by_month_by_user(engine, selected_year, selected_month)

        # # Convert 'event_month' to datetime for plotting and ensure no missing values
        # df['event_month'] = pd.to_datetime(df['event_month'])

        # # Ensure the data is clean
        # df = df.dropna(subset=['event_month'])

        # # Plot the graph for users
        # st.subheader("Users by Month")
        # fig = plot_events_by_month_by_user(df)
        # st.plotly_chart(fig, use_container_width=True)

        # df = fetch_events_by_month_by_team(engine, selected_year, selected_month)

        # # Convert 'event_month' to datetime for plotting and ensure no missing values
        # df['event_month'] = pd.to_datetime(df['event_month'])

        # # Ensure the data is clean
        # df = df.dropna(subset=['event_month'])

        # # Plot the graph for users
        # st.subheader("Users by Team")
        # fig = plot_events_by_month_by_user(df)
        # st.plotly_chart(fig, use_container_width=True)

        st.subheader("Monthly Event Engagement analysis")
        st.markdown("User/Team monthly event  attendance")
        selected_data_type = st.radio("Select Type", ("User", "Team"))

        if selected_data_type == "User Count":
            df = fetch_events_by_month_by_user(engine, selected_year, selected_month)
        else:
            df = fetch_events_by_month_by_team(engine, selected_year, selected_month)

        # Ensure 'event_month' is properly formatted
        df['event_month'] = pd.to_datetime(df['event_month'])

        # Create the bar chart using Plotly
        fig = px.bar(
            df, 
            x="event_month", 
            y=df.columns[2],  # Dynamically select user/team count column
            color="location",  # Color by location
            labels={"event_month": "Month", df.columns[2]: "Count"},
            hover_data={"location": True, "event_month": True, df.columns[2]: True}
        )

        # Display the plot
        st.plotly_chart(fig)

        # Filtered data for hosts and speakers
        # df = fetch_events_by_month_with_hosts_and_speakers(engine, selected_year, selected_month)

        # # Ensure that the event_month is in datetime format
        # df['event_month'] = pd.to_datetime(df['event_month'])

        # # Plot the graph for hosts and speakers
        # st.subheader("No. of Hosts / Speakers by Month")
        # fig = plot_events_by_month_with_hosts_and_speakers(df)
        # st.plotly_chart(fig, use_container_width=True)

        # st.title("Attendees with and without Office Hours")
    
        # Database connection
        engine = get_database_connection()
        
        # Fetch data
        data = fetch_attendee_data(engine, selected_year, selected_month)
        
        # Display data as a table
        # st.subheader("Attendee Data")
        # st.dataframe(data)

        # Plot pie chart
        st.subheader("Percentage of Attendees with and without Office Hours")
        fig = px.pie(data, 
                    names="office_hours_status", 
                    values="attendee_count", 
                    # title="Percentage of Attendees with and without Office Hours",
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig)

        visualize_event_topic_distribution(get_database_connection(), selected_year, selected_month)
        st.subheader("Attendees Breakdown by Topic / Skill")
        st.markdown("Distribution of event attendees by Topic / Skill")

        option = st.radio(
            "Select Type:",
            ('Topic', 'Skill')
        )

        # Depending on the selected option, display the corresponding graph
        if option == 'Topic':
            visualize_attendees_by_topic(engine, selected_year, selected_month)
        elif option == 'Skill':
            visualize_attendees_by_skill(engine, selected_year, selected_month)
    
        df = fetch_hosting_teams_by_focus_area(engine, selected_year, selected_month)

        # Plot the graph for hosting teams by focus area
        st.subheader("Focus area segmentation of Hosting Teams")
        st.markdown("Distribution of Hosting Teams by focus areas")
        fig = plot_hosting_teams_by_focus_area(df)
        st.plotly_chart(fig, use_container_width=True)

        df = fetch_hosts_and_speakers_by_month(engine, selected_year, selected_month)

        # Plotting the distribution of hosts and speakers
        st.subheader("Distribution of Hosts and Speakers by Month")
        st.markdown("Distribution of Speakers and Hosting teams/members by month")
        fig = plot_hosts_and_speakers_distribution(df)
        st.plotly_chart(fig, use_container_width=True)

    elif page == 'Hackathons':
        st.title("Hackathons")

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
        st.title("Network Growth")

        # st.subheader("Filters")
        # years = ["All", "2024"] 

        # month_mapping = {
        #     "January": 1,
        #     "February": 2,
        #     "March": 3,
        #     "April": 4,
        #     "May": 5,
        #     "June": 6,
        #     "July": 7,
        #     "August": 8,
        #     "September": 9,
        #     "October": 10,
        #     "November": 11,
        #     "December": 12
        # }

        # months = [
        #     'January', 'February', 'March', 'April', 'May', 'June', 
        #     'July', 'August', 'September', 'October', 'November', 'December'
        # ]

        # selected_year = st.selectbox("Select Year", years, index=0)
        # selected_month = st.selectbox("Select Month", ["All"] + months, index=0)

        st.subheader("MoM analysis on Network Growth")

        df_projects = fetch_data_network_growth(engine, "Project")
        plot_mom_analysis(df_projects, "Projects")

        # 2. Fetch data for Teams
        df_teams = fetch_data_network_growth(engine, "Team")
        plot_mom_analysis(df_teams, "Teams")

        df_members = fetch_data_network_growth(engine, "Member")
        plot_mom_analysis(df_members    , "Member")

        st.subheader("Distribution graph of people by skills")
        dummy_image_url = "https://plabs-assets.s3.us-west-1.amazonaws.com/Coming+Soon.png"
        st.image(dummy_image_url, caption="Distribution graph of people by skills", width=900)

    elif page == 'Search Usage':
        st.title("Search Usage")

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
        event_name_mapping = {
            'Project':'projects-search',
            'IRL':'irl-guest-list-search',
            'Member':'member-list-search-clicked',
            'Team':'team-search'
        }

        # Streamlit selectbox for selecting event type
        event_type = st.selectbox(
            'Select Page Type',
            ['Project', 'IRL', 'Member', 'Team']
        )

        selected_event_name = event_name_mapping[event_type]

        df_event = fetch_search_event_data(engine, selected_event_name, selected_year, selected_month)

        # Plot the user-level event data
        fig = px.bar(
            df_event,
            x='name',  # User's name
            y='event_count',  # Event count
            text='event_count',  # Display event count as text
            labels={'name': 'Members', 'event_count': 'Event Count'},
            title=f"Top 10 Users for Event: {event_type}",
        )

        # Update layout to display counts above each bar
        fig.update_traces(texttemplate='%{text}', textposition='outside', insidetextanchor='middle')

        # Display the Plotly chart for user-level data
        st.plotly_chart(fig)

        with st.expander("Overall Data"):
            df_modified = df_event.copy()

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

        # Fetch monthly event data based on selected event type, year, and month
        df_event_monthly = fetch_search_event_data_by_month(engine, selected_event_name, selected_year, selected_month)

        # Format the month for display
        df_event_monthly['month_start'] = pd.to_datetime(df_event_monthly['month_start']).dt.strftime('%B %Y')

        # Plot the monthly event data
        fig_monthly = px.bar(
            df_event_monthly,
            x='month_start',  # Formatted month and year
            y='event_count',  # Event count
            text='event_count',  # Display event count as text
            labels={'month_start': 'Month', 'event_count': 'Event Count'},
            title=f"Monthly Event Count for {event_type}",
        )

        # Update layout to display counts above each bar
        fig_monthly.update_traces(texttemplate='%{text}', textposition='outside', insidetextanchor='middle')

        # Display the Plotly chart for monthly data
        st.plotly_chart(fig_monthly)

        contact_event_type = st.selectbox(
        'Select Event Type',
        ['Email', 'Twitter', 'Linkedin', 'Github', 'Telegram']
        )

        contact_event_type_mapping = {
            'Email':'email',
            'Twitter':'twitter',
            'Linkedin':'linkedin',
            'Github':'github',
            'Telegram':'telegram'
        }

        contact_event_type = contact_event_type_mapping[contact_event_type]

        query = fetch_data_by_contact_event(contact_event_type, selected_year, selected_month)
        df_result = pd.read_sql(query, engine)

        # Plot the data using Plotly (bar chart for event counts by contact type)
        fig = px.bar(
            df_result,
            x='name',  # User's name
            y='event_count',  # Event count
            # color='type',  # Color by social media type
            text='event_count',  # Display event count as text above bars
            labels={'name': 'Users', 'event_count': 'Event Count', 'type': 'Social Media Type'},
            title=f"Top 10 Users for Event: {contact_event_type.capitalize()} Social Links Clicked",
        )

        # Update layout to display counts above each bar
        fig.update_traces(texttemplate='%{text}', textposition='outside', insidetextanchor='middle')

        # Display the Plotly chart
        st.plotly_chart(fig)


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

# def create_bubble_chart(team_stats):
#     """Creates a bubble chart for team statistics."""
#     # Calculate average counts
#     average_count = team_stats['Count'].mean()
#     max_count = team_stats['Count'].max()
#     min_count = team_stats['Count'].min()

#     # Prepare data for bubble chart
#     stats_df = pd.DataFrame({
#         'Category': ['Above Average', 'Below Average', 'Maximum', 'Minimum'],
#         'Count': [
#             len(team_stats[team_stats['Count'] > average_count]),
#             len(team_stats[team_stats['Count'] < average_count]),
#             len(team_stats[team_stats['Count'] == max_count]),
#             len(team_stats[team_stats['Count'] == min_count])
#         ]
#     })

#     fig = px.scatter(
#         stats_df,
#         x='Category',
#         y='Count',
#         size='Count',
#         hover_name='Category',
#         title='Team Interaction Statistics',
#         labels={'Count': 'Number of Teams', 'Category': 'Interaction Category'},
#         size_max=60  # Adjust the maximum size of the bubbles
#     )

#     st.plotly_chart(fig)

#     return stats_df  # Return the stats DataFrame for further processing

def create_pie_chart(team_stats):
    """Creates a pie chart for team statistics."""
    # Calculate average counts
    average_count = team_stats['Count'].mean()
    max_count = team_stats['Count'].max()
    min_count = team_stats['Count'].min()

    # Prepare data for pie chart
    stats_df = pd.DataFrame({
        'Category': ['Above Average', 'Below Average', 'Maximum', 'Minimum'],
        'Count': [
            len(team_stats[team_stats['Count'] > average_count]),
            len(team_stats[team_stats['Count'] < average_count]),
            len(team_stats[team_stats['Count'] == max_count]),
            len(team_stats[team_stats['Count'] == min_count])
        ]
    })

    # Create a pie chart
    fig = px.pie(
        stats_df,
        names='Category',
        values='Count',
        title='Team Interaction Statistics Distribution',
        color='Category',
        color_discrete_map={'Above Average': 'green', 'Below Average': 'red', 'Maximum': 'blue', 'Minimum': 'orange'}
    )

    # Show the plot
    st.plotly_chart(fig)

    return stats_df 

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

def display_filters(df, key="test"):
    """Displays the network filters and applies the selected filter."""
    avg_connections = df['ConnectionCount'].mean()
    max_connections = df['ConnectionCount'].max()
    min_connections = df['ConnectionCount'].min()

    filter_option = st.selectbox(
        "Filter by connection count",
        options=["None", "Above Average", "Below Average", "Minimum", "Maximum"],
        key=f"{key}-filter_option_selectbox"
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

def networkStrength(file_path, key="test"):
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
        selection_type = st.selectbox("Select:", options=["Member", "Team"], index=0, key=f"{key}-meber-team-selection_type_selectbox")

        if selection_type == "Member":
            # Dropdown for members
            unique_members = df['Member'].unique()
            selected_member = st.selectbox("Select a member:", options=['All'] + list(unique_members), key=f"{key}-member_selectbox")

            if selected_member != 'All':
                filtered_df = df[(df['Member'] == selected_member) | (df['NetworkConnections'] == selected_member)]
                visualize_network(filtered_df)
            else:
                option, filtered_df = display_filters(df, key)
                if not filtered_df.empty:
                    #st.dataframe(filtered_df, height=300)  # Display DataFrame with a scrollable view
                    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"Download Network Connectivity Data as CSV",
                        data=csv_data,
                        file_name=f"{option}_network_connectivity_data.csv",
                        mime='text/csv',
                        key=f"{key}-download_button"
                    )
                    visualize_network(filtered_df)

        elif selection_type == "Team":
            # Show team interactions
            team_stats = calculate_team_statistics(df)
            bubble_stats = create_pie_chart(team_stats)
            selected_category = st.selectbox("Select a category to view teams:", bubble_stats['Category'].unique())
            create_stacked_bar_chart(selected_category, team_stats)


if __name__ == "__main__":
    main()