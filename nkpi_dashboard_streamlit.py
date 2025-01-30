import os
import calendar
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

load_dotenv()

@st.cache_data
def execute_query(query):
    engine = get_database_connection()
    if engine:
        try:
            with engine.connect() as connection:
                return pd.read_sql(query, connection)
        except Exception as e:
            st.error(f"Error executing query: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def get_database_connection():
    DATABASE_URL = os.getenv("DB_URL")
    engine = create_engine(DATABASE_URL)
    return engine


def fetch_team_data():
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
    return execute_query(query)


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


def fetch_focus_area_data():
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
    return execute_query(query)

def fetch_total_teams_count():
    query = """
        SELECT COUNT(*) AS total_teams
        FROM public."Team"
    """
    return execute_query(query)['total_teams'][0]

def fetch_teams_per_month(year=None, month=None):
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
    return execute_query(query)


def visualize_teams_per_month(data):
    if not pd.api.types.is_datetime64_any_dtype(data['month']):
        data['month'] = pd.to_datetime(data['month'], format='%Y-%m')

    data_grouped = data.groupby(data['month'].dt.to_period('M')).agg({'team_count': 'sum'}).reset_index()
    
    data_grouped['month'] = data_grouped['month'].dt.strftime('%b %Y') 
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

def member_interaction_feedback(selected_year, selected_month):
    query = """
    select count(*) as interaction_count, CASE 
            WHEN mfu.status = 'COMPLETED' THEN 
                CASE 
                    WHEN mf.response = 'POSITIVE' THEN 'Responded as Yes'
                    WHEN mf.response = 'NEGATIVE' THEN 'Responded as No'
                    ELSE 'Completed with unknown response'
                END
            WHEN mfu.status = 'CLOSED' THEN 'Pop-up Dismissed'
            WHEN mfu.status = 'PENDING' THEN 'Did not respond'
            ELSE 'Unknown status'
        END AS feedback_response_status FROM 
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
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM mi.\"createdAt\") = {selected_year}"

    if selected_month != "All":
        month_number = month_mapping[selected_month]
        query += f" AND EXTRACT(MONTH FROM mi.\"createdAt\") = {month_number}"

    query += " group by feedback_response_status"
    return execute_query(query)

def plot_top_10_interaction_count(df, title, color, source):
    member_interaction_counts = df.groupby('source_member_name')['interaction_count'].sum().reset_index()

    top_10_member_interactions = member_interaction_counts.sort_values(by='interaction_count', ascending=False).head(10)
    
    st.subheader(f"Top 10 {title}")
    
    fig = px.bar(
        top_10_member_interactions,
        x='source_member_name',
        y='interaction_count',
        labels={"source_member_name": f"{source} Member", "interaction_count": "No. of Interaction"}, 
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
                    p.properties->'user'->>'name',
                    p.properties->>'name'
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
                p.properties->'user'->>'name',
                p.properties->>'name'
            ) NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
        )
    """

    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM COALESCE(NULLIF((p.properties ->> '$sent_at')::text, ''), '1970-01-01')::timestamp) = {selected_year}"

    if selected_month != "All":
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
                    p.properties->'user'->>'name',
                    p.properties->>'name'
                ),
        p."event"
    ORDER BY 
        year, 
        month, 
        interaction_count;
    """

    return execute_query(query)


def plot_bar_chart_of_OH(df, breakdown_type):
    fig = go.Figure()

    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    df['month_order'] = df['month_name'].apply(lambda x: month_order.index(x) + 1)

    if breakdown_type == "Team Breakdown":
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
        filtered_df = df[df['page_type'].isin(['IRL Page', 'Member Page'])]
        aggregated_data_irl = filtered_df[filtered_df['page_type'] == 'IRL Page'].groupby('month_name').agg(
            interaction_count=('interaction_count', 'sum')
        ).reset_index()

        aggregated_data_member = filtered_df[filtered_df['page_type'] == 'Member Page'].groupby('month_name').agg(
            interaction_count=('interaction_count', 'sum')
        ).reset_index()

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
                    p.properties->'user'->>'name',
                    p.properties->>'name'
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
            tm.name NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
        )
        AND (
            sm.name NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
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
                p.properties->'user'->>'name',
                p.properties->>'name'
            ),
        (p.properties ->> '$sent_at'),
        p."event"
    ORDER BY 
        year, 
        month, 
        interaction_count DESC;
    """

    return execute_query(base_query)

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
                p.properties->'user'->>'name',
                p.properties->>'name'
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
            p.properties->'user'->>'name',
            p.properties->>'name'
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
                p.properties->'user'->>'name',
                p.properties->>'name'
            ),
        p."event",
        TO_DATE(SUBSTRING(p.properties ->> '$sent_at' FROM 1 FOR 10), 'YYYY-MM-DD')  -- Added this line to group by date expression
    ORDER BY 
        year, 
        month, 
        interaction_count DESC;
    """

    return execute_query(base_query)


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
                p.properties->'user'->>'name',
                p.properties->>'name'
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
            tm.name NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
        )
        AND (
            sm.name NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
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
                p.properties->'user'->>'name',
                p.properties->>'name'
            ),
        (p.properties ->> '$sent_at'),
        p."event"
    ORDER BY 
        year, 
        month, 
        interaction_count DESC;
    """

    return execute_query(base_query)

def fetch_events_by_month_location(selected_year, selected_month):
    query = """
        SELECT 
            pel."location",  
            DATE_TRUNC('month', pe."startDate") AS event_month, 
            COUNT(*) AS event_count
        FROM public."PLEvent" pe
        INNER JOIN public."PLEventLocation" pel ON pe."locationUid" = pel."uid"
        WHERE pe."startDate" IS NOT NULL
    """
    
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM pe.\"startDate\") = {selected_year}"
    
    if selected_month != "All":
        month_num = month_mapping[selected_month]
        query += f" AND EXTRACT(MONTH FROM pe.\"startDate\") = {month_num}"

    query += """
        GROUP BY event_month, pel."location"
        ORDER BY event_month;
    """
    return execute_query(query)

def fetch_member_by_skills():
    query = """
    SELECT 
            "Skill"."title" AS skill_name,
            COUNT(DISTINCT "Member"."uid") AS member_count
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
        GROUP BY 
            "Skill"."title"
        ORDER BY 
            member_count desc
   """
    return execute_query(query)

def husky_feedback():
    query = """
    select count(*) as interaction_count, name from public.huskyfeedback h WHERE 
    "name" NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay', 'Muhammed B', 'Yosuva R', 'Prasanth Radhakrishnan') 
GROUP BY 
    "name";
   """
    return execute_query(query)

def teams_by_focus_area():
    query = """
    SELECT 
            COALESCE(parentFA.title, fa.title, 'Undefined') AS focus_area, 
            COUNT(DISTINCT t.uid) AS "Teams"
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
        ORDER BY "Teams" DESC;
        """
    return execute_query(query)

def projects_by_focus_area():
    query = """
    SELECT 
            COALESCE(parentFA.title, fa.title, 'Undefined') AS focus_area, 
            COUNT(DISTINCT p.uid) AS "Projects"
        FROM 
            public."Project"  p 
        LEFT JOIN 
            public."ProjectFocusArea" pfa ON p.uid = pfa."projectUid"
        LEFT JOIN 
            public."FocusArea" fa ON fa.uid = pfa."focusAreaUid"
        LEFT JOIN 
            public."FocusAreaHierarchy" fah ON fa.uid = fah."subFocusAreaUid"
        LEFT JOIN 
            public."FocusArea" parentFA ON fah."focusAreaUid" = parentFA.uid
        GROUP BY 
            COALESCE(parentFA.title, fa.title, 'Undefined')
        ORDER BY "Projects" desc
        """
    return execute_query(query)

def husky_prompt_interaction():
    query = """
    select count(*) as interaction_count, COALESCE("name", 'Untracked User') AS "name" from public.huskyconversation h WHERE 
    "name" NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay', 'Muhammed B', 'Yosuva R', 'Prasanth Radhakrishnan') 
    OR "name" IS null group by "name"; 
        """
    return execute_query(query)

def fetch_events_by_month_by_user_1(selected_year, selected_month):
    query = """
        SELECT 
            pe."name" AS event_name,
            COUNT(DISTINCT CASE WHEN eg."isHost" = true THEN eg."memberUid" END) AS host_count,
            COUNT(DISTINCT CASE WHEN eg."isSpeaker" = true THEN eg."memberUid" END) AS speaker_count,
            COUNT(DISTINCT CASE WHEN eg."isHost" = false AND eg."isSpeaker" = false THEN eg."memberUid" END) AS attendee_count,
            COUNT(DISTINCT eg."memberUid") AS total_member_count  -- Total number of distinct members in each event
        FROM 
            public."PLEvent" pe
        LEFT JOIN 
            public."PLEventGuest" eg ON pe."uid" = eg."eventUid"
        LEFT JOIN 
            public."Member" m ON eg."memberUid" = m."uid"
        WHERE 
            pe."startDate" IS NOT NULL
    """
    
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM pe.\"startDate\") = {selected_year}"
    
    if selected_month != "All":
        month_num = month_mapping[selected_month]
        query += f" AND EXTRACT(MONTH FROM pe.\"startDate\") = {month_num}"

    query += """
        GROUP BY 
        pe."name"
    ORDER BY 
        total_member_count DESC  -- Sorting by the total member count in descending order
    LIMIT 20;
    """
    return execute_query(query)    

def fetch_session_data(selected_year, selected_month):
    query = """
    WITH session_durations AS (
        SELECT 
            properties->>'$session_id' AS session_id,
            EXTRACT(EPOCH FROM MAX(timestamp) - MIN(timestamp)) AS session_duration_seconds,
            MIN(timestamp) AS min_timestamp
        FROM 
            public.posthogevents
        WHERE 
            properties->>'$session_id' IS NOT NULL
    """
    
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM MIN(timestamp)) = {selected_year}"
    
    if selected_month != "All":
        query += f" AND EXTRACT(MONTH FROM MIN(timestamp)) = {selected_month}"
    
    query += """
    GROUP BY 
        properties->>'$session_id'
    )
    SELECT 
        EXTRACT(YEAR FROM min_timestamp) AS year,
        EXTRACT(MONTH FROM min_timestamp) AS month,
        FLOOR(AVG(session_duration_seconds) / 60) AS average_duration_minutes,
        MOD(AVG(session_duration_seconds), 60) AS average_duration_seconds
    FROM 
        session_durations
    GROUP BY 
        year, month
    ORDER BY 
        year, month;
    """
    return execute_query(query)

def fetch_events_by_month_by_team(selected_year, selected_month):
    query = """
        SELECT 
            pe."name" AS event_name,
            COUNT(DISTINCT CASE WHEN eg."isHost" = true THEN eg."teamUid" END) AS host_count,
            COUNT(DISTINCT CASE WHEN eg."isSpeaker" = true THEN eg."teamUid" END) AS speaker_count,
            COUNT(DISTINCT CASE WHEN eg."isHost" = false AND eg."isSpeaker" = false THEN eg."teamUid" END) AS attendee_count,
            COUNT(DISTINCT eg."teamUid") AS total_team_count  -- Total number of distinct team in each event
        FROM 
            public."PLEvent" pe
        LEFT JOIN 
            public."PLEventGuest" eg ON pe."uid" = eg."eventUid"
        LEFT JOIN 
            public."Team" t ON eg."teamUid" = t."uid"
        WHERE 
            pe."startDate" IS NOT NULL
    """
    
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM pe.\"startDate\") = {selected_year}"
    
    if selected_month != "All":
        month_num = month_mapping[selected_month]
        query += f" AND EXTRACT(MONTH FROM pe.\"startDate\") = {month_num}"

    query += """
        GROUP BY 
        pe."name"
    ORDER BY 
        total_team_count DESC  
    LIMIT 20;
    """
    
    return execute_query(query)


def plot_events_by_month(dataframe):
    dataframe['event_month'] = pd.to_datetime(dataframe['event_month'])
    dataframe = dataframe.sort_values(by='event_month')
    dataframe['formatted_month'] = dataframe['event_month'].dt.strftime('%b %Y')

    fig = px.bar(
        dataframe, 
        x="formatted_month",  
        y="event_count", 
        color="location",  
        labels={"formatted_month": "Month", "event_count": "Number of Events"},
        hover_data={"location": True, "event_month": True, "event_count": True} 
    )
    
    fig.update_traces(
        hovertemplate="<b>Month:</b> %{x}<br><b>Location:</b> %{customdata[0]}<br><b>Number of Events:</b> %{y}<extra></extra>"
    )
    
    fig.update_layout(
        xaxis=dict(
            type='category', 
            categoryorder='array', 
            categoryarray=dataframe['formatted_month'].unique(), 
        ),
        xaxis_title="Month", 
        yaxis_title="No. of Events", 
        showlegend=True, 
    )

    return fig

def fetch_attendee_data(selected_year, selected_month):
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

    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM pe.\"createdAt\") = {selected_year}"

    if selected_month != "All":
        month_num = month_mapping[selected_month] 
        query += f" AND EXTRACT(MONTH FROM pe.\"createdAt\") = {month_num}"

    query += """
        GROUP BY office_hours_status;
    """

    return execute_query(query)

def fetch_event_topic_distribution(selected_year, selected_month):
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
    
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM \"PLEvent\".\"startDate\") = {selected_year}"

    if selected_month != "All":
        month_num = month_mapping[selected_month]  
        query += f" AND EXTRACT(MONTH FROM \"PLEvent\".\"startDate\") = {month_num}"

    query += """
        GROUP BY 
            topic
        ORDER BY 
            event_count DESC
        LIMIT 10;
    """
    
    return execute_query(query)


def visualize_event_topic_distribution(selected_year, selected_month):
    st.subheader("Top Events by Topic")
    st.markdown("Distribution of events by topic")
    
    df = fetch_event_topic_distribution(selected_year, selected_month)

    fig = px.bar(
        df,
        x="topic",
        y="event_count",
        labels={"topic": "Topic", "event_count": "No. of Events"},
        text_auto=True,
        color="event_count",
    )
    fig.update_layout(xaxis_title="Topic", yaxis_title="No. of Events")
    st.plotly_chart(fig)

def fetch_attendee_data_by_topic(selected_year, selected_month):
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
    
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM \"PLEvent\".\"startDate\") = {selected_year}"

    if selected_month != "All":
        month_num = month_mapping[selected_month]  
        query += f" AND EXTRACT(MONTH FROM \"PLEvent\".\"startDate\") = {month_num}"

    query += """
        GROUP BY 
            topic
        ORDER BY 
            attendee_count DESC
        LIMIT 10;
    """
    
    return execute_query(query)

def visualize_attendees_by_topic(selected_year, selected_month):
    
    df = fetch_attendee_data_by_topic(selected_year, selected_month)
    
    fig = px.bar(
        df, 
        x="topic", 
        y="attendee_count", 
        labels={"topic": "Topic", "attendee_count": "No. of Attendees"}, 
        text_auto=True,
        color="attendee_count"    )
    fig.update_layout(xaxis_title="Topic", yaxis_title="No. of Attendees")
    st.plotly_chart(fig)

def filter_data_by_month_and_year(dataframe, selected_year, selected_month):
    dataframe['event_month'] = pd.to_datetime(dataframe['event_month'], errors='coerce')
    
    if selected_year != "All":
        dataframe = dataframe[dataframe['event_month'].dt.year == int(selected_year)]
    
    if selected_month != "All":
        month_number = pd.to_datetime(selected_month, format='%B').month
        dataframe = dataframe[dataframe['event_month'].dt.month == month_number]
    
    return dataframe

def visualize_attendees_by_skill(selected_year, selected_month):
    month_mapping = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
        "All": None
    }

    selected_month_num = month_mapping.get(selected_month, None)

    year_condition = ""
    month_condition = ""
    
    if selected_year != "All":
        year_condition = f"EXTRACT(YEAR FROM \"PLEvent\".\"startDate\") = {selected_year}"
    
    if selected_month_num is not None:
        month_condition = f"EXTRACT(MONTH FROM \"PLEvent\".\"startDate\") = {selected_month_num}"
    
    where_conditions = []
    if year_condition:
        where_conditions.append(year_condition)
    if month_condition:
        where_conditions.append(month_condition)
    
    where_clause = " AND ".join(where_conditions)
    if where_clause:
        where_clause = "WHERE " + where_clause
    
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

    df = execute_query(query)

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


def fetch_hosting_teams_by_focus_area(selected_year, selected_month):
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
    
    return execute_query(query)

def plot_hosting_teams_by_focus_area(dataframe):
    dataframe = dataframe.dropna(subset=['focus_area', 'Teams'])

    if 'Teams' not in dataframe.columns:
        raise ValueError("Column 'Teams' does not exist in the DataFrame.")
    
    dataframe['Teams'] = pd.to_numeric(dataframe['Teams'], errors='coerce')

    dataframe = dataframe.dropna(subset=['Teams'])

    fig = px.bar(
        dataframe, 
        x="focus_area", 
        y="Teams", 
        labels={"focus_area": "Focus Area", "Teams": "No. of Hosting Teams"},
        hover_data={"focus_area": True, "Teams": True}  
    )
    
    fig.update_traces(
        hovertemplate="<b>Focus Area:</b> %{x}<br><b>Number of Hosting Teams:</b> %{y}<extra></extra>"
    )

    fig.update_layout(
        xaxis_title="Focus Area", 
        yaxis_title="No. of Hosting Teams", 
        showlegend=False  
    )

    return fig

def fetch_hosts_and_speakers_by_month(selected_year, selected_month):
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

    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM pe.\"startDate\") = {selected_year}"

    if selected_month != "All":
        month_num = month_mapping[selected_month]  
        query += f" AND EXTRACT(MONTH FROM pe.\"startDate\") = {month_num}"

    query += """
        GROUP BY 
            event_month
        ORDER BY 
            event_month;
    """
    
    return execute_query(query)


def plot_hosts_and_speakers_distribution(dataframe):
    dataframe['event_month'] = pd.to_datetime(dataframe['event_month']).dt.strftime('%B %Y')

    long_df = pd.melt(dataframe, id_vars=['event_month'], 
                      value_vars=['host_members', 'host_teams', 'speaker_members', 'speaker_teams'],
                      var_name='role_type', value_name='count')

    role_type_map = {
        'host_members': 'Hosts(Members)',
        'host_teams': 'Hosts(Teams)',
        'speaker_members': 'Speakers(Members)',
        'speaker_teams': 'Speakers(Teams)'
    }

    long_df['role_type'] = long_df['role_type'].map(role_type_map)

    fig = px.bar(long_df, 
                 x="event_month", 
                 y="count", 
                 color="role_type", 
                 labels={"event_month": "Month", "count": "No. of Hosts/Speakers", "role_type": "Role Type"},
                 category_orders={"role_type": ["Hosts(Members)", "Hosts(Teams)", "Speakers(Members)", "Speakers(Teams)"]},
                 hover_data={"event_month": True, "role_type": True, "count": True})
    
    fig.update_layout(
        xaxis_title="Month", 
        yaxis_title="No. of Hosts/Speakers", 
        showlegend=True
    )

    return fig

def fetch_active_users():
    query = """
        WITH guest_sessions AS (
            SELECT
                EXTRACT(YEAR FROM timestamp) AS year,
                EXTRACT(MONTH FROM timestamp) AS month,
                COUNT(properties->>'$session_id') AS session_count,
                COALESCE(
                    properties->>'userName',
                    properties->>'loggedInUserName',
                    properties->'user'->>'name',
                    'guest_user'  -- Treat NULL users as 'guest_user'
                ) AS user_name,
                properties->>'$session_id' AS session_id  -- Track unique session
            FROM 
                public.posthogevents
            WHERE
                properties->>'$session_id' IS NOT NULL
                AND (
                    COALESCE(
                        properties->>'userName', 
                        properties->>'loggedInUserName', 
                        properties->'user'->>'name'
                    ) IS NULL  -- Only for guest users
                    OR properties->>'userName' NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
                )
            GROUP BY
                EXTRACT(YEAR FROM timestamp), 
                EXTRACT(MONTH FROM timestamp),
                properties->>'$session_id', 
                COALESCE(
                    properties->>'userName',
                    properties->>'loggedInUserName',
                    properties->'user'->>'name',
                    'guest_user'  -- Treat NULL users as 'guest_user'
                )
            HAVING 
                COUNT(properties->>'$session_id') > 5  -- Only include guest sessions with more than 5 occurrences
        ),
        active_users AS (
            -- Find active users (non-guest users)
            SELECT
                EXTRACT(YEAR FROM timestamp) AS year,
                EXTRACT(MONTH FROM timestamp) AS month,
                COUNT(DISTINCT
                    COALESCE(
                        properties->>'userName', 
                        properties->>'loggedInUserName', 
                        properties->'user'->>'name',
                        'guest_user'
                    )
                ) AS active_user_count
            FROM 
                public.posthogevents
            WHERE
                properties->>'$session_id' IS NOT NULL
                AND (
                    COALESCE(
                        properties->>'userName', 
                        properties->>'loggedInUserName', 
                        properties->'user'->>'name'
                    ) NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
                    OR properties->>'userName' IS NULL
                )
            GROUP BY
                EXTRACT(YEAR FROM timestamp), 
                EXTRACT(MONTH FROM timestamp)
        )

        SELECT
            gs.year,
            gs.month,
            COUNT(DISTINCT gs.session_id) AS guest_user_count,  -- Count unique guest user sessions
            au.active_user_count
        FROM
            guest_sessions gs
        JOIN
            active_users au
        ON
            gs.year = au.year
            AND gs.month = au.month
        GROUP BY
            gs.year,
            gs.month,
            au.active_user_count
        ORDER BY 
            gs.year, gs.month;

    """
    return execute_query(query)


def fetch_average_session_time():
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
    return execute_query(query)

def fetch_data_from_db_pagetype(selected_year, selected_month):
    query = """
    WITH guest_sessions AS (
        SELECT
            EXTRACT(YEAR FROM timestamp) AS year,
            EXTRACT(MONTH FROM timestamp) AS month,
            COUNT(properties->>'$session_id') AS session_count,
            COALESCE(
                properties->>'userName',
                properties->>'loggedInUserName',
                properties->'user'->>'name',
                'guest_user'  -- Treat NULL users as 'guest_user'
            ) AS user_name,
            properties->>'$session_id' AS session_id  -- Track unique session
        FROM 
            public.posthogevents
        WHERE
            properties->>'$session_id' IS NOT NULL
            AND (
                COALESCE(
                    properties->>'userName', 
                    properties->>'loggedInUserName', 
                    properties->'user'->>'name'
                ) IS NULL  -- Only for guest users
                OR properties->>'userName' NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
            )
        GROUP BY
            EXTRACT(YEAR FROM timestamp), 
            EXTRACT(MONTH FROM timestamp),
            properties->>'$session_id', 
            COALESCE(
                properties->>'userName',
                properties->>'loggedInUserName',
                properties->'user'->>'name',
                'guest_user'  -- Treat NULL users as 'guest_user'
            )
        HAVING 
            COUNT(properties->>'$session_id') > 5  -- Only include guest sessions with more than 5 occurrences
    ),
    active_users AS (
        -- Find active users (non-guest users)
        SELECT
            EXTRACT(YEAR FROM timestamp) AS year,
            EXTRACT(MONTH FROM timestamp) AS month,
            COUNT(DISTINCT COALESCE(
                properties->>'userName',    
                properties->>'loggedInUserName', 
                properties->'user'->>'name',
                'guest_user'
            )) AS active_user_count
        FROM 
            public.posthogevents
        WHERE
            properties->>'$session_id' IS NOT NULL
            AND (
                COALESCE(
                    properties->>'userName', 
                    properties->>'loggedInUserName', 
                    properties->'user'->>'name'
                ) NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
                OR properties->>'userName' IS NULL
            )
        GROUP BY
            EXTRACT(YEAR FROM timestamp), 
            EXTRACT(MONTH FROM timestamp)
    )

    SELECT 
        gs.year,
        gs.month,
        -- Determine the page type based on the URL
        CASE
            WHEN properties->'$set'->>'$current_url' = 'https://directory.plnetwork.io/members' THEN 'Member Landing Page'
            WHEN properties->'$set'->>'$current_url' = 'https://directory.plnetwork.io/projects' THEN 'Project Landing Page'
            WHEN properties->'$set'->>'$current_url' = 'https://directory.plnetwork.io/irl' THEN 'IRL Landing Page'
            WHEN properties->'$set'->>'$current_url' = 'https://directory.plnetwork.io/teams' THEN 'Team Landing Page'
            WHEN properties->'$set'->>'$current_url' LIKE 'https://directory.plnetwork.io/members%%' THEN 'Member Details Page'
            WHEN properties->'$set'->>'$current_url' LIKE 'https://directory.plnetwork.io/projects%%' THEN 'Project Details Page'
            WHEN properties->'$set'->>'$current_url' LIKE 'https://directory.plnetwork.io/teams%%' THEN 'Team Details Page'
            WHEN properties->'$set'->>'$current_url' LIKE 'https://directory.plnetwork.io/irl%%' THEN 'IRL Details Page'
            ELSE 'Other'
        END AS page_type,
        -- Count the guest users per page type for each month
        COUNT(DISTINCT gs.session_id) AS guest_user_count,  -- Track guest sessions
        -- Active user count
        au.active_user_count
    FROM 
        public.posthogevents AS properties
    JOIN 
        guest_sessions AS gs
        ON properties->>'$session_id' = gs.session_id
        AND EXTRACT(YEAR FROM properties.timestamp) = gs.year
        AND EXTRACT(MONTH FROM properties.timestamp) = gs.month
    LEFT JOIN 
        active_users AS au
        ON EXTRACT(YEAR FROM properties.timestamp) = au.year
        AND EXTRACT(MONTH FROM properties.timestamp) = au.month
    """

    month_mapping = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }

    selected_month_num = month_mapping.get(selected_month, None)

    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM properties.timestamp) = {selected_year}"
    
    if selected_month != "All" and selected_month_num is not None:
        query += f" AND EXTRACT(MONTH FROM properties.timestamp) = {selected_month_num}"

    query += """
    GROUP BY 
        gs.year, gs.month, page_type, au.active_user_count
    ORDER BY 
        gs.year, gs.month, page_type;
    """

    return execute_query(query)


def plot_events_by_page_type(df):
    fig = px.bar(
        df, 
        x="month", 
        y="active_user_count", 
        color="page_type",      
        labels={"month": "Month", "active_user_count": "Active Users", "page_type": "Page Type"},
        barmode="stack",       
        hover_data={"page_type": True, "active_user_count": True, "month": True}
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Active User",
        showlegend=True,
    )

    fig.update_xaxes(
        tickmode='array',
        tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )

    return fig

def plot_events_by_page_type_1(df):
    fig = px.bar(
        df, 
        x="month", 
        y="active_user_count",  
        color="page_type",     
        labels={"month": "Month", "active_user_count": "Active Users", "page_type": "Page Type"},
        barmode="stack",       
        hover_data={"page_type": True, "active_user_count": True, "month": True}
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Active User",
        showlegend=True,
    )

    fig.update_xaxes(
        tickmode='array',
        tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )

    return fig

def fetch_data_from_db(selected_year, selected_month):
    query = """
    SELECT
        CASE
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/members/%%' THEN 'Member Detail'
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/members?searchBy%%' THEN 'Member Search'
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/projects/%%' THEN 'Project Detail'
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/projects?searchBy%%' THEN 'Project Search'
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/teams/%%' THEN 'Team Detail'
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/teams?searchBy%%' THEN 'Team Search'
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
        -- For Project Search, extract the value after 'searchBy'
        CASE
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/projects?searchBy=%%' THEN
                SUBSTRING(properties->'$set'->>'$current_url' FROM 'searchBy=([^&]+)')
            ELSE NULL
        END AS project_search_value,
        -- For Member Search, extract the value after 'searchBy'
        CASE
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/members?searchBy=%%' THEN
                SUBSTRING(properties->'$set'->>'$current_url' FROM 'searchBy=([^&]+)')
            ELSE NULL
        END AS member_search_value,
        -- For Team Search, extract the value after 'searchBy'
        CASE
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/teams?searchBy=%%' THEN
                SUBSTRING(properties->'$set'->>'$current_url' FROM 'searchBy=([^&]+)')
            ELSE NULL
        END AS team_search_value,
        -- For IRL Search, extract the value after 'search'
        CASE
            WHEN properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/irl?search=%%' THEN
                SUBSTRING(properties->'$set'->>'$current_url' FROM 'search=([^&]+)')
            ELSE NULL
        END AS irl_search_value
    FROM public.posthogevents
    -- Join the members table based on the extracted member UID from the URL
    LEFT JOIN public."Member" m ON properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/members/%%' AND m.uid = SPLIT_PART(properties->'$set'->>'$current_url', '/', 5)
    -- Join the projects table based on the extracted project UID from the URL
    LEFT JOIN public."Project" p ON properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/projects/%%' AND p.uid = SPLIT_PART(properties->'$set'->>'$current_url', '/', 5)
    -- Join the teams table based on the extracted team UID from the URL
    LEFT JOIN public."Team" t ON properties->'$set'->>'$current_url' LIKE '%%https://directory.plnetwork.io/teams/%%' AND t.uid = SPLIT_PART(properties->'$set'->>'$current_url', '/', 5)
    WHERE properties->'$set'->>'$current_url' IS NOT NULL
    """

    selected_month_num = month_mapping.get(selected_month, None)

    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM timestamp) = {selected_year}"
    
    if selected_month != "All":
        query += f" AND EXTRACT(MONTH FROM timestamp) = {selected_month_num}"

    query += """
    GROUP BY page_type, entity_name, project_search_value, member_search_value, team_search_value, irl_search_value
    ORDER BY page_type, event_count DESC;
    """
    
    return execute_query(query)

def create_bar_chart(data, selected_page_type):
    filtered_data = data[data['page_type'] == selected_page_type]
    
    if selected_page_type == 'Member Search':
        filtered_data['display_label'] = filtered_data['member_search_value'].fillna('No Search')
    elif selected_page_type == 'Project Search':
        filtered_data['display_label'] = filtered_data['project_search_value'].fillna('No Search')
    elif selected_page_type == 'Team Search':
        filtered_data['display_label'] = filtered_data['team_search_value'].fillna('No Search')
    elif selected_page_type == 'IRL Search':
        filtered_data['display_label'] = filtered_data['irl_search_value'].fillna('No Search')
    else:
        filtered_data['display_label'] = filtered_data['entity_name']

    fig = px.bar(
        filtered_data,
        x='display_label', 
        y='event_count',
        labels={'display_label': 'Entity Name / Search Term', 'event_count': 'Count', 'page_type': 'Page Type'},
        color='display_label',  
        category_orders={'display_label': filtered_data['display_label'].unique().tolist()},  
        text='event_count', 
    )
    
    st.plotly_chart(fig)

def fetch_data_network_growth_1(table_name):
    query = f"""
    SELECT
        DATE_TRUNC('month', "createdAt") AS month,
        COUNT(DISTINCT "uid") AS new_entries
    FROM 
        public."{table_name}"
    WHERE 
        "createdAt" IS NOT NULL
    GROUP BY 
        DATE_TRUNC('month', "createdAt")
    ORDER BY 
        month;
    """
    return execute_query(query)

def fetch_and_plot_all_entries(selected_year, selected_month):
    df_projects = fetch_data_network_growth_1("Project")
    df_projects['category'] = 'Project'  
    df_teams = fetch_data_network_growth_1("Team")
    df_teams['category'] = 'Team' 
    df_members = fetch_data_network_growth_1("Member")
    df_members['category'] = 'Member' 
    df_combined = pd.concat([df_projects, df_teams, df_members], ignore_index=True)
    df_combined['month_year'] = pd.to_datetime(df_combined['month']).dt.strftime('%Y-%m')

    if selected_year != "All":
        df_combined = df_combined[df_combined['month_year'].str.startswith(selected_year)]
    
    if selected_month != "All":
        selected_month_num = month_mapping[selected_month]
        df_combined = df_combined[pd.to_datetime(df_combined['month']).dt.month == selected_month_num]

    all_months = pd.date_range(start=df_combined['month'].min(), end=df_combined['month'].max(), freq='MS')
    df_complete = pd.DataFrame({'month': all_months})
    df_combined = pd.merge(df_complete, df_combined, on='month', how='left')
    df_combined['new_entries'] = df_combined['new_entries'].fillna(0)
    plot_combined_bar_graph(df_combined)

def plot_combined_bar_graph(df):
    df['month_year'] = pd.to_datetime(df['month']).dt.strftime('%b %Y')

    fig = px.bar(
        df,
        x='month_year',
        y='new_entries',
        color='category', 
        labels={'month_year': 'Month-Year', 'new_entries': 'New Entries', 'category': 'Category'},
        text='new_entries', 
        title="New Entries Trend by Month (Project, Team, Member)"
    )

    fig.update_layout(
        xaxis_title="Month-Year",
        yaxis_title="New Entries",
        xaxis=dict(
            tickmode='array',
            tickvals=df['month_year'],  
            ticktext=df['month_year'].unique(), 
            tickangle=45,  
        ),
        showlegend=True,  
        margin=dict(t=50, b=50, l=50, r=50)
    )

    fig.update_traces(textposition='outside', texttemplate='%{text}')
    st.plotly_chart(fig)


def fetch_search_event_data(event_name, selected_year, selected_month, user_status):
    query = f"""
    SELECT 
        COALESCE(
            properties->>'loggedInUserEmail', 
            properties->'user'->>'email'
        ) AS email,
        COALESCE(
            NULLIF(properties->>'loggedInUserName', ''),  
            NULLIF(properties->'user'->>'name', ''), 
            NULLIF(properties->>'userName', ''), 
            'Guest User'  
        ) AS name,
        COUNT(*) AS event_count,
        COALESCE(
            CASE
                WHEN properties->>'loggedInUserEmail' IS NOT NULL AND properties->>'loggedInUserEmail' != '' THEN 'Logged-In'
                ELSE 'Logged-Out'
            END, 'Logged-Out'
        ) AS user_status,
        -- Ensure search_value is selected for the specific event
        CASE
            WHEN '{event_name}' = 'projects-search' THEN COALESCE(properties->>'search', properties->>'name')
            WHEN '{event_name}' = 'team-search' THEN properties->>'value'
            WHEN '{event_name}' = 'member-list-search-clicked' THEN properties->>'searchValue'
            WHEN '{event_name}' = 'irl-guest-list-search' THEN COALESCE(properties->>'searchQuery', properties->>'searchTerm')
            ELSE NULL
        END AS search_value
    """

    query += f"""
    FROM posthogevents
    WHERE "event" = '{event_name}' 
    AND (
        (
            COALESCE(
                properties->>'userName', 
                properties->>'loggedInUserName', 
                properties->'user'->>'name'
            ) NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
        )
        OR 
        (properties->>'loggedInUserName' IS NULL AND properties->'user'->>'name' IS NULL AND  properties->>'userName' IS NULL)
    )
    """

    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM timestamp) = {selected_year}"

    if selected_month != "All":
        month_number = month_mapping[selected_month]
        query += f" AND EXTRACT(MONTH FROM timestamp) = {month_number}"

    if user_status != "All":
        query += f" AND CASE WHEN properties->>'loggedInUserEmail' IS NOT NULL AND properties->>'loggedInUserEmail' != '' THEN 'Logged-In' ELSE 'Logged-Out' END = '{user_status}'"

    query += """
    GROUP BY email, name, search_value, user_status
    ORDER BY event_count DESC;
    """

    return execute_query(query)


def fetch_search_event_data_by_month(event_type, selected_year, selected_month, user_status_filter=None):
    query = f"""
    SELECT 
        DATE_TRUNC('month', timestamp) AS month_start,
        COUNT(*) AS event_count,
        CASE
            WHEN COALESCE(
                    properties->>'userName',
                    properties->>'loggedInUserName', 
                    properties->'user'->>'name'
                ) IS NOT NULL 
                AND COALESCE(
                    properties->>'userName',
                    properties->>'loggedInUserName', 
                    properties->'user'->>'name'
                ) != '' 
            THEN 'Logged-In'
            ELSE 'Logged-Out'
        END AS user_status
    FROM posthogevents
    WHERE "event" = '{event_type}' 
    AND (
        -- Exclude the specific users by name (whether logged-in or logged-out)
        COALESCE(
            properties->>'userName',
            properties->>'loggedInUserName', 
            properties->'user'->>'name'
        ) NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
        -- Include logged-out users (where all user-identifying fields are NULL)
        OR (
            properties->>'userName' IS NULL AND 
            properties->>'loggedInUserName' IS NULL AND 
            properties->'user'->>'name' IS NULL
        )
    )
    """

    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM timestamp) = {selected_year}"

    if selected_month != "All":
        month_number = month_mapping[selected_month]
        query += f" AND EXTRACT(MONTH FROM timestamp) = {month_number}"

    if user_status_filter and user_status_filter != "All":
        query += f"""
        AND (
            CASE
                WHEN COALESCE(
                        properties->>'userName',
                        properties->>'loggedInUserName', 
                        properties->'user'->>'name'
                    ) IS NOT NULL 
                    AND COALESCE(
                        properties->>'userName',
                        properties->>'loggedInUserName', 
                        properties->'user'->>'name'
                    ) != '' 
                THEN 'Logged-In'
                ELSE 'Logged-Out'
            END = '{user_status_filter}'
        )
        """

    query += """
    GROUP BY month_start, user_status
    ORDER BY month_start;
    """

    return execute_query(query)


def fetch_data_by_contact_event(selected_type, selected_year, selected_month):
    query = f"""
    SELECT 
        COALESCE(
            NULLIF(TRIM(properties->>'loggedInUserName'), ''),  -- Treat empty strings as NULL and trim whitespace
            NULLIF(TRIM(properties->'user'->>'name'), ''), 
            'Guest User'  -- Default value if name is missing or empty
        ) AS name,
        COALESCE(properties->>'type', 'Unknown') AS type,
        COALESCE(properties->>'name', 'Unknown') AS clicked_name,
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
    
    if selected_type != "All":
        query += f" AND COALESCE(properties->>'type', 'Unknown') = '{selected_type}'"

    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM timestamp) = {selected_year}"

    if selected_month != "All":
        month_number = month_mapping[selected_month]
        query += f" AND EXTRACT(MONTH FROM timestamp) = {month_number}"

    query += """
    GROUP BY name, type, clicked_name
    ORDER BY event_count DESC;    """
    return query


def fetch_team_data_clicked(selected_year, selected_month):
    query = """
        SELECT 
            COALESCE(properties->>'loggedInUserName', properties->'user'->>'name') AS ClickedBy,
            COALESCE (properties->'teamName', properties->'name') as team_name,
            TO_CHAR(timestamp, 'Month YYYY') AS month,
            EXTRACT(YEAR FROM timestamp) AS year,
            CASE
                WHEN "event" = 'irl-guest-list-table-team-clicked' THEN 'IRL Page'
                WHEN "event" = 'team-clicked' THEN 'Teams Landing Page'
                WHEN "event" = 'memeber-detail-team-clicked' THEN 'Member Profile Page'
                WHEN "event" = 'project-detail-maintainer-team-clicked' THEN 'Project Page'
                ELSE 'Other'
            END AS page_type,
            COUNT(*) AS event_count
        FROM posthogevents
        WHERE 
            (properties->>'loggedInUserEmail' IS NOT NULL OR properties->'user'->>'email' IS NOT NULL)
            AND "event" IN ('irl-guest-list-table-team-clicked', 'team-clicked', 'memeber-detail-team-clicked', 'project-detail-maintainer-team-clicked')
             AND (
        -- Exclude the specific users by name (whether logged-in or logged-out)
        (
            COALESCE(
                properties->>'loggedInUserName', 
                properties->'user'->>'name'
            ) NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
        )
    )
    """
    
    if selected_year != "All":
        query += f" AND EXTRACT(YEAR FROM timestamp) = {selected_year}"
    
    if selected_month != "All":
        query += f" AND TO_CHAR(timestamp, 'Month YYYY') = '{selected_month}'"
    
    query += """
        GROUP BY 
            ClickedBy, month, year, page_type, team_name
        ORDER BY event_count DESC;
    """
    
    return execute_query(query)    

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
            "Usage Activity"
        ]
    )

    engine = get_database_connection()

    if page == "Activity - Directory Teams":
        st.title("Activity -- Directory Teams")

        st.markdown("""Breakdown of user engagement and activity on Directory team profiles""")

        df = fetch_team_data()
        df['year'] = df['year'].astype(int)
        

        total_teams = fetch_total_teams_count()

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
            year=int(selected_year) if selected_year != "All" else None,
            month=month_mapping.get(selected_month, None) if selected_month != "All" else None,
        )

        visualize_teams_per_month(teams_per_month)

        st.subheader("Team Segmentation by Focus Areas")
        focus_area_data = fetch_focus_area_data()
        pie_fig = plot_pie_chart_for_team_focus(focus_area_data)
        st.plotly_chart(pie_fig, use_container_width=True)

    elif page == "Office Hours Usage":
        st.title("Office Hours Usage")
        
        st.markdown("""
            Breakdown of OH activity on Member and Team Profile
        """)

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


        df = fetch_oh_data(selected_year, selected_month, month_mapping)
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

        df = member_interaction_feedback(selected_year, selected_month)
        fig = px.pie(
            df, 
            names="feedback_response_status",  
            values="interaction_count", 
            color_discrete_sequence=px.colors.qualitative.Pastel  
        )

        fig.update_traces(
            hovertemplate="<b>Feedback:</b> %{label}<br><b>Interaction Count:</b> %{value}<br><b>Percent:</b> %{percent}<extra></extra>"
        )
        fig.update_layout(
            legend_title="Feedback",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, use_container_width=True)
        selected_month_num = month_mapping.get(selected_month, None) if selected_month != "All" else None

        source_to_target_df = build_source_to_target_query(selected_year, selected_month_num)
        target_to_source_df = build_target_to_source_query(selected_year, selected_month_num)
        source_to_target_team_df = build_source_to_team_query(selected_year, selected_month_num)

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

        df = load_data("./OH/OH-Data-Members.csv")
        try:
            df['month'] = pd.to_datetime(df['month'], format='%b %Y')
        except ValueError:
            st.error("Ensure the `month` column is in the format 'Nov 2022'.")
            st.stop()

        df = load_data("./OH/OH-Data-Members.csv")

        df['month'] = pd.to_datetime(df['month'], format='%b %Y')
        df = df.sort_values(by='month')


        st.subheader("Bar Chart: Counts for Users and Teams with Office Hours")

        fig_count = go.Figure()

        fig_count.add_trace(go.Bar(
            x=df['month'].dt.strftime('%b %Y'),
            y=df['no_users_w_oh'],
            name='Users with Office Hours (Count)',
            marker_color='skyblue',
            hoverinfo='text',
            text=[f"{val} Users" for val in df['no_users_w_oh']]
        ))

        fig_count.add_trace(go.Bar(
            x=df['month'].dt.strftime('%b %Y'),
            y=df['no_teams_w_oh'],
            name='Teams with Office Hours (Count)',
            marker_color='red',
            hoverinfo='text',
            text=[f"{val} Teams" for val in df['no_teams_w_oh']]
        ))

        fig_count.update_layout(
            title="Users and Teams with Office Hours (Count)",
            xaxis_title="Month",
            yaxis_title="Count",
            barmode='stack', 
            xaxis_tickangle=-45,
            showlegend=True
        )

        st.plotly_chart(fig_count)

        st.subheader("Bar Chart: Percentages for Users and Teams with Office Hours")

        fig_percent = go.Figure()

        fig_percent.add_trace(go.Bar(
            x=df['month'].dt.strftime('%b %Y'),
            y=df['precent_users_w_oh'],
            name='Users with Office Hours (%)',
            marker_color='skyblue',
            hoverinfo='text',
            text=[f"{val}% Users" for val in df['precent_users_w_oh']]
        ))

        fig_percent.add_trace(go.Bar(
            x=df['month'].dt.strftime('%b %Y'),
            y=df['percent_teams_w_oh'],
            name='Teams with Office Hours (%)',
            marker_color='red',
            hoverinfo='text',
            text=[f"{val}% Teams" for val in df['percent_teams_w_oh']]
        ))

        fig_percent.update_layout(
            title="Users and Teams with Office Hours (Percentage)",
            xaxis_title="Month",
            yaxis_title="Percentage",
            barmode='stack',  
            xaxis_tickangle=-45,
            showlegend=True
        )

        st.plotly_chart(fig_percent)


        total_scheduled = 62
        confirmed_meetings = 19
        gave_feedback = 43
        did_not_give_feedback = 62 - gave_feedback  
        nps_from_feedback = "80%"

        fig_confirmed_meetings = go.Figure(data=[go.Pie(
            labels=["Confirmed Meetings", "Unconfirmed Meetings","Gave Feedback", "Did Not Give Feedback"],
            values=[confirmed_meetings, total_scheduled - confirmed_meetings,gave_feedback, did_not_give_feedback],
            marker=dict(colors=["#1f77b4", "#ff7f0e","#2ca02c", "#d62728"]),
            hoverinfo="label+percent",
            textinfo="value+percent"
        )])

    
        st.title("Meeting Feedback Analysis")

        st.plotly_chart(fig_confirmed_meetings)


        st.title("Aggregated OH Data")
        df = load_data("./OH/OH Data-AggregatedOHs.csv")

        if "Date" in df.columns and "user-oh" in df.columns and "irl-user-oh" in df.columns and "team-oh" in df.columns and "combined-oh" in df.columns:
            
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
            st.subheader("Bar Chart of Office Hours")

            fig = go.Figure()

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

            fig.update_layout(
                title="Office Hours for Different Categories Over Time",
                xaxis_title="Date",
                yaxis_title="Office Hours",
                barmode='stack',  
                xaxis_tickangle=-45, 
                showlegend=True
            )

            st.plotly_chart(fig)

        else:
            st.error("The CSV file is missing some required columns. Please ensure the file contains 'Date', 'user-oh', 'irl-user-oh', 'team-oh', and 'combined-oh' columns.")
        
        st.title("Responses PMF-v1")
        df = load_data("./OH/OH Data-PMFv1.csv")
        if "Response Category" in df.columns and "Count" in df.columns and "Percentage" in df.columns:
            
            fig = go.Figure(data=[go.Pie(
                labels=df['Response Category'],
                values=df['Count'],
                hoverinfo='label+percent',  
                textinfo='percent',  
                marker=dict(colors=["#1f77b4", "#ff7f0e", "#2ca02c"])  
            )])

            fig.update_layout(
                title="Distribution of Response Categories",
                showlegend=True
            )

            st.plotly_chart(fig)
            
        else:
            st.error("The CSV file is missing some required columns. Please ensure the file contains 'Response Category', 'Count', and 'Percentage' columns.")

    elif page == "Directory MAUs":
        st.title("Directory MAUs")

        df = fetch_average_session_time()
        avg_minutes = int(df['average_duration_minutes'][0])
        avg_seconds = int(df['average_duration_seconds'][0])

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

        st.subheader("Session Duration Analysis by Month")

        df = fetch_session_data(selected_year, selected_month)

        df['year_month'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

        df['total_duration_seconds'] = df['average_duration_minutes'] * 60 + df['average_duration_seconds']

        fig = px.line(
            df, 
            x="year_month", 
            y="total_duration_seconds", 
            labels={"year_month": "Month-Year", "total_duration_seconds": "Average Duration (Seconds)"},
            markers=True
        )

        st.plotly_chart(fig)

        st.subheader("Active Users Analysis")

        tab = st.radio("Select Visualization", ["Monthly Active Users", "Page Type Analytics"])

        active_df = fetch_active_users()
        
        if selected_year != "All":
            active_df = active_df[active_df['year'] == int(selected_year)]

        if selected_month != "All":
            month_number = month_mapping[selected_month]
            active_df = active_df[active_df['month'] == month_number]

        if tab == "Monthly Active Users":
            active_df['month_year'] = pd.to_datetime(active_df[['year', 'month']].assign(day=1))
            active_df['formatted_month'] = active_df['month_year'].dt.strftime('%b %Y')

            df_melted = active_df.melt(
                id_vars=["formatted_month"], 
                value_vars=["guest_user_count", "active_user_count"],
                var_name="user_type", 
                value_name="user_count"
            )

            user_type_map = {
                "guest_user_count": "Guest User",
                "active_user_count": "Active User"
            }
            df_melted['user_type'] = df_melted['user_type'].map(user_type_map)

            df_melted['user_count'] = df_melted['user_count'].fillna(0)
            df_melted['user_count'] = pd.to_numeric(df_melted['user_count'], errors='coerce')

            fig_active_users = px.bar(
                df_melted, 
                x="formatted_month", 
                y="user_count", 
                color="user_type", 
                labels={"formatted_month": "Month-Year", "user_count": "User Count", "user_type": "User Type"},
                hover_data={"formatted_month": True, "user_type": True, "user_count": True},
                title="Monthly Active Users vs Guest Users"            )

            st.plotly_chart(fig_active_users)

        elif tab == "Page Type Analytics":
            st.subheader("Page Type Analytics (Active Users)")
            st.markdown("Analysis of active users breakdown on page type by monthly basis")

            df = fetch_data_from_db_pagetype(selected_year, selected_month)

            fig_page_type = plot_events_by_page_type(df)

            st.plotly_chart(fig_page_type, use_container_width=True, key="bar_chart_1")

        st.subheader("Page Type Analytics (COMPLETE)")
        st.markdown("Complete (2024) analysis of page type on a monthly basis")
        df = fetch_data_from_db_pagetype(selected_year, selected_month)

        fig = plot_events_by_page_type_1(df)
        st.plotly_chart(fig, use_container_width=True, key="bar_chart_2")

        st.subheader("Page-Wise Analytics")
        st.markdown("Page-level analytics breakdown")

        df = fetch_data_from_db(selected_year, selected_month)

        page_type = st.selectbox(
            'Select Page Type',
            df[df['page_type'].isin(['IRL Detail', 'Other']) == False]['page_type'].unique()
        )

        top_10_df = (
            df[df['page_type'] == page_type]
            .sort_values(by='event_count', ascending=False)
            .head(10)
        )

        create_bar_chart(top_10_df, page_type)

    elif page == 'Network Density':
        st.title("Network Density")

        st.subheader("Member Connection Strength")
        file_path = "./network-strength/followersfollowing.csv"
        networkStrength(file_path, "member-strength")
       
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

        st.title("Charts with Filters")
        st.subheader("Filters")
        years = ["All", "2023", "2024"]
        months = [
            "January", "February", "March", "April", "May", "June", 
            "July", "August", "September", "October", "November", "December"
        ]
        
        selected_year = st.selectbox("Select Year", years, index=0)
        selected_month = st.selectbox("Select Month", ["All"] + months, index=0)

        df = fetch_events_by_month_location(selected_year, selected_month)
        df['event_month'] = pd.to_datetime(df['event_month'])
        if df.empty:
            fig = px.bar()
            fig.update_layout(
                annotations=[dict(
                    text="No data available for selected filters",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=24, color="red", family="Arial Black"),
                    align="center"
                )]
            )
        else:
            fig = plot_events_by_month(df)

        df = fetch_hosts_and_speakers_by_month(selected_year, selected_month)
        if df.empty:
            fig_1 = px.bar()
            fig_1.update_layout(
                annotations=[dict(
                    text="No data available for selected filters",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=24, color="red", family="Arial Black"),
                    align="center"
                )]
            )
        else:
            fig_1 = plot_hosts_and_speakers_distribution(df)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Events by Month")
            st.markdown("Monthly event distribution by geography")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Distribution of Hosts and Speakers by Month")
            st.markdown("Distribution of Speakers and Hosting teams/members by month")
            st.plotly_chart(fig_1, use_container_width=True, key=1)

        selected_data_type = st.radio("Select Type", ("User", "Team"))

        if selected_data_type == "User":
            st.subheader("Tracking Event Participation by Member Role Over Time")
            event_data = fetch_events_by_month_by_user_1(selected_year, selected_month)
        else:
            st.subheader("Tracking Event Participation by Team Role Over Time")
            event_data = fetch_events_by_month_by_team(selected_year, selected_month)

        event_data_melted = event_data.melt(id_vars=["event_name"],
                                            value_vars=["host_count", "speaker_count", "attendee_count"],
                                            var_name="role", value_name="count")

        role_map = {"host_count": "Host", "speaker_count": "Speaker", "attendee_count": "Attendee"}
        event_data_melted['role'] = event_data_melted['role'].map(role_map)
        if event_data_melted.empty:
            fig_2 = px.bar()
            fig_2.update_layout(
                annotations=[dict(
                    text="No data available for selected filters",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=24, color="red", family="Arial Black"),
                    align="center"
                )]
            )
        else:
            fig_2 = px.bar(
                event_data_melted,
                x="event_name", 
                y="count",  
                color="role",  
                labels={"event_name": "Event Name", "count": "Count", "role": "Role"},
                hover_data={"event_name": True, "role": True, "count": True},
                barmode="stack"  
            )

            fig_2.update_layout(
                xaxis_title="Event Name",
                yaxis_title="Count",
                xaxis_tickangle=-45,  
                plot_bgcolor='rgba(0,0,0,0)',  
                paper_bgcolor='rgba(0,0,0,0)',  
            )
        
        data = fetch_attendee_data("All", "All")
        fig_3 = px.pie(data, 
                    names="office_hours_status", 
                    values="attendee_count", 
                    color_discrete_sequence=px.colors.qualitative.Set3)
        

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Event Engagement Analysis")
            st.markdown("User/Team event attendance")
            st.plotly_chart(fig_2)
        
        st.title("Charts with No Filters")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Attendees Breakdown by Topic / Skill")
            st.markdown("Distribution of event attendees by Topic / Skill")
            option = st.radio("Select Type:", ('Topic', 'Skill'))
            if option == 'Topic':
                visualize_attendees_by_topic("All", "All")
            elif option == 'Skill':
                visualize_attendees_by_skill("All", "All")
        with col2:
            st.subheader("Percentage of Attendees with and without Office Hours")
            st.plotly_chart(fig_3)

        df = fetch_hosting_teams_by_focus_area("All", "All")
        fig_4 = plot_hosting_teams_by_focus_area(df)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Focus Area Segmentation of Hosting Teams")
            st.markdown("Distribution of Hosting Teams by focus areas")
            st.plotly_chart(fig_4, use_container_width=True)
        with col2:
            visualize_event_topic_distribution("All", "All")

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

        st.subheader("Filters")
        years = ["All","2021","2022","2023","2024"] 

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

        st.subheader("New Entries Trend by Month")
        fetch_and_plot_all_entries(selected_year, selected_month)

        df = fetch_member_by_skills()
        fig = px.pie(
            df, 
            names="skill_name",  
            values="member_count", 
            color_discrete_sequence=px.colors.qualitative.Pastel  
        )

        fig.update_traces(
            hovertemplate="<b>Skill:</b> %{label}<br><b>Members:</b> %{value}<br><b>Percent:</b> %{percent}<extra></extra>"
        )
        fig.update_layout(
            legend_title="Skills",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )

        df = teams_by_focus_area()

        fig_1 = px.pie(
            df, 
            names="focus_area",  
            values="Teams", 
            color_discrete_sequence=px.colors.qualitative.Pastel  
        )

        fig_1.update_traces(
            hovertemplate="<b>Skill:</b> %{label}<br><b>Members:</b> %{value}<br><b>Percent:</b> %{percent}<extra></extra>"
        )
        fig_1.update_layout(
            legend_title="Focus Area",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribution graph of people by skills")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Distribution graph of Team by Focus Area")
            st.plotly_chart(fig_1, use_container_width=True)

        df = projects_by_focus_area()

        fig_2 = px.pie(
            df, 
            names="focus_area",  
            values="Projects", 
            color_discrete_sequence=px.colors.qualitative.Pastel  
        )

        fig_2.update_traces(
            hovertemplate="<b>Skill:</b> %{label}<br><b>Members:</b> %{value}<br><b>Percent:</b> %{percent}<extra></extra>"
        )
        fig_2.update_layout(
            legend_title="Focus Area",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribution graph of Project by Focus Area")
            st.plotly_chart(fig_2, use_container_width=True)

    elif page == 'Usage Activity':
        st.title("Usage Activity")

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

        st.title("Search Usage")
        event_type = st.selectbox(
            'Select Page Type',
            ['Project', 'IRL', 'Member', 'Team']
        )

        selected_event_name = event_name_mapping[event_type]

        user_status = st.selectbox(
            'Select User Status',
            ['All', 'Logged-In', 'Logged-Out']
        )

        df_event = fetch_search_event_data(selected_event_name, selected_year, selected_month, user_status)

        df_event_top_10 = df_event.sort_values(by='event_count', ascending=False).head(10)

        df_event_top_10_grouped = df_event_top_10.groupby(['name', 'search_value'], as_index=False).agg({'event_count': 'sum'})

        df_event_top_10_grouped['color_group'] = df_event_top_10_grouped['search_value'].fillna('Guest User')

        df_event_top_10_grouped['user_search_combined'] = df_event_top_10_grouped['name'] + ' - ' + df_event_top_10_grouped['search_value'].fillna('No Search Value')

        fig = px.bar(
            df_event_top_10_grouped,
            x='user_search_combined',  
            y='event_count', 
            text='event_count',  
            labels={'user_search_combined': 'User - Search Value', 'event_count': 'Count', 'search_value': 'Search Value'}        )

        fig.update_traces(texttemplate='%{text}', textposition='outside', insidetextanchor='middle')

        with st.expander("Overall Data"):

            df_modified = df_event.copy()

            column_mapping = {
                'name': 'Name',
                'search_value': 'Search Value',
                'event_count': 'Event Count'
            }

            df_modified.columns = df_modified.columns.str.lower().str.replace(' ', '_')

            df_modified = df_modified.rename(columns=column_mapping)

            if 'Search Value' in df_modified.columns:
                df_modified = df_modified[['Name', 'Search Value', 'Event Count']]
            else:
                st.warning("'Search Value' column is missing. Showing other columns.")
                df_modified = df_modified[['Name', 'Event Count']]

            st.dataframe(df_modified, use_container_width=True)

        df_event_monthly = fetch_search_event_data_by_month(selected_event_name, selected_year, selected_month, user_status)

        df_event_monthly['month_start'] = pd.to_datetime(df_event_monthly['month_start']).dt.strftime('%B %Y')


        fig_monthly = px.bar(
            df_event_monthly,
            x='month_start',  
            y='event_count', 
            color='user_status',  
            labels={'month_start': 'Month-Year', 'event_count': 'Count', 'user_status': 'User Status'},
        )

        fig_monthly.update_traces(
            hovertemplate='Count: %{y}<extra></extra>',  
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Top Users Engaging with Event Searches')
            st.markdown("Shows Users who are most active in searching for events")
            st.plotly_chart(fig)
        with col2:
            st.subheader("Monthly Breakdown of Top Users Engaging in Event Searches")
            st.plotly_chart(fig_monthly)

        st.title("Social Link Engagement Overview")
        st.markdown("Members who clicked on other Members social links, indicating engagement and interaction within the platform")

        contact_event_type = st.selectbox(
        'Select Social Link Type',
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
        df_result =execute_query(query)

        df_event_top_10 = df_result.sort_values(by='event_count', ascending=False).head(10)

        df_event_top_10['user_clicked'] = df_event_top_10['name'] + " - " + df_event_top_10['clicked_name']

        fig = px.bar(
            df_event_top_10,
            x='user_clicked',  
            y='event_count',  
            text='event_count', 
            labels={'user_clicked': 'User - Clicked Name', 'event_count': 'Count'},
        )

        fig.update_traces(textposition='outside') 

        fig.update_layout(
            margin=dict(l=50, r=50, t=50, b=50),  
            xaxis_title='User - Clicked Name',  
            yaxis_title='Count',
        )

        fig.update_traces(texttemplate='%{text}', textposition='outside', insidetextanchor='middle')

        fig.update_layout(xaxis_tickangle=45)  
        st.plotly_chart(fig)

        with st.expander("Overall Data"):

            df_modified = df_result.copy()

            df_modified.columns = df_modified.columns.str.lower().str.replace(' ', '_')

            if 'search_value' in df_modified.columns:
                df_modified = df_modified[['email', 'name', 'search_value', 'event_count', 'user_status']]
            else:
                df_modified = df_modified[['clicked_name', 'type', 'name', 'event_count']]

            df_modified = df_modified.rename(columns={
                'clicked_name': 'Clicked By',
                'type': 'Social Link Type',
                'name': 'Clicked Member',
                'event_count': 'Event Count'
            })

            st.dataframe(df_modified, use_container_width=True)

        st.title("Monthly Clicks Breakdown by Page Type")

        page = st.selectbox("Select Page", ["Team Click Through", "Member Click Through", "Project Click Through"])
        if page == 'Team Click Through':
            df = fetch_team_data_clicked(selected_year, selected_month)
            df_top_10 = df.groupby(['clickedby', 'page_type'], as_index=False).agg({'event_count': 'sum'})

            df_top_10_sorted = df_top_10.sort_values(by='event_count', ascending=False).head(10)

            fig_top_10 = px.bar(
                df_top_10_sorted,
                x='clickedby',  
                y='event_count',  
                text='event_count',  
                labels={'clickedby': 'User', 'event_count': 'Count', 'page_type': 'Page Type'},
                color='page_type',  
            )

            fig_top_10.update_traces(texttemplate='%{text}', textposition='outside', insidetextanchor='middle')

            fig_top_10.update_layout(showlegend=True)
            
            with st.expander("Overall Data"):
                df_modified = df.copy()

                df_modified.columns = df_modified.columns.str.lower().str.replace(' ', '_')

                df_modified = df_modified[['clickedby', 'team_name', 'page_type', 'event_count']]

                df_modified = df_modified.rename(columns={
                    'clickedby': 'Clicked By',
                    'team_name': 'Team Name',
                    'page_type': 'Page Type',
                    'event_count': 'Event Count'
                })

                df_modified = df_modified.dropna()

                st.dataframe(df_modified, use_container_width=True)

            df['month'] = pd.to_datetime(df['month'], format='%B %Y')  

            df_monthwise = df.groupby(['month', 'page_type'], as_index=False)['event_count'].sum()

            fig_monthwise = px.bar(
                df_monthwise,
                x='month',  
                y='event_count',  
                color='page_type',  
                labels={'month': 'Month-Year', 'event_count': 'Count', 'page_type': 'Page Type'},
            )

            fig_monthwise.update_layout(
                xaxis_title='Month-Year',
                yaxis_title='Click Count',
                legend_title='Page Type',
                xaxis=dict(
                    tickformat='%b %Y',
                    tickmode='array',
                    tickvals=df_monthwise['month'].unique(),  
                    ticktext=[month.strftime('%b %Y') for month in sorted(df_monthwise['month'].unique())]
                ),
            )
            
        elif page == 'Member Click Through':
            pass

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Analysis of Click-Through Rates by Team (By User)")
            st.markdown("Click-through rates for each team by user, helping to understand how different teams are engaging with the content")
            st.plotly_chart(fig_top_10)
        with col2:
            st.subheader("Analysis of Click-Through Rates by Team (Monthly)")
            st.markdown("Click-through rates by team, helping to identify variations in user engagement across different time periods")
            st.plotly_chart(fig_monthwise)

        st.title("Husky Interaction")
        df = husky_prompt_interaction()

        fig = px.pie(
            df, 
            names="name",  
            values="interaction_count", 
            color_discrete_sequence=px.colors.qualitative.Pastel  
        )

        fig.update_traces(
            hovertemplate="<b>UserName:</b> %{label}<br><b>InteractionCount:</b> %{value}<br><b>Percent:</b> %{percent}<extra></extra>"
        )
        fig.update_layout(
            legend_title="User Name",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )

        df = husky_feedback()
        fig_1 = px.pie(
            df, 
            names="name",  
            values="interaction_count", 
            color_discrete_sequence=px.colors.qualitative.Pastel  
        )

        fig_1.update_traces(
            hovertemplate="<b>UserName:</b> %{label}<br><b>Interaction count:</b> %{value}<br><b>Percent:</b> %{percent}<extra></extra>"
        )
        fig_1.update_layout(
            legend_title="User Name",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Interaction with Husky")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Husky Feedback")
            st.plotly_chart(fig_1, use_container_width=True)

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

def create_pie_chart(team_stats):
    """Creates a pie chart for team statistics."""
    average_count = team_stats['Count'].mean()
    max_count = team_stats['Count'].max()
    min_count = team_stats['Count'].min()

    stats_df = pd.DataFrame({
        'Category': ['Above Average', 'Below Average', 'Maximum', 'Minimum'],
        'Count': [
            len(team_stats[team_stats['Count'] > average_count]),
            len(team_stats[team_stats['Count'] < average_count]),
            len(team_stats[team_stats['Count'] == max_count]),
            len(team_stats[team_stats['Count'] == min_count])
        ]
    })

    fig = px.pie(
        stats_df,
        names='Category',
        values='Count',
        title='Team Interaction Statistics Distribution',
        color='Category',
        color_discrete_map={'Above Average': 'green', 'Below Average': 'red', 'Maximum': 'blue', 'Minimum': 'orange'}
    )

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

    fig = px.bar(filtered_teams,
                 x='Team',
                 y='Count',
                 title=f'Teams with {selected_category} Interactions',
                 labels={'Team': 'Team', 'Count': 'Interaction Count'},
                 color='Team',
                 text=None) 

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

    return df  

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