import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine

# Database connection
def get_database_connection():
    DATABASE_URL = """postgresql://postgres:brLmktUZ5tBjShjk@metabase-analytics.c0eujdj3dv3n.us-west-1.rds.amazonaws.com:5432/postgres"""
    engine = create_engine(DATABASE_URL)
    return engine

# Fetch data for Team Click Through (Team Page)
def fetch_team_data(engine): 
    query = """
     SELECT 
    COALESCE(
        pe.properties->'teamName',
        pe.properties->'name') as team_name,
    COALESCE(
        pe.properties->>'loggedInUserName', 
        pe.properties->'user'->>'name'
    ) AS ClickedBy,
    pe.timestamp,
    TO_CHAR(pe.timestamp, 'Month') AS month,
    EXTRACT(DAY FROM pe.timestamp) AS day,
    EXTRACT(YEAR FROM pe.timestamp) AS year,
    CASE
        WHEN pe."event" = 'irl-guest-list-table-team-clicked' THEN 'IRL Page'
        WHEN pe."event" = 'team-clicked' THEN 'Teams Landing Page'
        WHEN pe."event" = 'memeber-detail-team-clicked' THEN 'Member Profile Page'
        WHEN pe."event" = 'project-detail-maintainer-team-clicked' THEN 'Project Page'
        ELSE 'Other'
    END AS page_type,
    CASE
        WHEN MAX(pe2.timestamp) IS NOT NULL THEN 'LoggedOut'
        ELSE 'LoggedIn'
    END AS user_status
FROM posthogevents pe
LEFT JOIN posthogevents pe2 
    ON pe2.properties->>'$session_id' = pe.properties->>'$session_id'
    AND (pe2.properties->>'loggedInUserEmail' IS NOT NULL 
        OR pe2.properties->'user'->>'email' IS NULL)
    AND pe2.timestamp < pe.timestamp
WHERE 
    (pe.properties->>'loggedInUserEmail' IS NOT NULL 
     OR pe.properties->'user'->>'email' IS NOT NULL)
    AND pe."event" IN ('irl-guest-list-table-team-clicked', 'team-clicked', 'memeber-detail-team-clicked', 'project-detail-maintainer-team-clicked')
    AND COALESCE(
        pe.properties->>'loggedInUserName', 
        pe.properties->'user'->>'name'
    ) NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
GROUP BY 
    ClickedBy, pe.timestamp, month, day, year, page_type, team_name
ORDER BY 
    ClickedBy, pe.timestamp;
    """
    return pd.read_sql(query, engine)

# Helper function to calculate "Month X Week Y"
def calculate_week(df):
    df['week_of_month'] = ((df['day'] - 1) // 7 + 1).astype(int)  # Ensure it's an integer
    
    df['week_label'] = (
        df['month'].str.strip() + " " +  # Month name
        df['week_of_month'].apply(lambda x: f"{x}{'st' if x == 1 else 'nd' if x == 2 else 'rd' if x == 3 else 'th'}") + " Week"
    )
    return df

# Create a bar chart for team clicks
def plot_bar_chart(data, time_aggregation):
    if time_aggregation == "Month":
        group_col = 'month'
    else:
        group_col = 'week_label'

    # Aggregate the data
    click_counts = data.groupby([group_col, 'page_type']).size().reset_index(name='click_count')

    # Create the bar chart
    fig = px.bar(
        click_counts,
        x=group_col,
        y='click_count',
        color='page_type',
        labels={"click_count": "Number of Clicks", group_col: time_aggregation, "page_type": "Page Type"},
        title=f"Team Clicks Over Time by {time_aggregation}",
        category_orders={group_col: sorted(click_counts[group_col].unique())}
    )

    fig.update_layout(
        barmode='stack',
        legend_title="Page Type",
        legend=dict(
            x=1,
            y=1,
            traceorder='normal',
            orientation='h',
            font=dict(size=12)
        ),
        showlegend=True
    )

    return fig

# Fetch focus area data
def fetch_focus_area_data(engine):
    query = """
        SELECT 
            COALESCE(parentFA.title, fa.title) AS "FocusArea",
            COUNT(t.uid) AS "TeamCount"
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
            COALESCE(parentFA.title, fa.title)
        ORDER BY 
            "TeamCount" DESC;
    """
    return pd.read_sql(query, engine)

# Create a pie chart for focus area data
def plot_pie_chart(data):
    fig = px.pie(
        data,
        names="FocusArea",
        values="TeamCount",
        title="Focus Area Distribution of Teams",
        labels={"FocusArea": "Focus Area", "TeamCount": "Number of Teams"}
    )
    fig.update_traces(textinfo='percent+label')
    return fig

def fetch_total_teams_count(engine):
    query = """
        SELECT COUNT(*) AS total_teams
        FROM public."Team"
    """
    result = pd.read_sql(query, engine)
    return result['total_teams'][0]

 # Fetch data for Team Search
def fetch_team_search_data(engine):
    query = """
        SELECT 
            COALESCE(
                properties->>'loggedInUserEmail', 
                properties->'user'->>'email'
            ) AS email,
            COALESCE(
                NULLIF(properties->>'loggedInUserName', ''),  -- Treat empty strings as NULL
                NULLIF(properties->'user'->>'name', ''), 
                'Untracked User'  -- Default value if name is missing or empty
            ) AS name,
            COUNT(*) AS event_count
        FROM posthogevents
        WHERE 
            "event" = 'team-search'
        GROUP BY 
            email, name
        ORDER BY 
            event_count DESC
        LIMIT 10;
    """
    return pd.read_sql(query, engine)
    
# Create a bar chart for Team Search
def plot_team_search_bar_chart(data):
    fig = px.bar(
        data,
        x='name',
        y='event_count',
        labels={"event_count": "Number of Searches", "name": "User Name"},
        title="Top 10 Users by Team Search Event Count",
        color='name'
    )
    fig.update_layout(
        xaxis_title="User Name",
        yaxis_title="Event Count",
        showlegend=False
    )
    return fig

def main():
    st.set_page_config(page_title="nKPI Dashboard", layout="wide")

    # Sidebar for page navigation
    page = st.sidebar.radio("Select Page", ["Team Analysis",])

    engine = get_database_connection()

    if page == "Team Analysis":
        st.title("Team Analysis")

        st.markdown("""
           This page visualizes the number of team clicks across different events, such as "IRL Page," "Team Page," "Member Profile Page," and "Project Page."
            You can filter the data by selecting a specific year and month to view the trends.
            Additionally, the data can be grouped either by month or week for detailed insights.
            The bar chart provides a visual representation of the number of clicks over time, segmented by the event type, while the table below displays the raw data for further analysis.
        """)

        df = fetch_team_data(engine)

        # Fix Year as an integer
        df['year'] = df['year'].astype(int)

        # Calculate "Month X Week Y" labels
        df = calculate_week(df)

        # Fetch total number of unique teams from the database
        total_teams = fetch_total_teams_count(engine)

        # Filters for Year and Month
        st.subheader("Filters")
        years = sorted(df['year'].unique())
        months = sorted(df['month'].unique())

        years = ["All"] + years
        selected_year = st.selectbox("Select Year", years, index=0)
        selected_month = st.selectbox("Select Month", ["All"] + months, index=0)
        loggedin_filter = st.selectbox("Select User Status", ["All", "LoggedIn - Active", "LoggedOut User"], index=0)


        if selected_year != "All":
            df = df[df['year'] == int(selected_year)]
        if selected_month != "All":
            df = df[df['month'] == selected_month]

        # Display the total teams count on the left
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Total number of teams included in the network")
            st.metric(label="Total Teams", value=f"{total_teams}")

        # Weekly or Monthly aggregation
        time_aggregation = st.radio("Select Time Aggregation", ["Month", "Week"], index=0)

        with col2:
            st.subheader("Views on Team Profile Pages(MAUs, WAUs)")
            fig = plot_bar_chart(df, time_aggregation)
            st.plotly_chart(fig, use_container_width=True)

        # Pie Chart for Focus Area Distribution
        st.subheader("Percentage split of teams across the three focus areas")
        focus_area_data = fetch_focus_area_data(engine)
        pie_fig = plot_pie_chart(focus_area_data)
        st.plotly_chart(pie_fig, use_container_width=True)

        # Fetch Team Search Data
        team_search_data = fetch_team_search_data(engine)

        # Bar chart for Team Search Events
        st.subheader("Top 10 Team Search Users")
        team_search_fig = plot_team_search_bar_chart(team_search_data)
        st.plotly_chart(team_search_fig, use_container_width=True)
        

        # Table view for the data
        st.subheader("Overall Raw Data")
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        if 'id' in df.columns:
            df = df.drop(columns=['id'])

        df['month'] = df['month'].str.strip()
        df['year'] = df['year'].apply(lambda x: str(x))
        df.columns = df.columns.str.replace('_', ' ').str.title().str.replace(' ', '')

        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()
