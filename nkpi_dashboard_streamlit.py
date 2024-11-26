import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine
import plotly.graph_objects as go
import calendar
import os
from dotenv import load_dotenv
load_dotenv()

# Database connection
def get_database_connection():
    DATABASE_URL = os.getenv("DB_URL")
    engine = create_engine(DATABASE_URL)
    return engine


def fetch_team_data(engine):
    query = """
        SELECT 
    COALESCE(
        pe.properties->'teamName',
        pe.properties->'name'
    ) AS team,
    COALESCE(
        pe.properties->>'loggedInUserName', 
        pe.properties->'user'->>'name'
    ) AS ViewedBy,
    pe.timestamp,
    EXTRACT(MONTH FROM pe.timestamp) AS month,  -- Extract as number
    EXTRACT(DAY FROM pe.timestamp) AS day,
    EXTRACT(YEAR FROM pe.timestamp) AS year,
    CASE
        WHEN pe."event" = 'irl-guest-list-table-team-clicked' THEN 'IRL Page'
        WHEN pe."event" = 'team-clicked' THEN 'Teams Landing Page'
        WHEN pe."event" = 'memeber-detail-team-clicked' THEN 'Member Profile Page'
        WHEN pe."event" = 'project-detail-maintainer-team-clicked' THEN 'Project Page'
        ELSE 'Other'
    END AS source_type,
    CASE
        WHEN (pe.properties->>'loggedInUserEmail' IS NOT NULL OR pe.properties->'user'->>'email' IS NOT NULL) THEN 'LoggedIn'
        ELSE 'LoggedOut'
    END AS user_status,
    COUNT(*) AS clicks
FROM 
    posthogevents pe
WHERE 
    pe."event" IN ('irl-guest-list-table-team-clicked', 'team-clicked', 'memeber-detail-team-clicked', 'project-detail-maintainer-team-clicked')
    AND COALESCE(
        pe.properties->>'loggedInUserName', 
        pe.properties->'user'->>'name'
    ) NOT IN ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
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
    EXTRACT(MONTH FROM pe.timestamp),
    pe.timestamp;
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

def plot_bar_chart(df, time_aggregation):
    if time_aggregation == "Month":
        df['month_name'] = df['month'].apply(lambda x: calendar.month_name[x])
        
        month_order = list(calendar.month_name[1:])  # January to December
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

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=time_aggregation,
        yaxis_title="Views",
        barmode='stack',
        template="plotly_dark",
        xaxis_tickangle=45 
    )
    
    return fig

def plot_pie_chart(focus_area_data):
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
           AND (
                COALESCE(properties->>'loggedInUserName', properties->'user'->>'name') NOT IN 
                ('La Christa Eccles', 'Winston Manuel Vijay A', 'Abarna Visvanathan', 'Winston Manuel Vijay')
                OR COALESCE(properties->>'loggedInUserName', properties->'user'->>'name') IS NULL
            )
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
        SELECT COUNT(*) AS total_interactions
        FROM public."MemberInteraction"
    """
    result = pd.read_sql(query, engine)
    return result['total_interactions'][0]


def fetch_member_interactions_data(engine):
    query = """
       SELECT 
            m.name AS office_hours_initiated_by,
            m2.name AS office_hours_initiated_to,
            TO_CHAR(mi."createdAt", 'Month YYYY') AS month,
            EXTRACT(YEAR FROM mi."createdAt") AS year,
            CASE 
                WHEN mfu.status = 'COMPLETED' THEN 
                    CASE 
                        WHEN mf.response = 'POSITIVE' THEN 'Responded'
                        WHEN mf.response = 'NEGATIVE' THEN 'Cancelled'
                        ELSE 'Completed with unknown response'
                    END
                WHEN mfu.status = 'CLOSED' THEN 'Dismissed'
                WHEN mfu.status = 'PENDING' THEN 'Did not respond'
                ELSE 'Unknown status'
            END AS meeting_status,
            CASE
                WHEN mf.response = 'NEGATIVE' THEN mf."comments"
                ELSE NULL
            END AS negative_feedback_comments
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
        ORDER BY 
            mi."createdAt" DESC;
    """
    return pd.read_sql(query, engine)
    

def main():
    st.set_page_config(page_title="nKPI Dashboard", layout="wide")

    page = st.sidebar.radio("Select Page", ["Team Analysis"])

    engine = get_database_connection()

    if page == "Team Analysis":
        st.title("Team Analysis")

        st.markdown("""
          This page provides a visualization of team clicks across various events, including "IRL Page," "Team Page," "Member Profile Page," and "Project Page." You can filter the data by selecting a specific year and month. The insights can be displayed either monthly or weekly, depending on your preference.
          In addition to the team clicks data, the page also includes team search information and the number of teams created each month.
        """)

        df = fetch_team_data(engine)
        df['year'] = df['year'].astype(int)
        

        total_teams = fetch_total_teams_count(engine)

        st.markdown(
            f"""
            <div style="background-color:#f5f5f5; padding:20px; border-radius:8px; text-align:center;">
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

        df = calculate_week(df)

        if selected_year != "All":
            df = df[df['year'] == int(selected_year)]

        if selected_month != "All":
            month_num = month_mapping.get(selected_month)  
            df = df[df['month'] == month_num]


        st.subheader("Activity (MAUs, WAUs) - Directory Teams")

        col1, col2 = st.columns([1, 3])
        with col1:
            time_aggregation = st.radio(
                "View by - Month, Week",
                ["Month", "Week"],
                index=0
            )

        with col2:
            fig = plot_bar_chart(df, time_aggregation)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Teams added to Network")
        teams_per_month = fetch_teams_per_month(
            engine,
            year=int(selected_year) if selected_year != "All" else None,
            month=month_mapping.get(selected_month, None) if selected_month != "All" else None,
        )

        visualize_teams_per_month(teams_per_month)

        st.subheader("Team Segmentation by Focus Areas")
        focus_area_data = fetch_focus_area_data(engine)
        pie_fig = plot_pie_chart(focus_area_data)
        st.plotly_chart(pie_fig, use_container_width=True)

        loggedin_filter = st.selectbox("Select User Status", ["All", "LoggedIn User(Active)", "LoggedOut User"], index=0)

        team_search_data = fetch_team_search_data(
            engine,
            year=int(selected_year) if selected_year != "All" else None,
            month=month_mapping.get(selected_month, None) if selected_month != "All" else None,
            user_status="loggedin" if loggedin_filter == "LoggedIn User(Active)" else ("loggedout" if loggedin_filter == "LoggedOut User" else None)
        )

        st.subheader("Team Search Data")
        team_search_data = team_search_data.drop(columns=['event_count'])
        team_search_data.columns = team_search_data.columns.str.replace('_', ' ').str.title().str.replace(' ', '')
        st.dataframe(team_search_data, use_container_width=True)


        with st.expander("Overall Data"):
            df_modified = df.copy()

            df_modified.columns = df_modified.columns.str.lower().str.replace(' ', '_')

            if 'id' in df_modified.columns:
                df_modified = df_modified.drop(columns=['id'])
            if 'timestamp' in df_modified.columns:
                df_modified['date'] = df_modified['timestamp'].dt.date 

            # df_modified = df_modified.drop(columns=['month', 'day', 'year', 'week_of_month', 'week_start_date', 'week_end_date', 'week_range', 'user_status', 'month-week', 'timestamp', 'clicks', 'month_name'])
            columns_to_drop = ['month', 'day', 'year', 'week_of_month', 'week_start_date', 'week_end_date', 'week_range', 
                   'user_status', 'month-week', 'timestamp', 'clicks', 'month_name','week_start', 'week_end' ]

            # Drop the columns if they exist in the DataFrame
            df_modified = df_modified.drop(columns=[col for col in columns_to_drop if col in df_modified.columns])
            df_modified.columns = df_modified.columns.str.replace('_', ' ').str.title().str.replace(' ', '')
            st.dataframe(df_modified, use_container_width=True)

    # elif page == "Office Hours Analysis":
    #     # Member Interactions Page
    #     st.title("Office Hours Analysis")

    #     # Show a description of the app
    #     st.markdown("""
    #         This page displays the details of interactions between members, including feedback and follow-up status.
    #         You can filter the data based on the year and month of the interaction.
    #     """)

    #     # Fetch total number of interactions
    #     total_interactions = fetch_total_interaction_count(engine)

    #     # **Move Total Teams Metric Above Filters**
    #     st.markdown(
    #         f"""
    #         <div style="background-color:#f5f5f5; padding:20px; border-radius:8px; text-align:center;">
    #             <h3>Number of teams in the Network</h3>
    #             <p style="font-size:28px; font-weight:bold; color:#2b2b2b;">{total_interactions}</p>
    #         </div>
    #         """,
    #         unsafe_allow_html=True
    #     )

    #     df = fetch_member_interactions_data(engine)
    #     df['year'] = df['year'].astype(int)
    #     years = sorted(df['year'].unique())
    #     months = sorted(df['month'].str.split(' ').str[0].unique())
    #     years = ["All"] + years
    #     selected_year = st.selectbox("Select Year", years, index=0)
    #     if selected_year != "All":
    #         months = sorted(df[df['year'] == int(selected_year)]['month'].str.split(' ').str[0].unique())
    #     months = ["All"] + months
    #     selected_month = st.selectbox("Select Month", months, index=0)

    #     # Apply filters based on selected year and month
    #     if selected_year != "All":
    #         df = df[df['year'] == int(selected_year)]  # Filter data by selected year
    #     if selected_month != "All":
    #         df['month'] = df['month'].str.split(' ').str[0]  # Extract only the month name
    #         df = df[df['month'] == selected_month] 

    #     st.subheader("Data Overview")
    #     df.columns = df.columns.str.lower().str.replace(' ', '_')
    #     # Remove the first unwanted column (if it exists, like "id")
    #     if 'id' in df.columns:
    #         df = df.drop(columns=['id'])


    #     # Remove the year from the month column (for the table display only)
    #     df['month'] = df['month'].str.split(' ').str[0]  # Extract only the month name

    #     # Ensure the `year` is displayed correctly (no thousands separator)
    #     df['year'] = df['year'].apply(lambda x: str(x))  # Convert year to string without formatting issues

    #     # Capitalize column names with title case
    #     df.columns = df.columns.str.replace('_', ' ').str.title().str.replace(' ', '')  # Capitalize and remove underscores

    #     st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()