import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import numpy as np
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Performance Timeline - AI ShuttleCoach",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Dark theme base */
    .stApp {
        background-color: #121212;
        color: #f0f0f0;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #1e1e1e;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #1a1a1a;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        color: #4fc3f7;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #b0bec5;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #263238;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4fc3f7;
    }
    
    /* Metric box */
    .metric-box {
        background-color: #37474f;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 1rem;
        border: 1px solid #546e7a;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6, p, li, span, div {
        color: #f0f0f0;
    }
    
    /* Links */
    a {
        color: #4fc3f7;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #0d47a1;
        color: white;
        border: none;
        border-radius: 0.3rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: #263238;
        border-radius: 0.3rem;
        color: #b0bec5;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0d47a1;
        color: white;
    }
    
    /* Dataframe */
    .dataframe {
        background-color: #263238;
        border-radius: 0.5rem;
        overflow: hidden;
    }
    
    /* Success box */
    .success-box {
        background-color: #1b5e20;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='main-header'>📊 Performance Timeline</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #616161;'>Track Your Badminton Progress</h3>", unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class='info-box'>
    <h4>📈 How It Works</h4>
    <p>This timeline shows your performance metrics over time, helping you track your progress and identify areas for improvement. 
    The data is collected from your training sessions and matches with AI ShuttleCoach.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with options
with st.sidebar:
    st.markdown("## 🛠️ Options")
    
    # Date range selector
    st.markdown("### 📅 Date Range")
    today = datetime.now().date()  # Convert to date object
    default_start = today - timedelta(days=4)  # Changed from 7 to 4 days
    
    start_date = st.date_input(
        "Start Date",
        value=default_start,
        max_value=today,
        help="Select the start date for your performance data"
    )
    
    end_date = st.date_input(
        "End Date",
        value=today,
        max_value=today,
        help="Select the end date for your performance data"
    )
    
    # Validate date range
    if start_date > end_date:
        st.error("Start date must be before end date")
        st.stop()
    
    if start_date > today or end_date > today:
        st.error("Cannot select future dates")
        st.stop()
    
    # Metric selector
    st.markdown("### 📊 Metrics")
    selected_metrics = st.multiselect(
        "Select Metrics to Display",
        ["Pose Accuracy (%)", "Shots per Drill", "Avg Reaction Time (s)", "Mistakes per Session", 
         "Footwork Score", "Shot Accuracy (%)", "Serve Quality", "Smash Power"],
        default=["Pose Accuracy (%)", "Shots per Drill", "Avg Reaction Time (s)", "Mistakes per Session"],
        help="Choose which metrics to include in your timeline"
    )
    
    # Data source selector
    st.markdown("### 📁 Data Source")
    data_source = st.radio(
        "Select Data Source",
        ["All Sessions", "Training Only", "Matches Only"],
        help="Filter data by session type"
    )
    
    # Export options
    st.markdown("### 📥 Export Options")
    export_format = st.selectbox(
        "Export Format",
        ["Excel", "CSV", "PDF"],
        help="Choose the format for your exported report"
    )

# Generate simulated performance data
def generate_performance_data(start_date, end_date, data_source):
    # Convert dates to datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Calculate number of days
    days = (end - start).days + 1
    
    # Generate date range
    dates = pd.date_range(start=start, end=end, freq='D')
    
    # Base metrics with some randomness
    base_metrics = {
        "Pose Accuracy (%)": 85,
        "Shots per Drill": 40,
        "Avg Reaction Time (s)": 0.4,
        "Mistakes per Session": 4,
        "Footwork Score": 75,
        "Shot Accuracy (%)": 80,
        "Serve Quality": 70,
        "Smash Power": 65
    }
    
    # Generate data with trends and randomness
    data = {"Date": dates}
    
    for metric, base_value in base_metrics.items():
        # Add some randomness and a slight upward trend
        trend = np.linspace(0, 5, days)  # Slight upward trend
        noise = np.random.normal(0, 2, days)  # Random noise
        
        if "Time" in metric:  # For time metrics, lower is better
            values = base_value - trend + noise
        else:  # For other metrics, higher is better
            values = base_value + trend + noise
        
        # Ensure values are within reasonable bounds
        if "Accuracy" in metric or "Score" in metric or "Quality" in metric or "Power" in metric:
            values = np.clip(values, 0, 100)
        elif "Time" in metric:
            values = np.clip(values, 0.2, 1.0)
        elif "Mistakes" in metric:
            values = np.clip(values, 0, 10)
        else:
            values = np.clip(values, 10, 100)
        
        data[metric] = values
    
    # Filter by data source if needed
    if data_source == "Training Only":
        # Simulate training sessions on certain days
        mask = np.random.choice([True, False], size=days, p=[0.7, 0.3])
        for i, date in enumerate(dates):
            if not mask[i]:
                for metric in data.keys():
                    if metric != "Date":
                        data[metric][i] = np.nan
    elif data_source == "Matches Only":
        # Simulate matches on certain days
        mask = np.random.choice([True, False], size=days, p=[0.3, 0.7])
        for i, date in enumerate(dates):
            if not mask[i]:
                for metric in data.keys():
                    if metric != "Date":
                        data[metric][i] = np.nan
    
    return pd.DataFrame(data)

# Generate data
df = generate_performance_data(start_date, end_date, data_source)

# Filter by selected metrics
if selected_metrics:
    df = df[["Date"] + selected_metrics]

# Tabbed view for metrics
tabs = st.tabs(["📈 Trends", "📋 Daily Breakdown", "📥 Export Report"])

# --- Tab 1: Trends --- #
with tabs[0]:
    st.markdown("<h3 class='sub-header'>📈 Performance Trends Over Time</h3>", unsafe_allow_html=True)
    
    # Calculate number of rows and columns for subplots
    n_metrics = len(selected_metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + 1) // 2
    
    # Create figure with appropriate size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    fig.patch.set_facecolor('#1e1e1e')
    
    # Flatten axes for easier iteration
    if n_metrics > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]
    
    # Plot each metric
    for i, metric in enumerate(selected_metrics):
        ax = axes_flat[i]
        ax.set_facecolor('#263238')
        
        # Determine plot type based on metric
        if "Time" in metric:
            # For time metrics, lower is better
            sns.lineplot(data=df, x="Date", y=metric, marker='s', color='orange', ax=ax)
            ax.set_title(f"{metric} (Lower is Better)")
        elif "Mistakes" in metric:
            # For mistakes, lower is better
            sns.barplot(data=df, x="Date", y=metric, color='crimson', ax=ax)
            ax.set_title(f"{metric} (Lower is Better)")
        else:
            # For other metrics, higher is better
            sns.lineplot(data=df, x="Date", y=metric, marker='o', ax=ax)
            ax.set_title(f"{metric} (Higher is Better)")
        
        # Customize appearance
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Set text colors for dark theme
        ax.set_title(ax.get_title(), color='white')
        ax.set_xlabel(ax.get_xlabel(), color='white')
        ax.set_ylabel(ax.get_ylabel(), color='white')
        ax.tick_params(colors='white')
    
    # Hide empty subplots if any
    for i in range(len(selected_metrics), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot
    st.pyplot(fig)
    
    # Add insights
    st.markdown("### 💡 Insights")
    
    # Calculate insights based on data
    insights = []
    
    # Check for improvement in each metric
    for metric in selected_metrics:
        if metric in df.columns:
            # Remove NaN values
            valid_data = df[metric].dropna()
            if len(valid_data) >= 2:
                first_value = valid_data.iloc[0]
                last_value = valid_data.iloc[-1]
                
                if "Time" in metric or "Mistakes" in metric:
                    # For metrics where lower is better
                    if last_value < first_value:
                        improvement = ((first_value - last_value) / first_value) * 100
                        insights.append(f"✅ **{metric}** improved by {improvement:.1f}%")
                    elif last_value > first_value:
                        decline = ((last_value - first_value) / first_value) * 100
                        insights.append(f"⚠️ **{metric}** increased by {decline:.1f}%")
                else:
                    # For metrics where higher is better
                    if last_value > first_value:
                        improvement = ((last_value - first_value) / first_value) * 100
                        insights.append(f"✅ **{metric}** improved by {improvement:.1f}%")
                    elif last_value < first_value:
                        decline = ((first_value - last_value) / first_value) * 100
                        insights.append(f"⚠️ **{metric}** decreased by {decline:.1f}%")
    
    # Display insights
    if insights:
        for insight in insights:
            st.markdown(insight)
    else:
        st.markdown("No significant trends detected in the selected metrics.")

# --- Tab 2: Table View --- #
with tabs[1]:
    st.markdown("<h3 class='sub-header'>📋 Daily Performance Data</h3>", unsafe_allow_html=True)
    
    # Display the dataframe with styling
    st.dataframe(
        df.style
        .background_gradient(subset=selected_metrics, cmap='Blues')
        .format({col: '{:.1f}' for col in selected_metrics if col != "Date"})
        .format({"Date": lambda x: x.strftime('%Y-%m-%d')}),
        use_container_width=True
    )
    
    # Add summary statistics
    st.markdown("### 📊 Summary Statistics")
    
    # Calculate summary statistics
    summary = df[selected_metrics].describe()
    
    # Display summary statistics
    st.dataframe(
        summary.style
        .background_gradient(cmap='Blues')
        .format('{:.2f}'),
        use_container_width=True
    )

# --- Tab 3: Export --- #
with tabs[2]:
    st.markdown("<h3 class='sub-header'>📥 Export Your Training Report</h3>", unsafe_allow_html=True)
    
    # Function to convert dataframe to Excel
    def convert_df_to_excel(df):
        try:
            # Try to use xlsxwriter if available
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Performance')
                
                # Get workbook and worksheet objects
                workbook = writer.book
                worksheet = writer.sheets['Performance']
                
                # Add a chart sheet
                chart_sheet = workbook.add_worksheet('Charts')
                
                # Create charts for each metric
                for i, metric in enumerate(selected_metrics):
                    # Create a chart object
                    chart = workbook.add_chart({'type': 'line'})
                    
                    # Configure the chart
                    chart.add_series({
                        'name': metric,
                        'categories': ['Performance', 1, 0, len(df), 0],
                        'values': ['Performance', 1, i+1, len(df), i+1],
                    })
                    
                    # Set chart title and axis labels
                    chart.set_title({'name': metric})
                    chart.set_x_axis({'name': 'Date'})
                    chart.set_y_axis({'name': metric})
                    
                    # Insert the chart into the worksheet
                    chart_sheet.insert_chart(i*15, 0, chart)
                
                # Auto-adjust columns width
                for i, col in enumerate(df.columns):
                    max_length = max(df[col].astype(str).apply(len).max(), len(col)) + 2
                    worksheet.set_column(i, i, max_length)
            
            return buffer.getvalue()
        except ImportError:
            # Fallback to openpyxl if xlsxwriter is not available
            try:
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Performance')
                return buffer.getvalue()
            except ImportError:
                # If neither xlsxwriter nor openpyxl is available, return None
                return None
    
    # Function to convert dataframe to CSV
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    # Function to convert dataframe to PDF (simplified)
    def convert_df_to_pdf(df):
        # This is a simplified version - in a real app, you'd use a proper PDF library
        buffer = BytesIO()
        with open(buffer, 'w') as f:
            f.write("Performance Report\n\n")
            f.write(f"Date Range: {start_date} to {end_date}\n\n")
            f.write(df.to_string())
        return buffer.getvalue()
    
    # Export button
    if export_format == "Excel":
        excel_data = convert_df_to_excel(df)
        if excel_data is not None:
            b64 = base64.b64encode(excel_data).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="performance_report.xlsx">📩 Download Excel Report</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.error("Excel export is not available. Please install the required packages by running: `pip install xlsxwriter openpyxl`")
            st.markdown("You can still export your data in CSV format.")
    elif export_format == "CSV":
        csv_data = convert_df_to_csv(df)
        b64 = base64.b64encode(csv_data).decode()
        href = f'<a href="data:text/csv;base64,{b64}" download="performance_report.csv">📩 Download CSV Report</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:  # PDF
        pdf_data = convert_df_to_pdf(df)
        b64 = base64.b64encode(pdf_data).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="performance_report.pdf">📩 Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    # Add information about the report
    st.markdown("""
    <div class='info-box'>
        <h4>📊 About Your Report</h4>
        <p>Your performance report includes:</p>
        <ul>
            <li>Daily performance metrics for the selected date range</li>
            <li>Summary statistics for each metric</li>
            <li>Visual charts to help you track your progress</li>
        </ul>
        <p>Use this report to share your progress with coaches or to keep track of your improvement over time.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add installation instructions
    st.markdown("""
    <div class='info-box'>
        <h4>🔧 Installation Instructions</h4>
        <p>For full functionality, you may need to install additional packages:</p>
        <pre>pip install xlsxwriter openpyxl</pre>
        <p>These packages enable Excel export with charts and formatting.</p>
    </div>
    """, unsafe_allow_html=True)

# Add a section for recommendations
st.markdown("<h3 class='sub-header'>🎯 Personalized Recommendations</h3>", unsafe_allow_html=True)

# Generate recommendations based on the data
recommendations = []

# Check for areas of improvement
for metric in selected_metrics:
    if metric in df.columns:
        # Remove NaN values
        valid_data = df[metric].dropna()
        if len(valid_data) >= 2:
            last_value = valid_data.iloc[-1]
            
            if "Time" in metric and last_value > 0.35:
                recommendations.append("⏱️ **Reaction Time**: Focus on quick movement drills to improve your reaction time.")
            elif "Mistakes" in metric and last_value > 3:
                recommendations.append("❌ **Mistakes**: Practice consistency drills to reduce errors during play.")
            elif "Accuracy" in metric and last_value < 85:
                recommendations.append("🎯 **Accuracy**: Work on precision drills to improve your shot accuracy.")
            elif "Footwork" in metric and last_value < 80:
                recommendations.append("👣 **Footwork**: Incorporate more footwork drills to enhance your court movement.")
            elif "Serve" in metric and last_value < 75:
                recommendations.append("🎾 **Serve**: Dedicate time to serving practice to improve your serve quality.")
            elif "Smash" in metric and last_value < 70:
                recommendations.append("💥 **Smash**: Focus on power and technique drills to enhance your smash.")

# Display recommendations
if recommendations:
    for recommendation in recommendations:
        st.markdown(recommendation)
else:
    st.markdown("Great job! Your metrics are showing good performance across the board. Keep up the good work!")

# Add a button to start a new session
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🎮 Start New Training Session", type="primary"):
    st.switch_page("pages/1_Game_Mode_AI.py") 