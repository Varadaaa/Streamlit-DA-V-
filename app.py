import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(page_title="Bike Sharing EDA & Visualization Dashboard", layout="wide")

# -------------------------------------------------
# Load data (same preprocessing as Assignments 1 & 2)
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['hour'] = df['datetime'].dt.hour
    return df

df = load_data()

# -------------------------------------------------
# Title & description (matches written answers)
# -------------------------------------------------
st.title("🚲 Bike Sharing Dataset: EDA & Visualization")
st.markdown(
    "This dashboard is a visual summary of **Assignment 1 (Exploratory Data Analysis)** and "
    "**Assignment 2 (Data Visualization using Matplotlib and Seaborn)** performed on the Bike Sharing dataset."
)

# -------------------------------------------------
# Sidebar filters (interactive widgets)
# -------------------------------------------------
st.sidebar.header("Data Filters")

season_filter = st.sidebar.multiselect(
    "Season",
    options=sorted(df['season'].unique()),
    default=sorted(df['season'].unique())
)

weather_filter = st.sidebar.multiselect(
    "Weather",
    options=sorted(df['weather'].unique()),
    default=sorted(df['weather'].unique())
)

hour_range = st.sidebar.slider(
    "Hour of Day",
    min_value=0,
    max_value=23,
    value=(0, 23)
)

filtered_df = df[
    (df['season'].isin(season_filter)) &
    (df['weather'].isin(weather_filter)) &
    (df['hour'] >= hour_range[0]) &
    (df['hour'] <= hour_range[1])
]

# -------------------------------------------------
# Dataset overview (Assignment 1 content)
# -------------------------------------------------
st.subheader("Dataset Overview")
col1, col2, col3 = st.columns(3)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", int(df.isnull().sum().sum()))

st.markdown("The dataset contains no missing values, confirming it is clean and ready for analysis.")

# -------------------------------------------------
# Visualizations (Assignments 1 & 2 combined)
# -------------------------------------------------
st.subheader("Exploratory & Visual Analysis")

# Plot 1: Distribution of total rentals
fig1, ax1 = plt.subplots()
sns.histplot(filtered_df['count'], bins=30, kde=True, ax=ax1)
ax1.set_title("Distribution of Total Bike Rentals")
st.pyplot(fig1)

# Plot 2: Average rentals by hour
fig2, ax2 = plt.subplots()
sns.lineplot(data=filtered_df, x='hour', y='count', ax=ax2)
ax2.set_title("Average Bike Rentals by Hour of Day")
st.pyplot(fig2)

# Plot 3: Rentals by season
fig3, ax3 = plt.subplots()
sns.barplot(data=filtered_df, x='season', y='count', estimator=np.mean, ax=ax3)
ax3.set_title("Average Bike Rentals by Season")
st.pyplot(fig3)

# Plot 4: Weather vs rentals
fig4, ax4 = plt.subplots()
sns.boxplot(data=filtered_df, x='weather', y='count', ax=ax4)
ax4.set_title("Effect of Weather Conditions on Bike Rentals")
st.pyplot(fig4)

# Plot 5: Correlation heatmap (Assignment 1)
num_cols = ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']
corr_matrix = filtered_df[num_cols].corr()

fig5, ax5 = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', ax=ax5)
ax5.set_title("Correlation Heatmap of Numerical Features")
st.pyplot(fig5)

# -------------------------------------------------
# Data preview
# -------------------------------------------------
st.subheader("Filtered Data Preview")
st.dataframe(filtered_df.head(15))

# -------------------------------------------------
# Conclusion (matches assignment conclusions)
# -------------------------------------------------
st.markdown("---")
st.markdown(
    "**Conclusion:** The analysis shows that bike demand is strongly influenced by time-related features such as hour and season, "
    "as well as weather conditions and temperature. The correlation analysis confirms a strong positive relationship between "
    "registered users and total bike rentals."
)

st.markdown("Dashboard created using Streamlit as a consolidated visualization of Assignments 1 and 2.")
