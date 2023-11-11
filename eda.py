from turtle import color
from pyparsing import col
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64

# Function to load data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type.")
            return None
        return df
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to show dataframe
def show_dataframe(df):
    st.write(df)

# Function to create surface data
def create_surface_data(df):
    # This function would need to process 'df' to produce 'x', 'y', and 'z' for the surface plot
    # Here we just create a simple example with numpy
    x = np.outer(np.linspace(-10, 10, 30), np.ones(30))
    y = x.copy().T  # transpose
    z = np.cos(x ** 2 + y ** 2)
    return x, y, z

def get_image_as_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Main function where the app runs
def main():
    
    st.set_page_config(page_title="Data Analysis Application", page_icon="📊", layout="wide")
    st.title("Data Analysis Application")
    logo_base64 = get_image_as_base64("C:/Users/gmi/OneDrive - Hewlett Packard Enterprise/Pictures/HPE LOGO SUITE/hpe_logos/hpe logos PNG/hpe logos PNG/primary logo small png_files/hpesm_pri_grn_rev_rgb.png")
    st.sidebar.markdown(
        f'<img src="data:image/png;base64,{logo_base64}" alt="logo" width="200"><br><br>', unsafe_allow_html=True
    )
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file.", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            # Home Page Options
            st.sidebar.title("What would you like to do?")
            options = st.sidebar.radio("", ('EDA', 'Data Visualization'), label_visibility="collapsed")

            if options == 'EDA':
                # Display EDA options
                eda_option = st.sidebar.selectbox("Choose an EDA option:", 
                    ("Show dtypes", "Show columns", "Show summary", "Show missing values", 
                     "Show percentage of missing values", "Show number of unique values", 
                     "Show skewness and kurtosis", "Check for outliers"), label_visibility="collapsed")
                
                if eda_option == "Show dtypes":
                    st.write(df.dtypes)
                elif eda_option == "Show columns":
                    st.write(df.columns.tolist())
                elif eda_option == "Show summary":
                    st.write(df.describe())
                elif eda_option == "Show missing values":
                    st.write(df.isnull().sum())
                elif eda_option == "Show percentage of missing values":
                    st.write(df.isnull().mean() * 100)
                elif eda_option == "Show number of unique values":
                    st.write(df.nunique())
                elif eda_option == "Show skewness and kurtosis":
                    try:
                        st.write("Skewness:")
                        st.write(df.skew())
                        st.write("Kurtosis:")
                        st.write(df.kurtosis())
                    except Exception as e:
                        st.error(f"An error occurred when calculating skewness and kurtosis: {e}")
                elif eda_option == "Check for outliers":
                    # Select numeric columns, specify the data types explicitly
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                    selected_column = st.sidebar.selectbox("Select Column", numeric_cols, label_visibility="visible")
                    if st.button("Show Outliers for Selected Column"):
                        fig = px.box(df, y=selected_column)
                        st.plotly_chart(fig)
                        # Calculate Z-score and display outliers
                        z_scores = (df[selected_column] - df[selected_column].mean()) / df[selected_column].std()
                        st.write(df[abs(z_scores) > 3])
                
            elif options == 'Data Visualization':
                # Display Data Visualization options
                vis_option = st.sidebar.selectbox("Choose a plot type:", 
                    ("Univariate Plots", "Bivariate Plots", "Multivariate Plots"), label_visibility="visible")
                
                selected_color = st.sidebar.color_picker("Pick a color", "#01A982")

                # Universal plot settings
                if vis_option == "Univariate Plots":
                    column_to_plot = st.sidebar.selectbox("Choose a column to plot:", df.columns, label_visibility="visible")
                    plot_type = st.sidebar.selectbox("Choose plot type:", ("Bar", "Box", "Box Plot (enhanced)", "Histogram", 
                                                                           "Pie Chart", "Violin Plot", "Density Plot (KDE)", 
                                                                           "Area Chart", "Rug Plot", "Cumulative Distribution Function", 
                                                                           "Funnel Chart"), label_visibility="visible")
                    hue_column = None
                    if plot_type in ["Bar", "Box", "Box Plot (enhanced)", "Violin Plot", "Histogram", "Density Plot (KDE)", "Rug Plot", "Cumulative Distribution Function"]:
                        hue_options = [None] + list(df.select_dtypes(include=['object']).columns)
                        hue_column = st.sidebar.selectbox("Choose a categorical column for color coding hue:", hue_options, format_func=lambda x:'None' if x is None else x, label_visibility="visible")
                    if plot_type == "Bar":
                        fig = px.bar(df, y=column_to_plot, color=hue_column, color_discrete_sequence=[selected_color] if hue_column is None else None)
                        st.plotly_chart(fig)
                    elif plot_type == "Box":
                        fig = px.box(df, y=column_to_plot, color=hue_column, color_discrete_sequence=[selected_color])
                        st.plotly_chart(fig)
                    elif plot_type == "Box Plot (enhanced)":
                        fig = px.box(df, y=column_to_plot, points="all", color=hue_column, color_discrete_sequence=[selected_color] if hue_column is None else None)
                        st.plotly_chart(fig)
                    elif plot_type == "Histogram":
                        fig = px.histogram(df, x=column_to_plot, color=hue_column, color_discrete_sequence=[selected_color] if hue_column is None else None)
                        st.plotly_chart(fig)
                    elif plot_type == "Pie Chart":
                        fig = px.pie(df, names=column_to_plot, color_discrete_sequence=[selected_color])
                        st.plotly_chart(fig)
                    elif plot_type == "Violin Plot":
                        fig = px.violin(df, y=column_to_plot, color=hue_column, color_discrete_sequence=[selected_color] if hue_column is None else None)
                        st.plotly_chart(fig)
                    elif plot_type == "Density Plot (KDE)":  # KDE = Kernel Density Estimation
                        fig = px.density_contour(df, x=column_to_plot, marginal_x="histogram", color=hue_column, color_discrete_sequence=[selected_color] if hue_column is None else None)
                        st.plotly_chart(fig)
                    elif plot_type == "Area Chart":
                        fig = px.area(df, y=column_to_plot, color_discrete_sequence=[selected_color])
                        st.plotly_chart(fig)
                    elif plot_type == "Rug Plot":
                        fig = px.density_contour(df, x=column_to_plot, marginal_x="rug", color=hue_column, color_discrete_sequence=[selected_color] if hue_column is None else None)
                        st.plotly_chart(fig)
                    elif plot_type == "Cumulative Distribution Function":
                        fig = px.histogram(df, x=column_to_plot, cumulative=True, color=hue_column, color_discrete_sequence=[selected_color] if hue_column is None else None)
                        st.plotly_chart(fig)
                    elif plot_type == "Funnel Chart":
                        fig = px.funnel(df, y=column_to_plot, color_discrete_sequence=[selected_color] if hue_column is None else None)
                        st.plotly_chart(fig)

                elif vis_option == "Bivariate Plots":

                    # Select plot type
                    bivariate_plot_type = st.sidebar.selectbox(
                        "Choose bivariate plot type:", 
                        ("Scatter Plot", "Line Plot", "Bubble Chart", "Area Chart", 
                         "Joint Plot", "Stacked Bar Chart", "Grouped Bar Chart", 
                         "Contour Plot", "Box Plot", "Error Bars Plot", "Violin Plot", 
                         ), 
                        label_visibility="visible"
                    )
                    # User selects two columns for Bivariate plots
                    col1 = st.sidebar.selectbox("Choose the first column:", df.columns, label_visibility="visible")
                    col2 = st.sidebar.selectbox("Choose the second column:", df.columns, label_visibility="visible")
                    hue_column = None
                    if bivariate_plot_type in ["Scatter Plot", "Line Plot", "Bubble Chart", "Area Chart", "Joint Plot", "Stacked Bar Chart", "Grouped Bar Chart", "Contour Plot", "Box Plot", "Error Bars Plot", "Violin Plot"]:
                        hue_options = [None] + list(df.select_dtypes(include=['object']).columns)
                        hue_column = st.sidebar.selectbox(
                            "Choose a categorical column for color coding (hue):", 
                            hue_options, 
                            format_func=lambda x:'None' if x is None else x, 
                            label_visibility="visible"
                        )
                    
                    if bivariate_plot_type == "Scatter Plot":
                        fig = px.scatter(df, x=col1, y=col2, color=hue_column, trendline="ols", trendline_color_override="yellow",color_discrete_sequence=[selected_color] if hue_column is None else None)
                        st.plotly_chart(fig)
                    elif bivariate_plot_type == "Line Plot":
                        fig = px.line(df, x=col1, y=col2, color=hue_column, color_discrete_sequence=[selected_color] if hue_column is None else None)
                        st.plotly_chart(fig)
                    elif bivariate_plot_type == "Bubble Chart":
                        size_column = st.sidebar.selectbox("Choose a column for bubble size:", df.columns, label_visibility="visible")
                        fig = px.scatter(df, x=col1, y=col2, size=size_column, color=hue_column, color_discrete_sequence=[selected_color] if hue_column is None else None)
                        st.plotly_chart(fig)
                    elif bivariate_plot_type == "Area Chart":
                        fig = px.area(df, x=col1, y=col2, color=hue_column, color_discrete_sequence=[selected_color] if hue_column is None else None)
                        st.plotly_chart(fig)
                    elif bivariate_plot_type == "Joint Plot":
                        fig = px.scatter(df, x=col1, y=col2, color=hue_column, marginal_x="histogram", marginal_y="histogram", trendline="ols", trendline_color_override="yellow",color_discrete_sequence=[selected_color] if hue_column is None else None)
                        st.plotly_chart(fig)
                    elif bivariate_plot_type == "Stacked Bar Chart":
                        fig = px.bar(df, x=col1, y=col2, color=hue_column, color_discrete_sequence=[selected_color] if hue_column is None else None)
                        st.plotly_chart(fig)
                    elif bivariate_plot_type == "Grouped Bar Chart":
                        fig = px.bar(df, x=col1, y=col2, color=hue_column, barmode="group", color_discrete_sequence=[selected_color] if hue_column is None else None)
                        st.plotly_chart(fig)
                    elif bivariate_plot_type == "Contour Plot":
                        fig = px.density_contour(df, x=col1, y=col2, color=hue_column, color_discrete_sequence=[selected_color] if hue_column is None else None)
                        st.plotly_chart(fig)
                    elif bivariate_plot_type == "Box Plot":
                        fig = px.box(df, x=col1, y=col2, color=hue_column, color_discrete_sequence=[selected_color] if hue_column is None else None)
                        st.plotly_chart(fig)
                    elif bivariate_plot_type == "Error Bars Plot":
                        fig = px.scatter(df, x=col1, y=col2, color=hue_column, error_x=col1, error_y=col2, color_discrete_sequence=[selected_color] if hue_column is None else None)
                        st.plotly_chart(fig)
                    elif bivariate_plot_type == "Violin Plot":
                        fig = px.violin(df, x=col1, y=col2, color=hue_column, color_discrete_sequence=[selected_color] if hue_column is None else None)
                        st.plotly_chart(fig)


                elif vis_option == "Multivariate Plots":
    # Ensure `selected_columns` is a list, not a Pandas Index or Series.
                    selected_columns = st.sidebar.multiselect("Choose columns for multivariate plot:",
                                                            options=df.columns.tolist(),
                                                            default=df.columns[:3].tolist(),
                                                            label_visibility="visible")

                    # Check if the selection is not empty.
                    if selected_columns:  # This checks if the list is not empty
                        # Now we can proceed to select the plot type.
                        multivariate_plot_type = st.sidebar.selectbox(
                            "Choose multivariate plot type:",
                            ("3D Scatter Plot", "Parallel Coordinates", "Ternary Plot", "3D Surface Plot"),
                            label_visibility="visible"
                        )

                        if multivariate_plot_type == "3D Scatter Plot":
                            # Ensure that three distinct columns have been chosen.
                            if len(selected_columns) >= 3:
                                col1, col2, col3 = selected_columns[:3]  # Take the first three selections
                                fig = px.scatter_3d(df, x=col1, y=col2, z=col3, color=col1)
                                st.plotly_chart(fig)
                            else:
                                st.error("Please select at least three columns for the 3D Scatter Plot.")
                        
                        elif multivariate_plot_type == "Parallel Coordinates":
                            fig = px.parallel_coordinates(df, color=selected_columns[0])
                            st.plotly_chart(fig)
                        
                        elif multivariate_plot_type == "Ternary Plot":
                            # Ensure that three distinct columns have been chosen.
                            if len(selected_columns) >= 3:
                                col1, col2, col3 = selected_columns[:3]
                                fig = px.scatter_ternary(df, a=col1, b=col2, c=col3, color=col1)
                                st.plotly_chart(fig)
                            else:
                                st.error("Please select at least three columns for the Ternary Plot.")
                        
                        elif multivariate_plot_type == "3D Surface Plot":
                            # Ensure that three distinct columns have been chosen.
                            if len(selected_columns) >= 3:
                                col1, col2, col3 = selected_columns[:3]
                                x, y, z = create_surface_data(df)
                                fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])
                                st.plotly_chart(fig)
                            else:
                                st.error("Please select at least three columns for the 3D Surface Plot.")

                    else:
                        st.warning("Please select at least one column to create a plot.")



if __name__ == "__main__":
    main()