import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io
from io import BytesIO
from datetime import datetime

# Machine learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Metrics
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                            mean_squared_error, r2_score, mean_absolute_error, 
                            precision_score, recall_score, f1_score)

# Set up the app
st.set_page_config(page_title="Data Science Workbench", layout="wide")
st.title("🧪 Data Science Workbench")
st.write("An all-in-one application for data analysis and machine learning")

# Initialize session state for storing data
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'user_observations' not in st.session_state:
    st.session_state.user_observations = {}
if 'cleaning_methods' not in st.session_state:
    st.session_state.cleaning_methods = {}
if 'encoding_methods' not in st.session_state:
    st.session_state.encoding_methods = {}
if 'column_scaling_methods' not in st.session_state:
    st.session_state.column_scaling_methods = {}
if 'type_casting_methods' not in st.session_state:
    st.session_state.type_casting_methods = {}
if 'outlier_methods' not in st.session_state:
    st.session_state.outlier_methods = {}

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Select Task", 
                               ["Load Data", "Data Overview", "Data Cleaning", 
                                "Data Visualization", "Feature Engineering", 
                                "Model Training", "Results"])

# Function to load data
def load_data():
    st.header("📂 Load Dataset")
    data_source = st.radio("Select data source", 
                          ["Upload file", "Use sample dataset"])
    
    if data_source == "Upload file":
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'txt'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file, delimiter='\t')
                
                st.session_state.df = df
                st.session_state.processed_df = df.copy()
                st.success("Data loaded successfully!")
                
                if st.checkbox("Show first 5 rows"):
                    st.dataframe(df.head())
                    
            except Exception as e:
                st.error(f"Error loading file: {e}")
                
    else:
        sample_dataset = st.selectbox("Select sample dataset", 
                                    ["Iris", "Titanic", "Diabetes", "Wine Quality"])
        
        if sample_dataset == "Iris":
            df = sns.load_dataset('iris')
        elif sample_dataset == "Titanic":
            df = sns.load_dataset('titanic')
        elif sample_dataset == "Diabetes":
            from sklearn.datasets import load_diabetes
            data = load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
        else:  # Wine Quality
            df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
        
        st.session_state.df = df
        st.session_state.processed_df = df.copy()
        st.success(f"{sample_dataset} dataset loaded successfully!")
        st.dataframe(df.head())

# Function for data overview
def data_overview():
    st.header("🔍 Data Overview")
    
    if st.session_state.df is None:
        st.warning("Please load data first!")
        return
    
    df = st.session_state.df
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Shape")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")
        
        st.subheader("Column Information")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        # Display the selected column value_counts
        st.subheader("Values count for selected column")
        selected_column = st.selectbox("Select a column", df.columns, key="overview_col_select")
        if selected_column:
            st.write(f"Values count for {selected_column}")
            st.write(df[selected_column].value_counts())
    
    with col2:
        st.subheader("Statistical Summary - Numerical")
        st.dataframe(df.describe())
        
        # Statistical summary for categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            st.subheader("Statistical Summary - Categorical")
            st.dataframe(df[categorical_cols].describe())
        
        st.subheader("Data Types")
        st.write(df.dtypes)
    
    # Show missing values
    st.subheader("Missing Values")
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Values': df.isnull().sum(),
        'Percentage%': ((df.isnull().sum() / len(df)) * 100).round(2)
    })
    st.dataframe(missing_df)
    
    # Show duplicates - FIXED: Use proper duplicate detection
    st.subheader("Duplicate Rows")
    # Check for complete row duplicates (standard method)
    duplicate_count = df.duplicated().sum()
    
    st.write(f"Number of duplicate rows: {duplicate_count}")
    
    if duplicate_count > 0:
        st.subheader("Sample Duplicate Rows")
        # Show all duplicates including the first occurrence
        duplicate_mask = df.duplicated(keep=False)
        st.dataframe(df[duplicate_mask].head(5))

    # User observations
    st.subheader("Your Observations")
    observations_key = "overview_observations"
    if observations_key not in st.session_state.user_observations:
        st.session_state.user_observations[observations_key] = ""
    
    st.session_state.user_observations[observations_key] = st.text_area(
        "Add your observations about the dataset:",
        value=st.session_state.user_observations[observations_key],
        height=100,
        key="overview_obs"
    )

# Function for data cleaning
def data_cleaning():
    st.header("🧹 Data Cleaning")
    
    if st.session_state.df is None:
        st.warning("Please load data first!")
        return
    
    # Use processed_df if available, otherwise use original df
    if st.session_state.processed_df is not None:
        df = st.session_state.processed_df.copy()
    else:
        df = st.session_state.df.copy()
    
    # Handle missing values
    st.subheader("Handle Missing Values")
    missing_cols = df.columns[df.isnull().any()].tolist()
    
    if missing_cols:
        st.write("Columns with missing values:")
        for col in missing_cols:
            missing_percent = df[col].isnull().sum() / len(df) * 100
            st.write(f"- {col}: {df[col].isnull().sum()} missing values ({missing_percent:.2f}%)")
            
        for col in missing_cols:
            st.write(f"#### Handling missing values for {col}")
            col_type = df[col].dtype
            
            # Initialize cleaning method for this column if not exists
            if col not in st.session_state.cleaning_methods:
                st.session_state.cleaning_methods[col] = None
            
            if col_type in ['int64', 'float64']:
                method = st.selectbox(
                    f"Method for {col}",
                    ["Select method", "Drop rows", "Fill with mean", "Fill with median", "Fill with mode", 
                     "Forward fill (ffill)", "Backward fill (bfill)", "KNN Imputer", "Iterative Imputer"],
                    key=f"num_{col}",
                    index=0 if st.session_state.cleaning_methods[col] is None else [
                        "Select method", "Drop rows", "Fill with mean", "Fill with median", "Fill with mode", 
                        "Forward fill (ffill)", "Backward fill (bfill)", "KNN Imputer", "Iterative Imputer"
                    ].index(st.session_state.cleaning_methods[col])
                )
                
                if method != "Select method":
                    st.session_state.cleaning_methods[col] = method
                    
                    if method == "Drop rows":
                        df.dropna(subset=[col], inplace=True)
                    elif method == "Fill with mean":
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif method == "Fill with median":
                        df[col].fillna(df[col].median(), inplace=True)
                    elif method == "Fill with mode":
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    elif method == "Forward fill (ffill)":
                        df[col].fillna(method='ffill', inplace=True)
                    elif method == "Backward fill (bfill)":
                        df[col].fillna(method='bfill', inplace=True)
                    elif method == "KNN Imputer":
                        imputer = KNNImputer(n_neighbors=5)
                        df[col] = imputer.fit_transform(df[[col]]).ravel()
                    elif method == "Iterative Imputer":
                        imputer = IterativeImputer(random_state=42)
                        df[col] = imputer.fit_transform(df[[col]]).ravel()
                    
            else:  # Categorical columns
                method = st.selectbox(
                    f"Method for {col}",
                    ["Select method", "Drop rows", "Fill with mode", "Forward fill (ffill)", "Backward fill (bfill)"],
                    key=f"cat_{col}",
                    index=0 if st.session_state.cleaning_methods[col] is None else [
                        "Select method", "Drop rows", "Fill with mode", "Forward fill (ffill)", "Backward fill (bfill)"
                    ].index(st.session_state.cleaning_methods[col])
                )
                
                if method != "Select method":
                    st.session_state.cleaning_methods[col] = method
                    
                    if method == "Drop rows":
                        df.dropna(subset=[col], inplace=True)
                    elif method == "Fill with mode":
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    elif method == "Forward fill (ffill)":
                        df[col].fillna(method='ffill', inplace=True)
                    elif method == "Backward fill (bfill)":
                        df[col].fillna(method='bfill', inplace=True)
    else:
        st.success("No missing values found!")
    
    # Handle duplicates - FIXED: Use proper duplicate detection
    st.subheader("Handle Duplicates")
    duplicate_count = df.duplicated().sum()
    
    if duplicate_count > 0:
        remove_duplicates = st.checkbox("Remove duplicate rows", key="remove_duplicates")
        if remove_duplicates:
            df.drop_duplicates(inplace=True)
            st.write(f"Removed {duplicate_count} duplicate rows")
    else:
        st.info("No duplicate rows found")
    
    # Type casting functionality - IMPROVED VERSION
    st.subheader("Type Casting")
    st.write("Change data types of columns:")
    
    # Let user select a column to modify
    col_to_cast = st.selectbox("Select column to change type", df.columns, key="col_to_cast")
    
    if col_to_cast:
        current_type = str(df[col_to_cast].dtype)
        st.write(f"Current type of '{col_to_cast}': {current_type}")
        
        # Initialize type casting method for this column if not exists
        if col_to_cast not in st.session_state.type_casting_methods:
            st.session_state.type_casting_methods[col_to_cast] = current_type
        
        # Determine allowed conversions based on current type
        if current_type.startswith('int') or current_type.startswith('float'):
            allowed_types = ["Keep current", "int64", "float64", "str", "category"]
        elif current_type == 'object':
            # Check if column contains date-like strings
            sample_value = str(df[col_to_cast].dropna().iloc[0]) if not df[col_to_cast].dropna().empty else ""
            is_date_like = False
            try:
                pd.to_datetime(sample_value)
                is_date_like = True
            except:
                pass
                
            if is_date_like:
                allowed_types = ["Keep current", "datetime64[ns]", "str", "category"]
            else:
                allowed_types = ["Keep current", "str", "category"]
        elif current_type.startswith('datetime'):
            allowed_types = ["Keep current", "str", "category"]
        elif current_type == 'category':
            allowed_types = ["Keep current", "str"]
        else:
            allowed_types = ["Keep current", "str", "category"]
        
        new_type = st.selectbox(
            f"Select new type for '{col_to_cast}'",
            allowed_types,
            key=f"type_{col_to_cast}",
            index=allowed_types.index(st.session_state.type_casting_methods[col_to_cast]) 
            if st.session_state.type_casting_methods[col_to_cast] in allowed_types 
            else 0
        )
        
        if st.button("Apply Type Change", key=f"apply_type_{col_to_cast}"):
            if new_type != "Keep current":
                try:
                    if new_type == "int64":
                        df[col_to_cast] = pd.to_numeric(df[col_to_cast], errors='coerce').astype('Int64')
                    elif new_type == "float64":
                        df[col_to_cast] = pd.to_numeric(df[col_to_cast], errors='coerce').astype('float64')
                    elif new_type == "str":
                        df[col_to_cast] = df[col_to_cast].astype(str)
                    elif new_type == "datetime64[ns]":
                        df[col_to_cast] = pd.to_datetime(df[col_to_cast], errors='coerce')
                    elif new_type == "category":
                        df[col_to_cast] = df[col_to_cast].astype('category')
                    
                    st.session_state.type_casting_methods[col_to_cast] = new_type
                    st.success(f"Changed '{col_to_cast}' from {current_type} to {new_type}")
                    
                    # Update the current type display
                    current_type = str(df[col_to_cast].dtype)
                    
                except Exception as e:
                    st.error(f"Error converting '{col_to_cast}' to {new_type}: {e}")
            else:
                st.info("No change applied")
    
    # Outlier handling functionality
    st.subheader("Outlier Handling")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        st.write("Handle outliers in numeric columns:")
        
        for col in numeric_cols:
            # Skip if column has too few unique values
            if df[col].nunique() < 2:
                continue
                
            st.write(f"#### Outlier handling for {col}")
            
            # Initialize outlier method for this column if not exists
            if col not in st.session_state.outlier_methods:
                st.session_state.outlier_methods[col] = "No action"
                
            # Show summary statistics for the column
            col_stats = df[col].describe()
            st.write(f"Summary: min={col_stats['min']:.2f}, max={col_stats['max']:.2f}, mean={col_stats['mean']:.2f}")
            
            # Detect potential outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                st.write(f"Potential outliers detected: {outlier_count} (using IQR method)")
                
                action = st.selectbox(
                    f"Action for outliers in {col}",
                    ["No action", "Remove outliers", "Cap outliers"],
                    key=f"outlier_{col}",
                    index=0 if st.session_state.outlier_methods[col] == "No action" else [
                        "No action", "Remove outliers", "Cap outliers"
                    ].index(st.session_state.outlier_methods[col])
                )
                
                if action != "No action":
                    st.session_state.outlier_methods[col] = action
                    
                    if action == "Remove outliers":
                        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                        st.write(f"Removed {outlier_count} outliers from {col}")
                    elif action == "Cap outliers":
                        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
                        st.write(f"Capped outliers in {col} to bounds [{lower_bound:.2f}, {upper_bound:.2f}]")
            else:
                st.info("No potential outliers detected using IQR method")
    else:
        st.info("No numeric columns found for outlier detection")
    
    # Column selection
    st.subheader("Column Selection")
    cols_to_keep = st.multiselect("Select columns to keep", df.columns.tolist(), default=df.columns.tolist(), key="cols_to_keep")
    df = df[cols_to_keep]
    
    st.session_state.processed_df = df
    
    st.subheader("Cleaned Data Preview")
    st.dataframe(df.head())
    
    # User observations
    st.subheader("Your Observations")
    observations_key = "cleaning_observations"
    if observations_key not in st.session_state.user_observations:
        st.session_state.user_observations[observations_key] = ""
    
    st.session_state.user_observations[observations_key] = st.text_area(
        "Add your observations about the data cleaning process:",
        value=st.session_state.user_observations[observations_key],
        height=100,
        key="cleaning_obs"
    )
    
    st.success("Data cleaning completed!")

# Function for data visualization
def data_visualization():
    st.header("📊 Data Visualization")
    
    if st.session_state.processed_df is None:
        st.warning("Please clean the data first!")
        return
    
    df = st.session_state.processed_df
    
    st.subheader("Select Visualization Type")
    viz_type = st.selectbox("Choose visualization type", 
                           ["Histogram", "Box Plot", "Scatter Plot", "Correlation Heatmap", 
                            "Bar Chart", "Pie Chart", "Line Chart"],
                           key="viz_type")
    
    if viz_type == "Histogram":
        col = st.selectbox("Select column", df.select_dtypes(include=np.number).columns.tolist(), key="hist_col")
        bins = st.slider("Number of bins", 5, 100, 20, key="hist_bins")
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=bins, edgecolor='black')
        ax.set_title(f"Histogram of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        
    elif viz_type == "Box Plot":
        col = st.selectbox("Select column", df.select_dtypes(include=np.number).columns.tolist(), key="box_col")
        fig, ax = plt.subplots()
        ax.boxplot(df[col].dropna())
        ax.set_title(f"Box Plot of {col}")
        ax.set_ylabel(col)
        st.pyplot(fig)
        
    elif viz_type == "Scatter Plot":
        col1 = st.selectbox("Select X axis", df.select_dtypes(include=np.number).columns.tolist(), key="scatter_x")
        col2 = st.selectbox("Select Y axis", df.select_dtypes(include=np.number).columns.tolist(), key="scatter_y")
        fig, ax = plt.subplots()
        ax.scatter(df[col1], df[col2], alpha=0.5)
        ax.set_title(f"Scatter Plot: {col1} vs {col2}")
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        st.pyplot(fig)
        
    elif viz_type == "Correlation Heatmap":
        numeric_df = df.select_dtypes(include=np.number)
        if len(numeric_df.columns) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
        else:
            st.warning("Need at least 2 numeric columns for correlation heatmap")
            
    elif viz_type == "Bar Chart":
        col = st.selectbox("Select column", df.select_dtypes(include=['object', 'category']).columns.tolist(), key="bar_col")
        value_counts = df[col].value_counts()
        fig, ax = plt.subplots()
        value_counts.plot(kind='bar', ax=ax)
        ax.set_title(f"Bar Chart of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    elif viz_type == "Pie Chart":
        col = st.selectbox("Select column", df.select_dtypes(include=['object', 'category']).columns.tolist(), key="pie_col")
        value_counts = df[col].value_counts()
        fig, ax = plt.subplots()
        ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
        ax.set_title(f"Pie Chart of {col}")
        st.pyplot(fig)
        
    elif viz_type == "Line Chart":
        col = st.selectbox("Select column", df.select_dtypes(include=np.number).columns.tolist(), key="line_col")
        if df.index.dtype in [np.int64, np.float64] or df.index.dtype.name.startswith('datetime'):
            x_values = df.index
        else:
            x_values = range(len(df))
            
        fig, ax = plt.subplots()
        ax.plot(x_values, df[col])
        ax.set_title(f"Line Chart of {col}")
        ax.set_xlabel("Index")
        ax.set_ylabel(col)
        st.pyplot(fig)

# Function for feature engineering
def feature_engineering():
    st.header("⚙️ Feature Engineering")
    
    if st.session_state.processed_df is None:
        st.warning("Please clean the data first!")
        return
    
    df = st.session_state.processed_df.copy()
    
    st.subheader("Current Data")
    st.dataframe(df.head())
    
    # Encode categorical variables
    st.subheader("Encode Categorical Variables")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
        for col in categorical_cols:
            st.write(f"Encoding {col}")
            
            # Initialize encoding method for this column if not exists
            if col not in st.session_state.encoding_methods:
                st.session_state.encoding_methods[col] = "No Encoding"
            
            encoding_method = st.radio(
                f"Encoding method for {col}",
                ["Label Encoding", "One-Hot Encoding", "Ordinal Encoding", "No Encoding"],
                key=f"encode_{col}",
                index=["Label Encoding", "One-Hot Encoding", "Ordinal Encoding", "No Encoding"].index(st.session_state.encoding_methods[col])
            )
            
            st.session_state.encoding_methods[col] = encoding_method
            
            if encoding_method != "No Encoding":
                if encoding_method == "Label Encoding":
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                elif encoding_method == "One-Hot Encoding":
                    df = pd.get_dummies(df, columns=[col], prefix=col)
                else:  # Ordinal Encoding
                    oe = OrdinalEncoder()
                    df[col] = oe.fit_transform(df[[col]])
                
        st.write("After encoding:")
        st.dataframe(df.head())
    else:
        st.info("No categorical columns found")
    
    # Feature scaling
    st.subheader("Feature Scaling")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        # Initialize scaling methods for each column if not exists
        if 'column_scaling_methods' not in st.session_state:
            st.session_state.column_scaling_methods = {}
        
        # Update the scaling methods dictionary with current numeric columns
        for col in numeric_cols:
            if col not in st.session_state.column_scaling_methods:
                st.session_state.column_scaling_methods[col] = "No Scaling"
        
        # Remove columns that no longer exist
        cols_to_remove = []
        for col in st.session_state.column_scaling_methods:
            if col not in numeric_cols:
                cols_to_remove.append(col)
        
        for col in cols_to_remove:
            del st.session_state.column_scaling_methods[col]
        
        st.write("Select scaling method for each numeric column:")
        
        # Create columns for better layout
        cols_per_row = 3
        cols = st.columns(cols_per_row)
        
        for i, col in enumerate(numeric_cols):
            with cols[i % cols_per_row]:
                st.markdown(f"**{col}**")
                
                scaling_method = st.selectbox(
                    f"Scaling method for {col}",
                    ["No Scaling", "Standard Scaler", "Min-Max Scaler", "Robust Scaler"],
                    key=f"scale_{col}",
                    index=["No Scaling", "Standard Scaler", "Min-Max Scaler", "Robust Scaler"].index(
                        st.session_state.column_scaling_methods[col]
                    )
                )
                
                st.session_state.column_scaling_methods[col] = scaling_method
        
        # Apply scaling button
        if st.button("Apply Scaling", key="apply_scaling_button"):
            # Apply scaling to each column based on selection
            scalers = {
                "Standard Scaler": StandardScaler(),
                "Min-Max Scaler": MinMaxScaler(),
                "Robust Scaler": RobustScaler()
            }
            
            scaling_applied = False
            scaled_columns_info = []
            scaled_column_names = []
            
            for col in numeric_cols:
                method = st.session_state.column_scaling_methods[col]
                if method != "No Scaling":
                    scaler = scalers[method]
                    df[col] = scaler.fit_transform(df[[col]])
                    scaling_applied = True
                    scaled_columns_info.append((col, method))
                    scaled_column_names.append(col)
            
            if scaling_applied:
                st.subheader("Scaling Applied")
                for col, method in scaled_columns_info:
                    st.write(f"- {col}: {method}")
                st.write("Data after scaling:")
                if scaled_column_names:
                    st.dataframe(df[scaled_column_names].head())
                else:
                    st.dataframe(df.head())
            else:
                st.info("No scaling applied to any column")
            
            # Update the processed dataframe
            st.session_state.processed_df = df
            st.success("Scaling applied successfully!")
        else:
            st.info("Click 'Apply Scaling' to apply your selected scaling methods")
    
    st.session_state.processed_df = df

# Function for model training
def model_training():
    st.header("🤖 Model Training")
    
    if st.session_state.processed_df is None:
        st.warning("Please process the data first!")
        return
    
    df = st.session_state.processed_df.copy()
    
    st.subheader("Select Target Variable")
    target = st.selectbox("Choose the target variable", df.columns.tolist(), key="target_var")
    
    # Determine problem type
    if df[target].dtype in ['int64', 'float64']:
        if len(df[target].unique()) < 20:  # Arbitrary threshold for classification
            problem_type = st.radio("Problem type", ["Classification", "Regression"], key="problem_type")
        else:
            problem_type = "Regression"
    else:
        problem_type = "Classification"
    
    st.write(f"Detected problem type: {problem_type}")
    
    # Select features
    features = [col for col in df.columns if col != target]
    selected_features = st.multiselect("Select features", features, default=features, key="selected_features")
    
    X = df[selected_features]
    y = df[target]
    
    # Train-test split
    test_size = st.slider("Test set size", 0.1, 0.5, 0.2, key="test_size")
    random_state = st.number_input("Random state", 0, 100, 42, key="random_state")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Model selection
    st.subheader("Model Selection")
    
    if problem_type == "Classification":
        model_options = {
            "Logistic Regression": LogisticRegression,
            "Support Vector Classifier (SVC)": SVC,
            "K-Nearest Neighbors": KNeighborsClassifier,
            "Decision Tree": DecisionTreeClassifier,
            "Random Forest": RandomForestClassifier,
            "Gradient Boosting": GradientBoostingClassifier,
            "AdaBoost": AdaBoostClassifier
        }
    else:  # Regression
        model_options = {
            "Linear Regression": LinearRegression,
            "Support Vector Regressor (SVR)": SVR,
            "K-Nearest Neighbors": KNeighborsRegressor,
            "Decision Tree": DecisionTreeRegressor,
            "Random Forest": RandomForestRegressor,
            "Gradient Boosting": GradientBoostingRegressor
        }
    
    model_choice = st.selectbox("Choose a model", list(model_options.keys()), key="model_choice")
    
    # Create model instance with default parameters
    model_class = model_options[model_choice]
    model = model_class()
    
    # Train model
    if st.button("Train Model", key="train_model_button"):
        with st.spinner("Training model..."):
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Display results
                st.subheader("Model Performance")
                
                if problem_type == "Classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                    with col2:
                        st.metric("Precision", f"{precision:.4f}")
                    with col3:
                        st.metric("Recall", f"{recall:.4f}")
                    with col4:
                        st.metric("F1 Score", f"{f1:.4f}")
                    
                    # Confusion matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
                    
                    # Classification report
                    st.subheader("Classification Report")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())
                    
                else:  # Regression
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Squared Error", f"{mse:.4f}")
                    with col2:
                        st.metric("Mean Absolute Error", f"{mae:.4f}")
                    with col3:
                        st.metric("R² Score", f"{r2:.4f}")
                    
                    # Plot actual vs predicted
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, y_pred, alpha=0.5)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    ax.set_title("Actual vs Predicted Values")
                    st.pyplot(fig)
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame({
                        'feature': selected_features,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig, ax = plt.subplots()
                    ax.barh(importance_df['feature'], importance_df['importance'])
                    ax.set_xlabel("Importance")
                    ax.set_title("Feature Importance")
                    st.pyplot(fig)
                    
                    st.dataframe(importance_df)
                
                # Save model to session state
                st.session_state.trained_model = model
                st.session_state.model_type = problem_type
                
                # Download model
                st.subheader("Download Trained Model")
                model_bytes = pickle.dumps(model)
                st.download_button(
                    label="Download Model",
                    data=model_bytes,
                    file_name=f"{model_choice.replace(' ', '_')}.pkl",
                    mime="application/octet-stream",
                    key="download_model"
                )
                
            except Exception as e:
                st.error(f"Error training model: {e}")

# Function for results
def results():
    st.header("📊 Results & Export")
    
    if st.session_state.processed_df is None:
        st.warning("No processed data to export!")
        return
    
    st.subheader("Processed Data")
    st.dataframe(st.session_state.processed_df.head())
    
    # Show model performance if available
    if st.session_state.trained_model is not None:
        st.subheader("Model Performance")
        if st.session_state.model_type == "Classification":
            st.info("Model trained for classification")
        else:
            st.info("Model trained for regression")
    
    # Export options
    st.subheader("Export Data")
    export_format = st.selectbox("Select export format", ["CSV", "Excel"], key="export_format")
    
    if export_format == "CSV":
        csv = st.session_state.processed_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv",
            key="download_csv"
        )
    else:
        # For Excel we need to use a different approach
        towrite = BytesIO()
        st.session_state.processed_df.to_excel(towrite, index=False, engine='openpyxl')
        towrite.seek(0)
        
        st.download_button(
            label="Download Excel",
            data=towrite,
            file_name="processed_data.xlsx",
            mime="application/vnd.ms-excel",
            key="download_excel"
        )

# Main app logic
if app_mode == "Load Data":
    load_data()
elif app_mode == "Data Overview":
    data_overview()
elif app_mode == "Data Cleaning":
    data_cleaning()
elif app_mode == "Data Visualization":
    data_visualization()
elif app_mode == "Feature Engineering":
    feature_engineering()
elif app_mode == "Model Training":
    model_training()
elif app_mode == "Results":
    results()

# Add some styling
st.sidebar.markdown("---")
st.sidebar.info("""
This application allows you to:
- Load datasets (upload or sample)
- Explore data statistics and information
- Clean data (handle missing values, duplicates, handle outliers, type casting,etc.)
- Visualize data with various plots
- Engineer features (encoding, scaling)
- Train machine learning models
- Export processed data and models
""")