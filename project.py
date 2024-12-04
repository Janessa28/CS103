import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Data Master: All-in-One Data Analysis Toolkit")

# Initialize an empty list to store datasets and their names
dfs = []
processed_dfs = {}

# File Upload: Allow multiple file uploads
uploaded_files = st.file_uploader("Upload your datasets", type=["csv", "xlsx"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)

        # Add dataset to the list
        dfs.append((uploaded_file.name, df))

        # Display uploaded data
        st.write(f"Dataset: {uploaded_file.name}")
        st.write(df)

    # Sidebar for selecting datasets
    st.sidebar.title("Data Operations")
    dataset_options = [name for name, _ in dfs]

    if len(dataset_options) > 1:
        selected_datasets = st.sidebar.multiselect("Select Datasets for Operations", dataset_options, default=dataset_options)
    else:
        selected_datasets = dataset_options

    # Perform operations only if datasets are selected
    if selected_datasets:
        # Data Deduplication
        if st.sidebar.checkbox("Data Deduplication"):
            for dataset_name in selected_datasets:
                st.subheader(f"Data Deduplication for {dataset_name}")
                df = next(df for name, df in dfs if name == dataset_name)
                dedup_df = df.drop_duplicates()
                st.write("Deduplicated Data:")
                st.write(dedup_df)

                # Store processed data
                processed_dfs[dataset_name] = dedup_df


        # Data Cleansing
        if st.sidebar.checkbox("Data Cleansing"):
            for dataset_name in selected_datasets:
                st.subheader(f"Data Cleansing for {dataset_name}")
                df = processed_dfs.get(dataset_name, next(df for name, df in dfs if name == dataset_name))
                
                # Drop columns with more than 50% missing values
                missing_threshold = 0.5
                missing_percentages = df.isnull().mean()
                columns_to_drop = missing_percentages[missing_percentages > missing_threshold].index
                df = df.drop(columns=columns_to_drop)
                st.write(f"Dropped columns with more than 50% missing values: {list(columns_to_drop)}")

                # Drop rows with any missing values
                df = df.dropna()
                st.write(f"Dropped rows with missing values. Remaining rows: {len(df)}")
                
                # Fill remaining missing values with 0
                filled_df = df.fillna(0)
                        
                # Replace negative values with 0
                numeric_cols = filled_df.select_dtypes(include=['number']).columns
                filled_df[numeric_cols] = filled_df[numeric_cols].clip(lower=0)
                        
                st.write("Cleansed Data:")
                st.write(filled_df)

                # Store processed data
                processed_dfs[dataset_name] = filled_df



        # Format Revision
        if st.sidebar.checkbox("Format Revision"):
            for dataset_name in selected_datasets:
                st.subheader(f"Format Revision for {dataset_name}")
                df = processed_dfs.get(dataset_name, next(df for name, df in dfs if name == dataset_name))
                
                # Display original data types
                st.write("Original Data Types:")
                st.write(df.dtypes)

                # Allow multiple column selection
                columns_to_convert = st.multiselect(f"Select columns to change data type ({dataset_name})", df.columns, key=f"convert_columns_{dataset_name}")
                
                # Select target data type
                data_type = st.selectbox(f"Select target data type ({dataset_name})", ["int", "float", "str", "datetime"], key=f"type_{dataset_name}")
                
                if st.button(f"Convert Selected Columns for {dataset_name}", key=f"convert_btn_{dataset_name}"):
                    try:
                        # Loop over each selected column
                        for column_to_convert in columns_to_convert:
                            # Show the original data type of the selected column
                            original_dtype = df[column_to_convert].dtype
                            st.write(f"Original Data Type of {column_to_convert}: {original_dtype}")

                            # Perform conversion based on selected type
                            if data_type == "int":
                                df[column_to_convert] = pd.to_numeric(df[column_to_convert], errors="coerce").astype("Int64")
                            elif data_type == "float":
                                df[column_to_convert] = pd.to_numeric(df[column_to_convert], errors="coerce")
                            elif data_type == "str":
                                df[column_to_convert] = df[column_to_convert].astype(str)
                            elif data_type == "datetime":
                                df[column_to_convert] = pd.to_datetime(df[column_to_convert], errors="coerce")

                            # Show updated data type of the selected column
                            updated_dtype = df[column_to_convert].dtype
                            st.write(f"Updated Data Type of {column_to_convert}: {updated_dtype}")

                        # Save the updated dataset back to processed_dfs
                        processed_dfs[dataset_name] = df

                        # Display the updated dataset
                        st.write("Updated Dataset:")
                        st.write(df)
                        
                    except Exception as e:
                        st.error(f"Error: {e}")


        # Merging/Joining
        if len(dfs) > 1 and st.sidebar.checkbox("Merge/Join Datasets"):
            st.subheader("Merge Datasets")
            dataset1 = st.selectbox("Select First Dataset", dataset_options, key="merge_dataset1")
            dataset2 = st.selectbox("Select Second Dataset", [name for name in dataset_options if name != dataset1], key="merge_dataset2")
            df1 = processed_dfs.get(dataset1, next(df for name, df in dfs if name == dataset1))
            df2 = processed_dfs.get(dataset2, next(df for name, df in dfs if name == dataset2))

            # Select columns to join on with dataset previews
            join_column1 = st.selectbox(f"Select join column from {dataset1}", df1.columns)
            st.write(f"Dataset: {dataset1} - Full Data:")
            st.write(df1)

            join_column2 = st.selectbox(f"Select join column from {dataset2}", df2.columns)
            st.write(f"Dataset: {dataset2} - Full Data:")
            st.write(df2)

            join_method = st.selectbox("Select join method", ["inner", "left", "right", "outer"])

            # Merge with full dataset display
            if st.button("Merge Datasets"):
                try:
                    merged_df = pd.merge(df1, df2, how=join_method, left_on=join_column1, right_on=join_column2)
                    st.success("Datasets merged successfully!")
                    
                    # Display full merged result
                    st.write("Merged Dataset:")
                    st.write(merged_df)

                    # Option to save the merged dataset
                    if st.checkbox("Save Merged Dataset"):
                        new_dataset_name = st.text_input("Enter name for the merged dataset", f"Merged_{dataset1}_{dataset2}")
                        if new_dataset_name:
                            processed_dfs[new_dataset_name] = merged_df
                            st.success(f"Merged dataset saved as '{new_dataset_name}'!")
                except Exception as e:
                    st.error(f"Error during merging: {e}")
                    
                    

        # Data Derivation
        if st.sidebar.checkbox("Data Derivation"):
            for dataset_name in selected_datasets:
                st.subheader(f"Data Derivation for {dataset_name}")
                df = processed_dfs.get(dataset_name, next(df for name, df in dfs if name == dataset_name))
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if len(numeric_cols) < 2:
                    st.warning("Not enough numeric columns to perform derivation.")
                else:
                    col1 = st.selectbox(f"Select first column ({dataset_name})", numeric_cols, key=f"{dataset_name}_col1")
                    col2 = st.selectbox(f"Select second column ({dataset_name})", numeric_cols, key=f"{dataset_name}_col2")
                    operation = st.selectbox("Select operation", ["Add", "Subtract", "Multiply", "Divide"], key=f"{dataset_name}_operation")
                    
                    if st.button(f"Apply Derivation on {dataset_name}", key=f"{dataset_name}_apply"):
                        try:
                            # Define new column name
                            new_col_name = f"{col1}_{operation.lower()}_{col2}"
                            
                            # Perform selected operation
                            if operation == "Add":
                                df[new_col_name] = df[col1] + df[col2]
                            elif operation == "Subtract":
                                df[new_col_name] = df[col1] - df[col2]
                            elif operation == "Multiply":
                                df[new_col_name] = df[col1] * df[col2]
                            elif operation == "Divide":
                                df[new_col_name] = df[col1] / df[col2].replace(0, float('nan'))
                                df[new_col_name] = df[new_col_name].fillna(0)  # Handle division by zero
                            
                            # Show the updated dataframe to the user
                            st.success(f"New column '{new_col_name}' added successfully!")
                            st.write(df)

                            # Provide an option to save the changes
                            if st.checkbox(f"Save changes for {dataset_name}", key=f"{dataset_name}_save"):
                                processed_dfs[dataset_name] = df
                                st.success(f"Changes saved for {dataset_name}.")
                            else:
                                st.warning("Changes not saved. Check 'Save changes' to persist changes.")
                        except Exception as e:
                            st.error(f"Error: {e}")


        # Data Aggregation
        if st.sidebar.checkbox("Data Aggregation"):
            for dataset_name in selected_datasets:
                st.subheader(f"Data Aggregation for {dataset_name}")
                df = processed_dfs.get(dataset_name, next(df for name, df in dfs if name == dataset_name))
                
                # Select column to group by
                group_by_col = st.selectbox(f"Select a column to group by ({dataset_name})", df.columns, key=f"groupby_{dataset_name}")
                
                # Select numeric columns to aggregate
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                agg_cols = st.multiselect(f"Select columns to aggregate ({dataset_name})", numeric_cols, key=f"aggcols_{dataset_name}")
                
                # Select aggregation functions
                available_funcs = ["sum", "mean", "min", "max"]
                agg_dict = {}
                
                for col in agg_cols:
                    func = st.selectbox(f"Select aggregation function for {col} ({dataset_name})", available_funcs, key=f"aggfunc_{dataset_name}_{col}")
                    agg_dict[col] = func
                
                # Perform aggregation
                if st.button(f"Perform Aggregation for {dataset_name}", key=f"agg_button_{dataset_name}"):
                    if not group_by_col or not agg_dict:
                        st.warning("Please select a group-by column and at least one column with an aggregation function.")
                    else:
                        try:
                            # Perform aggregation
                            aggregated_df = df.groupby(group_by_col).agg(agg_dict).reset_index()
                            st.success(f"Aggregation completed for {dataset_name}.")
                            st.write(aggregated_df)
                            
                            # Provide an option to save the aggregated results
                            if st.checkbox(f"Save Aggregated Data for {dataset_name}", key=f"save_agg_{dataset_name}"):
                                processed_dfs[dataset_name] = aggregated_df
                                st.success(f"Aggregated data saved for {dataset_name}.")
                        except Exception as e:
                            st.error(f"Error during aggregation: {e}")



        # Descriptive Statistics
        if st.sidebar.checkbox("Descriptive Statistics"):
            for dataset_name in selected_datasets:
                st.subheader(f"Descriptive Statistics for {dataset_name}")
                df = processed_dfs.get(dataset_name, next(df for name, df in dfs if name == dataset_name))
                st.write(df.describe())

        # Data Visualization
        if st.sidebar.checkbox("Data Visualization"):
            for dataset_name in selected_datasets:
                st.subheader(f"Data Visualization for {dataset_name}")
                df = processed_dfs.get(dataset_name, next(df for name, df in dfs if name == dataset_name))
                
                # Select chart type
                chart_type = st.selectbox(f"Select chart type ({dataset_name})", ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Count Plot", "Pie Chart"])
                
                # Separate numeric and non-numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
                
                if chart_type in ["Bar Chart", "Line Chart", "Histogram"]:
                    if numeric_cols:
                        column = st.selectbox(f"Select column to visualize ({dataset_name})", numeric_cols)
                    else:
                        st.warning(f"No numeric columns available for {chart_type}. Please choose another chart.")
                elif chart_type == "Scatter Plot":
                    if len(numeric_cols) >= 2:
                        x_col = st.selectbox(f"Select X-axis column ({dataset_name})", numeric_cols)
                        y_col = st.selectbox(f"Select Y-axis column ({dataset_name})", numeric_cols)
                    else:
                        st.warning(f"Not enough numeric columns for scatter plot in {dataset_name}.")
                elif chart_type == "Count Plot":
                    if non_numeric_cols:
                        column = st.selectbox(f"Select categorical column to visualize ({dataset_name})", non_numeric_cols)
                    else:
                        st.warning("No non-numeric columns available for count plot.")
                elif chart_type == "Pie Chart":
                    if non_numeric_cols:
                        column = st.selectbox(f"Select categorical column to visualize ({dataset_name})", non_numeric_cols)
                    else:
                        st.warning("No non-numeric columns available for pie chart.")
                
                # Generate the selected chart
                if st.button(f"Generate {chart_type} for {dataset_name}"):
                    fig, ax = plt.subplots()
                    try:
                        if chart_type == "Bar Chart":
                            df[column].value_counts().plot(kind='bar', ax=ax)
                        elif chart_type == "Line Chart":
                            ax.plot(df.index, df[column])
                        elif chart_type == "Scatter Plot":
                            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
                        elif chart_type == "Histogram":
                            df[column].plot(kind='hist', bins=20, ax=ax)
                        elif chart_type == "Count Plot":
                            sns.countplot(data=df, x=column, ax=ax)
                        elif chart_type == "Pie Chart":
                            df[column].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
                        
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error: {e}")