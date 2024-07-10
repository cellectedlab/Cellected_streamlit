import pandas as pd
import re
import streamlit as st
import os



# Function to read and clean data
def read_and_clean_data(input_file_1, input_file_2):
    try:
        # Define column names for df1
        df1_columns = ['chr', 'start', 'end', 'cnv', 'name', 'n1', 'n2', 'confidence']

        # Read the first dataset without headers and assign column names
        df1 = pd.read_csv(input_file_1, sep='\t', header=None, names=df1_columns)

        # Extract the sample_id from the 'name' column
        df1['sample_id'] = df1['name'].apply(lambda x: re.search(r'([^/]+)\.txt$', x).group(1))

        # Select and rename columns to match the desired format
        df1 = df1[['chr', 'start', 'end', 'cnv', 'sample_id', 'confidence']]
        df1 = df1.rename(columns={'confidence': 'value'})  # Rename 'confidence' to 'value' to match df2

        # Read the second dataset with explicit tab delimiter
        df2 = pd.read_csv(input_file_2, sep='\t')

        # Strip leading/trailing whitespace from all column names
        df1.columns = df1.columns.str.strip()
        df2.columns = df2.columns.str.strip()

        # Clean up sample_id in df2 and rename columns
        df2['SampleID'] = df2['SampleID'].str.strip()  # Remove trailing spaces from SampleID
        df2['sample_id'] = df2['SampleID'].str.extract(r'([^\\]+)\s*\[')[0].str.rstrip()  # Extract sample_id and remove trailing spaces

        # Rename columns in df2
        df2 = df2.rename(columns={
            'Chr': 'chr',
            'Start': 'start',
            'End': 'end',
            'Value': 'value'
        })

        # Select only the required columns in df2
        df2 = df2[['sample_id', 'chr', 'start', 'end', 'value']]

        # Add a new column to distinguish between the sources
        df1['cnv_algorithm'] = 'penn_cnv'
        df2['cnv_algorithm'] = 'cnv_partition'  # or any other label that fits your context

        # Calculate the size of the region for both dataframes
        df1['size'] = df1['end'] - df1['start']
        df2['size'] = df2['end'] - df2['start']

        return df1, df2
    
    except Exception as e:
        st.error(f"Error reading and cleaning data: {str(e)}")

def merge_sort_filter_data(df1, df2):
    # Merge the datasets based on common columns
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # Sort merged_df first by 'chr' and then by 'sample_id'
    merged_df = merged_df.sort_values(by=['chr', 'sample_id'])

    # Filter out rows where 'chr' column contains 'X' or 'Y'
    merged_df = merged_df[~merged_df['chr'].isin(['X', 'Y'])]

    # Filter out rows where 'size' is less than 50,000
    merged_df = merged_df[merged_df['size'] >= 50000]

    # Save merged_df to a relative path
    output_file = "merged_bookmarks_sorted_filtered.csv"
    merged_df.to_csv(output_file, sep='\t', index=False)

    # Return the relative path to the saved file
    return output_file


def find_overlaps_between_penncnv(file_path):
    # Read the CSV file with the correct delimiter
    data = pd.read_csv(file_path, delimiter='\t')

    # Strip whitespace from headers (if any)
    data.columns = data.columns.str.strip()

    # Ensure Start and End columns are strings, then remove commas and convert to integers
    data['start'] = data['start'].astype(str).str.replace(',', '').astype(int)
    data['end'] = data['end'].astype(str).str.replace(',', '').astype(int)

    # Split the data into two groups: one with "_penncnv" and one without
    data_penncnv = data[data['cnv_algorithm'].str.contains('penn_cnv')]
    data_non_penncnv = data[data['cnv_algorithm'].str.contains('cnv_partition')]

    # Dictionary to store overlapping entries
    overlapping_entries = []

    # Group data_penncnv by sample_id
    grouped_penncnv = data_penncnv.groupby('sample_id')

    # Iterate through each group
    for sample_id, group_penncnv in grouped_penncnv:
        # Find corresponding group in data_non_penncnv
        if sample_id in data_non_penncnv['sample_id'].values:
            group_non_penncnv = data_non_penncnv[data_non_penncnv['sample_id'] == sample_id]

            # Iterate through each row in penncnv group
            for _, row_penncnv in group_penncnv.iterrows():
                # Iterate through each row in non-penncnv group
                for _, row_non_penncnv in group_non_penncnv.iterrows():
                    # Check for overlap condition
                    if (row_penncnv['chr'] == row_non_penncnv['chr'] and
                        max(row_penncnv['start'], row_non_penncnv['start']) <= min(row_penncnv['end'], row_non_penncnv['end'])):
                        overlapping_entries.append({
                            'sample_id_1': row_penncnv['sample_id'],
                            'chr_1': row_penncnv['chr'],
                            'start_1': row_penncnv['start'],
                            'end_1': row_penncnv['end'],
                            'cnv_algorithm_1': 'penn_cnv',
                            'sample_id_2': row_non_penncnv['sample_id'],
                            'chr_2': row_non_penncnv['chr'],
                            'start_2': row_non_penncnv['start'],
                            'end_2': row_non_penncnv['end'],
                            'cnv_algorithm_2': 'cnv_partition',
                            'CN Value': row_non_penncnv['value']  # Assuming CN Value is from non-penncnv
                        })

    # Convert the list of dictionaries to a DataFrame
    overlap_df = pd.DataFrame(overlapping_entries)

    return overlap_df

    
# Function to clean and transform SampleID column
def transform_sample_id(sample_id):
    sample_id = sample_id.replace('-', '_')
    sample_id = sample_id.replace(' ', '_')
    sample_id = sample_id.replace(':', '_')
    return sample_id

def clean_sample_id(sample_id):
    idx = sample_id.find('_[')
    if idx != -1:
        return sample_id[:idx]
    else:
        return sample_id


# Streamlit app
def main():
    st.title('CNV Analysis App')
    st.markdown("""
    ### Requirements
    - The SNP output from the PennCNV code
    - The SNP bookmarks from GenomeStudio
    - The output of Streamlit for the CNV partition (snp_analysis)
    """)

    # File uploaders (only for SNP data file)
    input_file_1 = st.file_uploader('Upload SNP data from penncnv', type=['out', 'txt'])
    # File uploaders (only for SNP data file)
    input_file_2 = st.file_uploader('Upload SNP data bookmarks from genome studio', type=['csv', 'txt'])
    # File uploaders (only for SNP data file)
    export_file_path = st.file_uploader('Upload streamlit 1 output for cnv partition', type=['csv', 'txt'])

    # Button to start processing
    if st.button('Process Data'):
        try:
            # Read and clean the data
            df1, df2 = read_and_clean_data(input_file_1, input_file_2)

            # Merge, sort, and filter the data
            merged_df = merge_sort_filter_data(df1, df2)
            st.write(merged_df)

            # Find overlaps between penncnv and cnv_partition using the merged data
            overlap_df = find_overlaps_between_penncnv(merged_df)
            st.write(overlap_df)

            # Load and filter export data
            export_df = pd.read_csv(export_file_path)
            export_df['SampleID'] = export_df['SampleID'].apply(clean_sample_id)
            export_df['SampleID'] = export_df['SampleID'].apply(transform_sample_id)
            overlap_df['sample_id_1'] = overlap_df['sample_id_1'].apply(clean_sample_id)
            overlap_df['sample_id_1'] = overlap_df['sample_id_1'].apply(transform_sample_id)

            # Loop over rows in overlap_df and filter export_df
            filtered_df = pd.DataFrame()
            for index, row in overlap_df.iterrows():
                sample_id = row['sample_id_1']
                chr_value = row['chr_1']
                start_value = row['start_2']
                end_value = row['end_2']

                # Filter export_df based on SampleID and Chr
                filtered_rows = export_df[(export_df['SampleID'] == f"{sample_id}") & 
                                          (export_df['Chr'] == f"{chr_value}") & 
                                          (export_df['Start'] == start_value) & 
                                          (export_df['End'] == end_value)]

                # Append filtered rows to filtered_df
                filtered_df = pd.concat([filtered_df, filtered_rows], ignore_index=True)

            # Drop columns with all NaN values
            filtered_df = filtered_df.dropna(axis=1, how='all')

            # Drop specific columns ('Unnamed: 0' and the first column) if they exist
            columns_to_drop = ['Unnamed: 0']
            for column in columns_to_drop:
                if column in filtered_df.columns:
                    filtered_df = filtered_df.drop(columns=[column])

            # Drop duplicate rows based on all columns
            filtered_df = filtered_df.drop_duplicates()

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    main()


