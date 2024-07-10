import pandas as pd
import re
import streamlit as st
import os


def read_snp_data(df, sample_id):
    df = df.drop(columns=["Author", "CreatedDate", "BookmarkType"])
    df_filtered = df[df['SampleID'] == sample_id]
    return df_filtered

def transform_sample_id(sample_id):
    # Replace hyphens, spaces, and colons with underscores
    sample_id = sample_id.replace('-', '_')
    sample_id = sample_id.replace(' ', '_')
    sample_id = sample_id.replace(':', '_')
    return sample_id

def read_gene_annotation(gene_file_path):
    df_genes = pd.read_csv(gene_file_path, sep='\t', dtype={'Chr': 'object'})
    return df_genes

def get_gene_info(df_filtered, df_genes):
    regions = []
    for index, row in df_filtered.iterrows():
        region = {
            "Chr": row["Chr"],
            "start": row["Start"],
            "end": row["End"]
        }
        regions.append(region)

    gene_info = []
    for region in regions:
        chromosome = region["Chr"]
        start = region["start"]
        end = region["end"]

        genes_within_range = df_genes[(df_genes['Chr'] == chromosome) & 
                                       ((df_genes['MapInfo'] >= start) & (df_genes['MapInfo'] <= end))]

        gene_names = set(genes_within_range['Gene(s)'].str.split(',').explode().str.strip())
        gene_names_str = ','.join(str(gene) for gene in gene_names if pd.notnull(gene))
        gene_info.append(gene_names_str)

    df_filtered['Gene(s) within range'] = gene_info
    return df_filtered

def process_confidence(df_filtered):
    df_filtered['Confidence'] = df_filtered['Comment'].str.replace('CNV Confidence:', '').str.strip()
    df_filtered['LOH'] = df_filtered['Confidence'].str.contains('LOH', na=False)
    df_filtered['Confidence'] = df_filtered['Confidence'].str.replace('LOH Region', '')
    df_filtered.drop(columns=['Comment'], inplace=True)
    return df_filtered

def process_all_samples(snp_data, gene_data):
    # Read SNP data and process for each sample
    result_dfs = []
    sample_ids = get_unique_sample_ids(snp_data)
    for sample_id in sample_ids:
        df_filtered = read_snp_data(snp_data, sample_id)
        df_filtered = get_gene_info(df_filtered, gene_data)
        df_filtered = process_confidence(df_filtered)
        result_dfs.append(df_filtered)
    
    # Concatenate all processed DataFrames
    final_result_df = pd.concat(result_dfs, ignore_index=True)
    
    return final_result_df

# Function to get unique sample IDs
def get_unique_sample_ids(df):
    sample_ids = df['SampleID'].str.extract(r'(\w+\s*\[\d+\])')[0].unique()
    return sample_ids

def read_cancer_genes(df):
    # Select specific columns
    selected_columns = ['Hugo Symbol', 'Is Oncogene', 'Is Tumor Suppressor Gene']
    
    # Create a new DataFrame with only the selected columns
    selected_genes_df = df[selected_columns]
    
    # Rename columns
    selected_genes_df = selected_genes_df.rename(columns={
        'Hugo Symbol': 'Gene',
        'Is Oncogene': 'Oncogene',
        'Is Tumor Suppressor Gene': 'Tumor_suppressor_gene'
    })
    
    return selected_genes_df

def process_data(df_filtered, selected_genes_df):
    # Initialize a list to store the results
    results = []

    # Iterate over each row in df_filtered
    for index, row in df_filtered.iterrows():
        # Get the gene names for the current row
        genes_in_range = row['Gene(s) within range'].split(',')

        # Check if any of the genes in the current row are in the selected genes DataFrame
        genes_present = selected_genes_df[selected_genes_df['Gene'].isin(genes_in_range)]

        # Count the number of oncogenes and tumor suppressor genes
        num_oncogenes = sum(genes_present['Oncogene'] == 'Yes')
        num_tumor_suppressor_genes = sum(genes_present['Tumor_suppressor_gene'] == 'Yes')

        # Create a list of cancer genes
        cancer_genes = genes_present['Gene'].tolist()

        # Extract start and end from the current row
        start = row['Start']
        end = row['End']
        
        # Get the SampleID
        sample_id = row['SampleID']

        # Append the result to the list
        results.append((sample_id, start, end, num_oncogenes, num_tumor_suppressor_genes, cancer_genes))

    # Create a DataFrame from the results
    columns = ['SampleID', 'Start', 'End', 'Num Oncogenes', 'Num Tumor Suppressor Genes', 'Cancer Genes']
    additional_df = pd.DataFrame(results, columns=columns)
    
    return additional_df

def merge_data(df_filtered, additional_df):
    # Merge the dataframes based on their mutual start, end, and SampleID values
    merged_df = pd.merge(df_filtered, additional_df, on=['SampleID', 'Start', 'End'], how='inner')

    # Drop duplicate rows based on the combination of 'SampleID', 'Start', and 'End'
    merged_df = merged_df.drop_duplicates(subset=['SampleID', 'Start', 'End'])

    # Move the 'Gene(s) within range' column to the end
    gene_column = merged_df.pop('Gene(s) within range')
    merged_df['Gene(s) within range'] = gene_column
    
    return merged_df

def clean_gene_names(gene_names):
    # Convert list of gene names to a single string
    gene_names_str = ', '.join(gene_names)
    # Remove square brackets and single quotes
    cleaned_names = re.sub(r"[\[\]']", "", gene_names_str)
    # Split the cleaned names by comma and return as a list
    return cleaned_names.split(", ")

def process_snp_data(df, confidence_threshold=100, size_threshold=10000, remove_cn_value_2_loh_false=True):
    data_list = []

    # Use the provided DataFrame directly
    df_filtered = df.copy()

    # Drop rows with NaN values in 'Cancer Genes' column
    df_filtered = df_filtered.dropna(subset=['Cancer Genes'])

    # Convert 'Confidence' column to numeric
    df_filtered['Confidence'] = pd.to_numeric(df_filtered['Confidence'], errors='coerce')

    # Filter rows based on size and confidence thresholds
    df_filtered = df_filtered[df_filtered['Size'] > size_threshold]
    df_filtered = df_filtered[df_filtered['Confidence'] > confidence_threshold]

    # Optionally remove rows with CN Value of 2 and LOH False
    if remove_cn_value_2_loh_false:
        df_filtered = df_filtered[~((df_filtered['Value'] == 2) & (df_filtered['LOH'] == False))]

    # Sort and drop duplicates based on 'Cancer Genes'
    df_filtered = df_filtered.sort_values(by='Size', ascending=False)

    # Iterate over each row in the filtered DataFrame
    for index, row in df_filtered.iterrows():
        sample_id = row['SampleID']
        chr_value = row['Chr']
        start_value = row['Start']
        end_value = row['End']
        size_value = row['Size']
        cn_value = row['Value']
        confidence_value = row['Confidence']
        loh_value = row['LOH']

        # Check if the gene is an oncogene or tumor suppressor gene
        oncogene = check_oncogene(row['Cancer Genes'], row['Num Oncogenes'], cn_value)
        tumor_suppressor_gene = check_tumor_suppressor_gene(row['Cancer Genes'], row['Num Tumor Suppressor Genes'], cn_value)

        # Append data to the list
        data_list.append([sample_id, chr_value, start_value, end_value, size_value, cn_value, confidence_value, loh_value, oncogene, tumor_suppressor_gene, row['Cancer Genes']])

    # Define columns for the result DataFrame
    columns = ['SampleID', 'Chr', 'Start', 'End', 'Size', 'CN Value', 'Confidence', 'LOH', 'Oncogene', 'Tumor Supp', 'Cancer Genes']

    # Create the result DataFrame
    result_df = pd.DataFrame(data_list, columns=columns)

    # Sort the result DataFrame by 'Chr'
    result_df = result_df.sort_values(by='Chr')

    return result_df

def check_oncogene(gene_names, num_oncogenes, cn_value):
    if num_oncogenes > 0:
        return 'Yes'
    gene_names_list = clean_gene_names(gene_names)
    if 'Oncogene' in gene_names_list:
        return 'Yes'
    return 'No'

def check_tumor_suppressor_gene(gene_names, num_tumor_suppressor_genes, cn_value):
    if num_tumor_suppressor_genes:
        return 'Yes'
    gene_names_list = clean_gene_names(gene_names)
    if 'Tumor Suppressor Gene' in gene_names_list:
        return 'Yes'
    return 'No'

# Function to remove rows containing specific chromosomes
def remove_rows_with_chromosomes(df, chromosomes_to_remove):
    return df[~df['Chr'].isin(chromosomes_to_remove)]

def read_gene_annotation(gene_file_path):
    try:
        df_genes = pd.read_csv(gene_file_path, sep='\t', dtype={'Chr': 'object'})
        return df_genes
    except pd.errors.EmptyDataError:
        st.error("The uploaded gene annotation file is empty.")
        return None


def add_cytoband_info(df, cytoband_data):
    cytoband_info = []
    for index, row in df.iterrows():
        chromosome = row['Chr']
        start = row['Start']
        end = row['End']
        
        start_position = None
        end_position = None

        cytoband_range = cytoband_data[(cytoband_data['chromosome'] == chromosome) &
                                       (cytoband_data['bp_start'] <= end) & 
                                       (cytoband_data['bp_stop'] >= start)]

        if not cytoband_range.empty:
            for idx, cytoband_row in cytoband_range.iterrows():
                bp_start = cytoband_row['bp_start']
                bp_stop = cytoband_row['bp_stop']
                arm = cytoband_row['arm']
                band = cytoband_row['band']
                
                if bp_start <= start < bp_stop:
                    start_position = f"{chromosome}{arm}{band}"
                if bp_start < end <= bp_stop:
                    end_position = f"{chromosome}{arm}{band}"
            
            if start_position and end_position:
                cytoband_name = f"{start_position}-{end_position}"
            elif start_position:
                cytoband_name = f"{start_position}"
            else:
                cytoband_name = ''
        else:
            cytoband_name = ''
        
        cytoband_info.append(cytoband_name)

    df['Cytoband'] = cytoband_info
    return df

KNOWN_REGIONS = [
    '20q', '12p', 'xp', '18q', '9q', '17q', '5q', '11p', 
    '13q', '7q', '1p', '4q', '3p', '19p', '14q', '8q', 
    '15q', '6q', '7p', '22q', '16q', '2q'
]

def check_abnormalities(cytoband_info):
    for region in KNOWN_REGIONS:
        if region in cytoband_info:
            return region
    return None

def add_abnormalities_column(df):
    df['Abnormalities'] = df['Cytoband'].apply(check_abnormalities)
    return df


def main():
    st.title('SNP Data Processing App')

    # Input field for unwanted sample IDs
    unwanted_sample_ids = st.text_input('Enter unwanted Sample IDs (comma-separated)')

    # Input field for ignored chromosomes
    ignored_chromosomes = st.text_input('Enter chromosomes to ignore (comma-separated)')

    # Checkbox for removing CN values of 2
    remove_cn_value_2 = st.checkbox('Remove rows with CN value of 2')

    # File uploaders (only for SNP data file)
    file_path = st.file_uploader('Upload SNP data file (CSV or TXT)', type=['csv', 'txt'])

    # Input fields for setting thresholds
    confidence_threshold = st.number_input('Enter Confidence Threshold', min_value=0, max_value=10000, value=100)
    size_threshold = st.number_input('Enter Size Threshold', min_value=0, max_value=1000000000, value=100000)
    remove_cn_value_2_loh_false = st.checkbox('Remove CN values of 2 when Loss of heterozygosity is not present')

    # Determine the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative paths
    gene_file_path = os.path.join(script_dir, 'gene_list_genome_studio.txt')
    cancer_gene_file_path = os.path.join(script_dir, 'cancerGeneList.tsv')
    cytoband_file_path = os.path.join(script_dir, 'cytoband_positions_ncbi.csv')

    # Button to start processing
    if st.button('Process Data'):
        try:
            if not file_path:
                st.warning('Please upload the SNP data file.')
                return

            file_path.seek(0)
            file_extension = file_path.name.split('.')[-1].lower()
            if file_extension == 'csv':
                snp_data = pd.read_csv(file_path)
            elif file_extension == 'txt':
                snp_data = pd.read_csv(file_path, sep='\t')

            gene_data = pd.read_csv(gene_file_path, sep='\t', dtype={'Chr': 'object'})
            cancer_gene_data = pd.read_csv(cancer_gene_file_path, sep='\t')
            cytoband_data = pd.read_csv(cytoband_file_path)

            # Automatically transform SampleID column
            snp_data['SampleID'] = snp_data['SampleID'].apply(transform_sample_id)

            # Process SNP data for selected samples
            processed_data = process_all_samples(snp_data, gene_data)

            selected_genes_df = read_cancer_genes(cancer_gene_data)

            # Process additional data
            additional_df = process_data(processed_data, selected_genes_df)

            # Merge dataframes
            merged_df = merge_data(processed_data, additional_df)

            # Process the merged DataFrame with the selected thresholds
            processed_merged_df = process_snp_data(merged_df, confidence_threshold, size_threshold, remove_cn_value_2_loh_false)

            # Filter out unwanted sample IDs
            if unwanted_sample_ids:
                unwanted_sample_ids_list = [x.strip() for x in unwanted_sample_ids.split(',')]
                processed_merged_df = processed_merged_df[~processed_merged_df['SampleID'].isin(unwanted_sample_ids_list)]

            # Filter out ignored chromosomes
            if ignored_chromosomes:
                ignored_chromosomes_list = [x.strip() for x in ignored_chromosomes.split(',')]
                processed_merged_df = processed_merged_df[~processed_merged_df['Chr'].isin(ignored_chromosomes_list)]

            # Remove rows with CN value of 2 if selected
            if remove_cn_value_2:
                processed_merged_df = processed_merged_df[processed_merged_df['CN Value'] != 2]

            # Add cytoband information
            processed_merged_df_with_cytoband = add_cytoband_info(processed_merged_df, cytoband_data)

            # Add abnormalities column
            processed_merged_df_with_abnormalities = add_abnormalities_column(processed_merged_df_with_cytoband)

            processed_merged_df_with_abnormalities.reset_index(drop=True, inplace=True)
            st.write(processed_merged_df_with_abnormalities)

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")

if __name__ == '__main__':
    main()
