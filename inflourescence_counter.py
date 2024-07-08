import streamlit as st
import os
from cellpose.io import imread
from cellpose import models, utils
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import pandas as pd
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load Cellpose models
cellected_model_dapi = models.CellposeModel(pretrained_model=r'\\fs\fs\Felix\cellected_code\Cellected_work\cellpose_dapi_sox2\dapi_grey_tiff_v1')
cellected_model_sox2 = models.CellposeModel(pretrained_model=r'\\fs\fs\Felix\cellected_code\Cellected_work\cellpose_dapi_sox2\sox2_grey_tiff_v1')

def process_images(file_triples, input_dir):
    # Initialize a list to store results
    results = []

    # Create model_outputs directory in the input directory
    output_dir = os.path.join(input_dir, 'model_outputs')
    os.makedirs(output_dir, exist_ok=True)

    for dapi_file, fitc_file, zoverlay_file in file_triples:
        # Save uploaded files to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            dapi_path = os.path.join(temp_dir, dapi_file.name)
            fitc_path = os.path.join(temp_dir, fitc_file.name)
            zoverlay_path = os.path.join(temp_dir, zoverlay_file.name)
            with open(dapi_path, "wb") as f_dapi, open(fitc_path, "wb") as f_fitc, open(zoverlay_path, "wb") as f_zoverlay:
                f_dapi.write(dapi_file.read())
                f_fitc.write(fitc_file.read())
                f_zoverlay.write(zoverlay_file.read())

            # Skip files containing "oct4" in their names
            if 'oct4' in dapi_file.name.lower() or 'oct4' in fitc_file.name.lower() or 'oct4' in zoverlay_file.name.lower():
                st.warning(f"Skipping files {dapi_file.name}, {fitc_file.name}, and {zoverlay_file.name} because they contain 'oct4'")
                continue

            # Process DAPI image
            dapi_image = imread(dapi_path)
            dapi_image = (dapi_image - dapi_image.min()) / (dapi_image.max() - dapi_image.min()) * 255  # Normalize pixel values to [0, 255]
            st.image(dapi_image.astype(np.uint8), caption='DAPI Image', use_column_width=True)
            dapi_masks, dapi_flows, _ = cellected_model_dapi.eval(dapi_image, diameter=None)
            num_dapi_cells = dapi_masks.max()

            outlines_dapi = utils.outlines_list(dapi_masks)

            # Process FITC image
            fitc_image = imread(fitc_path)
            fitc_image = (fitc_image - fitc_image.min()) / (fitc_image.max() - fitc_image.min()) * 255  # Normalize pixel values to [0, 255]
            st.image(fitc_image.astype(np.uint8), caption='FITC Image', use_column_width=True)
            fitc_masks, fitc_flows, _ = cellected_model_sox2.eval(fitc_image, diameter=None)
            num_fitc_cells = fitc_masks.max()

            outlines_fitc = utils.outlines_list(fitc_masks)

            # Load the zoverlay image (converted to grayscale)
            zoverlay_image = Image.open(zoverlay_path).convert("L")
            zoverlay_image = np.array(zoverlay_image)
            zoverlay_image = (zoverlay_image - zoverlay_image.min()) / (zoverlay_image.max() - zoverlay_image.min()) * 255  # Normalize pixel values to [0, 255]
            st.image(zoverlay_image.astype(np.uint8), caption='zOverlay Image', use_column_width=True)

            # Create a combined overlay image
            plt.figure(figsize=(8,8))
            plt.imshow(zoverlay_image, cmap="gray")
            for o in outlines_dapi:
                plt.plot(o[:,0], o[:,1], color='b')
            for o in outlines_fitc:
                plt.plot(o[:,0], o[:,1], color='g')
            plt.axis('off')

            # Save the combined overlay image to the output directory
            combined_output_path = os.path.join(output_dir, f"{os.path.splitext(zoverlay_file.name)[0]}_combined_outlines.png")
            plt.savefig(combined_output_path)
            plt.close()

            # Display the combined overlay image in Streamlit
            st.image(combined_output_path, caption='Combined Outlines', use_column_width=True)

            # Store image paths, cell counts, and segmentation results
            results.append((dapi_path, fitc_path, zoverlay_path, num_dapi_cells, num_fitc_cells, dapi_masks, fitc_masks, dapi_flows, fitc_flows))

    return results, output_dir

# Streamlit app
def main():
    st.title("Cellpose Image Processing App")

    # Instruction section
    st.sidebar.markdown("### Instructions:")
    st.sidebar.markdown("1. Browse and select the DAPI, FITC, and zOverlay files.")
    st.sidebar.markdown("2. Click the 'Process Images' button to start image processing.")

    # File uploader for DAPI, FITC, and zOverlay images
    uploaded_files = st.sidebar.file_uploader("Upload DAPI, FITC, and zOverlay Images", type=["tif", "tiff", "jpg", "jpeg"], accept_multiple_files=True)

    if st.sidebar.button("Process Images"):
        if not uploaded_files:
            st.error("Please upload DAPI, FITC, and zOverlay images.")
        else:
            st.info("Processing images...")
            # Group files by their common names
            file_groups = {}
            for file in uploaded_files:
                file_name = os.path.splitext(file.name)[0]
                prefix = file_name.rsplit('_', 1)[0]  # Get the common prefix
                if prefix not in file_groups:
                    file_groups[prefix] = {'dapi': None, 'fitc': None, 'zoverlay': None}
                if 'dapi' in file_name and (file.name.endswith('.tif') or file.name.endswith('.tiff')):
                    file_groups[prefix]['dapi'] = file
                elif 'fitc' in file_name and (file.name.endswith('.tif') or file.name.endswith('.tiff')):
                    file_groups[prefix]['fitc'] = file
                elif 'zoverlay' in file_name and (file.name.endswith('.jpg') or file.name.endswith('.jpeg')):
                    file_groups[prefix]['zoverlay'] = file
            
            # Process each file triple
            file_triples = [(group['dapi'], group['fitc'], group['zoverlay']) for group in file_groups.values() if group['dapi'] and group['fitc'] and group['zoverlay']]
            input_dir = os.path.dirname(file_triples[0][0].name)  # Get the directory of the first uploaded file
            results, output_dir = process_images(file_triples, input_dir)
            st.success("Image processing complete!")

            # Initialize a list to store the data for CSV output
            csv_data = []

            # Display results and images
            for dapi_image_path, fitc_image_path, zoverlay_image_path, num_dapi_cells, num_fitc_cells, _, _, _, _ in results:
                # Extract the image name from the path
                dapi_image_name = os.path.splitext(os.path.basename(dapi_image_path))[0]
                # Remove the file extension
                dapi_image_name_short = os.path.splitext(dapi_image_name)[0]
                dapi_image_name_short = dapi_image_name_short.replace("_dapi", "")
                # Get the last part of the image name (without the directory)
                dapi_image_name_short = dapi_image_name_short.split(os.path.sep)[-1]
                percentage_fitc_to_dapi = round(num_fitc_cells / num_dapi_cells * 100, 3)
                csv_data.append((dapi_image_name_short, num_dapi_cells, num_fitc_cells, percentage_fitc_to_dapi))

            # Create a DataFrame from the csv_data list
            df_csv = pd.DataFrame(csv_data, columns=['Image Name', 'Num DAPI Cells', 'Num FITC Cells', 'Percentage FITC to DAPI'])
            st.write(df_csv)

            # Save the DataFrame to a CSV file in the output directory
            csv_output_path = os.path.join(output_dir, "cell_counts.csv")
            df_csv.to_csv(csv_output_path, index=False)

            # Provide a download link for the CSV file
            st.markdown(f"[Download CSV file](file:///{csv_output_path})")

if __name__ == "__main__":
    main()
