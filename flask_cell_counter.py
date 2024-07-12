import streamlit as st
from PIL import Image, ExifTags
import cv2
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
from cellpose import models, utils
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to extract image date and exact time from EXIF data
def get_image_date_and_time(image):
    exif_data = image._getexif()
    if exif_data is not None:
        for tag, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag, tag)
            if tag_name == 'DateTimeOriginal':
                date_time = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                return date_time.date(), date_time
    return None, None

# Function to split image into subpanels
def split_image_into_subpanels(image, horizontal_offset, vertical_offset, num_rows, num_cols):
    height, width = image.shape[:2]
    panel_size = min(height, width) // 8

    subpanels = []
    center_x = width // 2
    center_y = height // 2
    total_width = num_cols * panel_size + (num_cols - 1) * horizontal_offset
    total_height = num_rows * panel_size + (num_rows - 1) * vertical_offset
    start_x = center_x - total_width // 2
    start_y = center_y - total_height // 2

    image_with_outlines = image.copy()
    for row in range(num_rows):
        for col in range(num_cols):
            x1 = start_x + col * (panel_size + horizontal_offset)
            y1 = start_y + row * (panel_size + vertical_offset)
            subpanel_image = image[y1:y1+panel_size, x1:x1+panel_size]
            subpanels.append(subpanel_image)
            cv2.rectangle(image_with_outlines, (x1, y1), (x1 + panel_size, y1 + panel_size), (0, 255, 0), 2)
    return subpanels, image_with_outlines

# Function to segment subpanel using Cellpose model
def segment_with_cellpose(image, model):
    masks, _, _ = model.eval(image, diameter=None)
    num_cells = masks.max()

    outlines = utils.outlines_list(masks)
    image_with_outlines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for outline in outlines:
        for i in range(len(outline) - 1):
            cv2.line(image_with_outlines, tuple(outline[i]), tuple(outline[i + 1]), (0, 255, 0), 2)
    
    return num_cells, image_with_outlines

# Function to count cells in image
def count_cells_in_image(image, model, show_outlines=False):
    horizontal_offset = 50
    vertical_offset = 50
    num_rows = 5
    num_cols = 5
    
    subpanels, image_with_outlines = split_image_into_subpanels(image, horizontal_offset, vertical_offset, num_rows, num_cols)
    total_cell_count = 0
    cell_counts = []
    outlined_subpanels = []
    
    for subpanel in subpanels:
        gray_subpanel = cv2.cvtColor(subpanel, cv2.COLOR_BGR2GRAY)
        num_cells, outlined_subpanel = segment_with_cellpose(gray_subpanel, model)
        total_cell_count += num_cells
        cell_counts.append(num_cells)
        if show_outlines:
            outlined_subpanels.append(outlined_subpanel)
    
    return total_cell_count, cell_counts, image_with_outlines, outlined_subpanels

# Function to process images and generate daily summary
def process_images(images_by_date, model, show_outlines):
    daily_summary = []
    exact_times = defaultdict(list)
    
    for date, images in images_by_date.items():
        st.markdown(f"#### {date.strftime('%Y-%m-%d')}")
        total_cells = []
        
        for i, (image, exact_time) in enumerate(images, start=1):
            image_cv = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
            total_cells_count, cell_counts, image_with_outlines, outlined_subpanels = count_cells_in_image(image_cv, model, show_outlines)
            total_cells.append(total_cells_count)
            
            # Store the exact time the photo was taken
            if exact_time:
                exact_times[date].append(exact_time)
            
            # Display the image with or without outlines
            if show_outlines:
                st.image(cv2.cvtColor(image_with_outlines, cv2.COLOR_BGR2RGB), caption=f"Total Cells: {total_cells_count}", use_column_width=True)
                for j, outlined_subpanel in enumerate(outlined_subpanels):
                    st.image(outlined_subpanel, caption=f"Subpanel {j + 1}", use_column_width=True)
            else:
                st.image(cv2.cvtColor(image_with_outlines, cv2.COLOR_BGR2RGB), caption=f"Total Cells: {total_cells_count}", use_column_width=True)
        
        # Calculate mean, std dev, and coefficient of variation (CV) for total cells
        mean_total = np.mean(total_cells)
        std_dev_total = np.std(total_cells)
        coeff_var = std_dev_total / mean_total if mean_total != 0 else 0.0
        count_225 = mean_total * 13050.0146
        count_75 = count_225 / 3
        
        # Append daily summary
        summary_entry = {
            "Date": date,
            "Mean": mean_total,  # Mean of total cells
            "StDev": std_dev_total,  # Standard deviation of total cells
            "CoeffVar": coeff_var,  # Coefficient of variation of total cells
            "Total_225_flask": count_225,  # Number of cells in a 225 flask
            "Total_75_flask": count_75  # Number of cells in a 75 flask
        }
        
        # Add total cells for each image as separate columns
        for idx, total in enumerate(total_cells, start=1):
            summary_entry[f"Total_Image_{idx}"] = total
        
        daily_summary.append(summary_entry)
    
    return daily_summary, exact_times


# Function to predict future value based on doubling time
def predict_future_value(initial_mean, final_mean, initial_date, final_date, days_to_predict=1):
    """
    Predict future cell count based on the doubling time between two mean values.

    Parameters:
        initial_mean (float): Mean cell count at the initial date.
        final_mean (float): Mean cell count at the final date.
        initial_date (datetime.datetime): Initial date corresponding to initial_mean.
        final_date (datetime.datetime): Final date corresponding to final_mean.
        days_to_predict (int): Number of days to predict. Default is 1.

    Returns:
        list: List of predicted mean cell counts for future dates.
    """
    # Calculate doubling time in hours
    time_interval = (final_date - initial_date).total_seconds() / 3600  # Convert to hours
    doubling_time_hours = (np.log(2) * time_interval) / np.log(final_mean / initial_mean)
    
    # Calculate future dates and predicted cell counts
    next_date = final_date + timedelta(days=1)
    future_dates = [next_date + timedelta(days=i) for i in range(days_to_predict)]
    future_predictions = [final_mean * (2 ** ((i + 1) * 24 / doubling_time_hours)) for i in range(days_to_predict)]
    
    return future_dates, future_predictions


# Main function to run Streamlit app
def main():
    st.title("Cell Counter")
    # Instructions for inputting folder path
    st.markdown("### Instructions:")
    st.markdown("Upload your JPEG Images below.")
    
    # File upload and options
    uploaded_files = st.file_uploader("Choose JPEG images", type=["jpg", "jpeg"], accept_multiple_files=True)
    show_outlines = st.checkbox("Show Outlines")
    st.markdown("If multiple images across multiple days are shown tick below (if not just proceed without ticking this box):")
    plot_and_predict = st.checkbox("Plot and Predict Scores")
    
    if uploaded_files:
        st.markdown("### Uploaded Images:")
        
        # Dictionary to store images grouped by date
        images_by_date = defaultdict(list)
        
        for file in uploaded_files:
            # Check if the filename contains "10x"
            if "10x" in file.name:
                st.warning(f"Image '{file.name}' contains '10x' and will be ignored.")
                continue
            
            image = Image.open(file)
            image_date, exact_time = get_image_date_and_time(image)
            if image_date:
                images_by_date[image_date].append((image, exact_time))
        
        # Load Cellpose model
        model_path = r"./CP_20240320_105807"
        model = models.CellposeModel(pretrained_model=model_path)
        
        # Process images and display results
        if st.button("Process Images"):
            daily_summary, exact_times = process_images(images_by_date, model, show_outlines)
            
            # Convert daily summary to DataFrame
            df_daily_summary = pd.DataFrame(daily_summary)
            
            # Display DataFrame as table
            st.markdown("### Daily Summary Table")
            st.dataframe(df_daily_summary)
            
            # Plot mean cell counts over time if selected
            if plot_and_predict:
                # Plot mean cell counts over time
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df_daily_summary['Date'], df_daily_summary['Mean'], marker='o', linestyle='-', color='b', label='Mean Cell Count')
                ax.errorbar(df_daily_summary['Date'], df_daily_summary['Mean'], yerr=df_daily_summary['StDev'], fmt='o', color='b', alpha=0.5)
                
                # Predict future value
                if len(df_daily_summary) >= 2:
                    initial_mean = df_daily_summary['Mean'].iloc[-2]
                    final_mean = df_daily_summary['Mean'].iloc[-1]
                    initial_date = df_daily_summary['Date'].iloc[-2]
                    final_date = df_daily_summary['Date'].iloc[-1]
                    
                    days_to_predict = 5  # Number of days to predict
                    future_dates, future_predictions = predict_future_value(initial_mean, final_mean, initial_date, final_date, days_to_predict)
                    
                    # Plot predicted values
                    ax.plot(future_dates, future_predictions, marker='o', linestyle='--', color='g', label='Predicted Cell Count')
                    
                    # Add dashed grey line between mean and predicted values
                    ax.plot([final_date, future_dates[0]], [final_mean, future_predictions[0]], linestyle='--', color='grey')
                
                ax.set_ylabel('Mean Cell Count')
                ax.set_title('Mean Cell Counts Over Time')
                ax.legend()
                
                # Rotate the date labels vertically
                ax.tick_params(axis='x', rotation=90)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
