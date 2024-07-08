import os
import cv2
import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from math import pi
from datetime import datetime, timedelta
from PIL import Image
from PIL.ExifTags import TAGS
from cellpose import models
from cellpose.io import imread
from cellpose import io, utils




st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to split an image into subpanels
def split_image_into_subpanels(image, input_path, horizontal_offset, vertical_offset, num_rows, num_cols):
    # Get the filename of the input image
    filename = os.path.basename(input_path)
    # Remove the file extension from the filename
    filename_no_ext = os.path.splitext(filename)[0]
    
    # Get the directory of the input image
    input_dir = os.path.dirname(input_path)
    
    # Create the output directory with the image filename within the input directory
    output_dir = os.path.join(input_dir, f"{filename_no_ext}_subpanels")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the dimensions of the input image
    height, width = image.shape[:2]
    
    # Calculate the size of each subpanel (square panels)
    panel_size = min(height, width) // 8  # Ensure panels fit within circular region
    
    # Calculate the center of the image
    center_x = width // 2
    center_y = height // 2
    
    # Calculate the total width and height of the subpanels
    total_width = num_cols * panel_size + (num_cols - 1) * horizontal_offset
    total_height = num_rows * panel_size + (num_rows - 1) * vertical_offset
    
    # Calculate the starting coordinates for the top-left subpanel
    start_x = center_x - total_width // 2
    start_y = center_y - total_height // 2
    
    # Create a copy of the original image to draw outlines on
    image_with_outlines = image.copy()
    
    # Draw outlines for all subpanels
    for row in range(num_rows):
        for col in range(num_cols):
            # Calculate the coordinates of the current subpanel
            x1 = start_x + col * (panel_size + horizontal_offset)
            y1 = start_y + row * (panel_size + vertical_offset)
            
            # Draw outline around the subpanel on the original image
            cv2.rectangle(image_with_outlines, (x1, y1), (x1 + panel_size, y1 + panel_size), (0, 255, 0), 2)
            
    # Draw outline for the top-left subpanel separately
    cv2.rectangle(image_with_outlines, (start_x, start_y), (start_x + panel_size, start_y + panel_size), (0, 255, 0), 2)
    
    # Loop through each subpanel row
    for row in range(num_rows):
        # Calculate the y-coordinate of the current row of subpanels
        y1 = start_y + row * (panel_size + vertical_offset)
        
        # Loop through each subpanel column
        for col in range(num_cols):
            # Calculate the x-coordinate of the current column of subpanels
            x1 = start_x + col * (panel_size + horizontal_offset)
            
            # Ensure subpanel fits within circular region
            distance = np.sqrt((center_x - x1)**2 + (center_y - y1)**2)
            if distance + panel_size / 2 > min(center_x, center_y):
                continue
            
            # Extract the current subpanel from the image
            subpanel = image[y1:y1+panel_size, x1:x1+panel_size]
            
            # Generate the filename for the subpanel
            filename = os.path.join(output_dir, f"subpanel_{row}_{col}.png")
            
            # Save the subpanel as a separate image
            cv2.imwrite(filename, subpanel)
    
    # Save the top-left subpanel separately
    cv2.imwrite(os.path.join(output_dir, f"subpanel_0_0.png"), image[start_y:start_y+panel_size, start_x:start_x+panel_size])
    
    return image_with_outlines

# Function to count cells in a folder
def count_cells_in_folder(folder_path, model, output_dir):
    total_cell_count = 0
    
    # Create the output directory if it doesn't exist
    outlines_output_dir = os.path.join(output_dir, os.path.basename(folder_path) + "_cell_outlines")
    os.makedirs(outlines_output_dir, exist_ok=True)
    
    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # Load the image
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Count cells in the grayscale image using Cellpose model
            masks, _, _ = model.eval(gray_image, diameter=None)
            num_cells = masks.max()
            outlines_cells = utils.outlines_list(masks)
            
            # Overlay outlines on the original input image
            image_with_outlines = image.copy()
            for outline in outlines_cells:
                for i in range(len(outline) - 1):
                    cv2.line(image_with_outlines, tuple(outline[i]), tuple(outline[i + 1]), (0, 255, 0), 2)
            
            # Save the image with outlines in the output directory
            output_image_path = os.path.join(outlines_output_dir, filename)
            cv2.imwrite(output_image_path, image_with_outlines)
            
            # Update total cell count
            total_cell_count += num_cells
            
    
    return total_cell_count


def predict_cell_counts(input_folders):
    total_cell_counts = []
    
    # Load your Cellpose model
    model_path = r"\\FS\fs\Felix\cellected_code\Cellected_work\flask_counter_cellpose\models\CP_20240320_105807"
    model = models.CellposeModel(pretrained_model=model_path)
    
    # Iterate through each input folder
    for folder_path in input_folders:
        folder_cell_counts = []
        
        # Iterate through each image file in the input folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                # Load the image
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                
                # Split the image into subpanels
                split_image_into_subpanels(image, image_path, horizontal_offset=50, vertical_offset=50, num_rows=5, num_cols=5)
                
                # Get the path to the subpanels folder
                subpanels_folder = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}_subpanels")
                
                # Count cells in the subpanels folder
                total_cell_count = count_cells_in_folder(subpanels_folder, model, folder_path)
                folder_cell_counts.append(total_cell_count)

        # Calculate the mean cell count for the folder
        mean_cell_count = sum(folder_cell_counts) / len(folder_cell_counts)
        total_cell_counts.append(folder_cell_counts + [mean_cell_count])

    return total_cell_counts



def get_image_datetime_original(image_path):
    """
    Get the DateTimeOriginal metadata from an image.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        str: DateTimeOriginal metadata value.
    """
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data is not None:
                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag)
                    if tag_name == "DateTimeOriginal":
                        return value
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def find_image_in_folder(folder_path):
    """
    Recursively find an image file within a folder and its subfolders.

    Parameters:
        folder_path (str): Path to the folder.

    Returns:
        str: Path to the first image file found, or None if no image found.
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                return os.path.join(root, file)
    return None

def create_cell_count_dataframe(predicted_cell_counts, input_folders):
    data = {}
    for folder_counts, folder_path in zip(predicted_cell_counts, input_folders):
        folder_name = os.path.basename(folder_path)  # Extract the folder name from the path
        image_path = find_image_in_folder(folder_path)  # Find image within folder
        
        if image_path:
            datetime_original = get_image_datetime_original(image_path)  # Get DateTimeOriginal
            
            cell_counts = folder_counts[:-1]  # Exclude the mean cell count
            
            # Calculate mean, standard deviation, and predicted cell counts
            mean_count = np.mean(cell_counts)
            std_dev = np.std(cell_counts)
            count_225 = mean_count * 13050.0146
            count_75 = count_225 / 3
            
            # Calculate coefficient of variance (CV)
            cv = std_dev / mean_count if mean_count != 0 else 0
            
            # Calculate range
            cell_range = np.max(cell_counts) - np.min(cell_counts)
            
            data[folder_name] = {
                "Mean": mean_count,
                "Standard Deviation": std_dev,
                "Coefficient of Variance": cv,
                "Range": cell_range,
                "Predicted Count (225)": count_225,
                "Predicted Count (75)": count_75,
                "DateTimeOriginal": datetime_original,
                **{f"Image {i}": count for i, count in enumerate(cell_counts, start=1)}
            }
    
    # Create DataFrame
    cell_count_df = pd.DataFrame(data).T
    
    # Convert 'DateTimeOriginal' column to datetime objects with specified format
    cell_count_df['DateTimeOriginal'] = pd.to_datetime(cell_count_df['DateTimeOriginal'], format='%Y:%m:%d %H:%M:%S')
    
    # Sort DataFrame based on 'DateTimeOriginal' column in ascending order
    cell_count_df.sort_values(by='DateTimeOriginal', inplace=True)
    
    return cell_count_df






def get_subfolders(folder_path):
    if os.path.isdir(folder_path):
        # If it's a directory, check if it contains subfolders
        subfolders = [os.path.join(folder_path, name) for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
        if subfolders:
            # If subfolders are found, return them
            return subfolders
        else:
            # If no subfolders found, return the folder itself as a single-item list
            return [folder_path]
    else:
        # If it's not a directory, return the folder itself as a single-item list
        return [folder_path]



def count_photos_in_folders(folder_paths):
    total_photos = 0
    for folder_path in folder_paths:
        for _, _, files in os.walk(folder_path):
            total_photos += len([file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))])
    return total_photos



def estimate_blur(image):
    """
    Estimate the blur level of an image using the variance of Laplacian (LoG) operator.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        float: Blur level of the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance

def estimate_gradient_magnitude(image, square_size = 2000):
    """
    Estimate the gradient magnitude within a square region in the middle of an image
    using the Sobel operator.

    Parameters:
        image (numpy.ndarray): Input image.
        square_size (int): Size of the square region.

    Returns:
        float: Gradient magnitude within the square region of the image.
    """
    # Get the dimensions of the image
    height, width = image.shape[:2]
    
    # Calculate the coordinates of the square region
    top = (height - square_size) // 2
    bottom = top + square_size
    left = (width - square_size) // 2
    right = left + square_size
    
    # Extract the square region from the image
    square_region = image[top:bottom, left:right]
    
    # Convert the square region to grayscale
    gray = cv2.cvtColor(square_region, cv2.COLOR_BGR2GRAY)
    
    # Compute gradients using the Sobel operator
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    gradient_magnitude = cv2.magnitude(sobelx, sobely)
    
    # Calculate the mean gradient magnitude within the square region
    mean_gradient_magnitude = gradient_magnitude.mean()
    
    return mean_gradient_magnitude



def create_quality_control_dataframe(input_folders):
    data = {}
    for folder_path in input_folders:
        blur_scores = []
        gradient_magnitude_scores = []
        photo_names = []
        for filename in os.listdir(folder_path):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                blur_level = estimate_blur(image)
                gradient_magnitude = estimate_gradient_magnitude(image)
                blur_scores.append(blur_level)
                gradient_magnitude_scores.append(gradient_magnitude)
                photo_names.append(os.path.splitext(filename)[0])  # Remove file extension for photo name
        photo_data = {photo_names[i]: {"Blur Level": blur_scores[i], "Gradient Magnitude": gradient_magnitude_scores[i]}
                      for i in range(len(photo_names))}
        data.update(photo_data)
    quality_control_df = pd.DataFrame(data).T
    
    # Ensure that "Blur Level" column contains only numeric values
    quality_control_df["Blur Level"] = quality_control_df["Blur Level"].apply(lambda x: np.mean(x) if isinstance(x, list) else x)
    
    # Ensure that "Gradient Magnitude" column contains only numeric values
    quality_control_df["Gradient Magnitude"] = quality_control_df["Gradient Magnitude"].apply(lambda x: np.mean(x) if isinstance(x, list) else x)
    
    return quality_control_df





def parse_datetime(datetime_original):
    """
    Parse the DateTimeOriginal string into a datetime object.

    Parameters:
        datetime_original (str): DateTimeOriginal string.

    Returns:
        datetime: Parsed datetime object.
    """
    return datetime.strptime(datetime_original, '%Y:%m:%d %H:%M:%S')



def plot_mean_cell_count_over_time(dataframe, degree=2):
    """
    Plot mean cell count over time from a DataFrame containing 'DateTimeOriginal' and 'Mean' columns
    with error bars representing the standard deviation and a curved line of best fit.

    Parameters:
        dataframe (pandas.DataFrame): DataFrame containing 'DateTimeOriginal', 'Mean', and 'Standard Deviation' columns.
        degree (int): Degree of the polynomial for the curve fitting. Default is 2.

    Returns:
        None
    """
    # Convert 'DateTimeOriginal' column to datetime objects with specified format
    dataframe['DateTimeOriginal'] = pd.to_datetime(dataframe['DateTimeOriginal'], format='%Y:%m:%d %H:%M:%S')
    
    # Sort dataframe by datetime
    dataframe.sort_values(by='DateTimeOriginal', inplace=True)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.errorbar(dataframe['DateTimeOriginal'], dataframe['Mean'], yerr=dataframe['Standard Deviation'], 
                 marker='o', linestyle='-', capsize=5)  # Add error bars with standard deviation
    plt.xlabel('DateTimeOriginal')
    plt.ylabel('Mean Cell Count')
    plt.title('Mean Cell Count Over Time with Standard Deviation Error Bars')

    # Polynomial regression
    x = dataframe['DateTimeOriginal'].astype(np.int64) // 10**9  # Convert datetime to seconds since epoch
    y = dataframe['Mean']
    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(x.values.reshape(-1, 1))
    model = LinearRegression()
    model.fit(x_poly, y)
    y_pred = model.predict(x_poly)

    # Plot the curve of best fit
    plt.plot(dataframe['DateTimeOriginal'], y_pred, color='red', label=f'Polynomial Regression (Degree {degree})')

    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Display the plot using Streamlit
    st.pyplot()




def calculate_doubling_rate_from_dataframe(dataframe):
    """
    Calculate the doubling rate of cell count from a DataFrame.

    Parameters:
        dataframe (pandas.DataFrame): DataFrame containing 'DateTimeOriginal' and 'Mean' columns.

    Returns:
        doubling_rate (float): Doubling rate in time units (e.g., hours).
    """
    # Convert 'DateTimeOriginal' column to datetime objects with specified format
    dataframe['DateTimeOriginal'] = pd.to_datetime(dataframe['DateTimeOriginal'], format='%Y:%m:%d %H:%M:%S')

    # Sort DataFrame by 'DateTimeOriginal' column to get the oldest and newest counts
    dataframe = dataframe.sort_values(by='DateTimeOriginal')

    # Get the initial and final counts
    initial_count = dataframe['Mean'].iloc[0]
    final_count = dataframe['Mean'].iloc[-1]

    # Get the oldest and newest dates
    oldest_date = dataframe['DateTimeOriginal'].iloc[0]
    newest_date = dataframe['DateTimeOriginal'].iloc[-1]

    # Calculate the time interval between the initial and final counts
    time_interval = (newest_date - oldest_date).total_seconds() / 3600  # Convert to hours

    # Calculate the doubling rate
    doubling_rate = (np.log(2) * time_interval) / np.log(final_count / initial_count)

    st.write("doubling rate (hours): ", doubling_rate)
    return doubling_rate


def plot_mean_cell_count_over_time(dataframe, days_to_predict=5):
    """
    Plot mean cell count over time from a DataFrame containing 'DateTimeOriginal', 'Mean', and 'Standard Deviation' columns
    with error bars representing the standard deviation and predictions based on doubling time.

    Parameters:
        dataframe (pandas.DataFrame): DataFrame containing 'DateTimeOriginal', 'Mean', and 'Standard Deviation' columns.
        days_to_predict (int): Number of days to predict. Default is 5.

    Returns:
        matplotlib.figure.Figure: Matplotlib figure object containing the plot.
    """
    # Convert 'DateTimeOriginal' column to datetime format
    dataframe['DateTimeOriginal'] = pd.to_datetime(dataframe['DateTimeOriginal'], format='%Y-%m-%dT%H:%M:%S.000')

    # Sort dataframe by datetime
    dataframe.sort_values(by='DateTimeOriginal', inplace=True)

    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the data
    ax.errorbar(dataframe['DateTimeOriginal'], dataframe['Mean'], yerr=dataframe['Standard Deviation'], 
                marker='o', linestyle='-', capsize=5, label='Data')  # Add error bars with standard deviation
    ax.set_xlabel('DateTimeOriginal')
    ax.set_ylabel('Mean Cell Count')
    ax.set_title('Mean Cell Count Over Time with Standard Deviation Error Bars')

    # Calculate doubling time using the second last and last data points
    initial_count = dataframe['Mean'].iloc[-2]
    final_count = dataframe['Mean'].iloc[-1]
    initial_date = dataframe['DateTimeOriginal'].iloc[-2]
    final_date = dataframe['DateTimeOriginal'].iloc[-1]
    time_interval = (final_date - initial_date).total_seconds() / 3600  # Convert to hours
    doubling_time_hours = (np.log(2) * time_interval) / np.log(final_count / initial_count)

    # Calculate future dates and predicted cell counts
    last_date = dataframe['DateTimeOriginal'].iloc[-1]
    next_date = last_date + timedelta(hours=24)
    future_dates = [next_date + timedelta(days=i) for i in range(0, days_to_predict)]
    future_predictions = [dataframe['Mean'].iloc[-1] * (2 ** (i * 24 / doubling_time_hours)) for i in range(1, days_to_predict + 1)]

    # Plot the line between data and predicted points
    ax.plot([last_date, future_dates[0]], [dataframe['Mean'].iloc[-1], future_predictions[0]], linestyle='--', color='gray')

    # Plot the predicted values for future dates
    ax.plot(future_dates, future_predictions, marker='o', linestyle='-', color='green', label='Predicted')

    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()


    # Return the figure object
    return fig


def main():
    st.title("Cell Count Prediction App")

    # Instructions for inputting folder path
    st.markdown("### Instructions:")
    st.markdown("1. Copy the folder path (right-click folder with photos and copy as path).")
    st.markdown("2. Paste the folder path in the text box below.")
    st.markdown("3. Click the 'Start Processing' button to begin.")
    st.markdown("Enable 'Quality Control' to display image bluriness and gradient magnitude.")
    st.markdown("The doubling rate is calculated using the formula based on the oldest and newest dates. This is the measure from start to finish.")
    st.markdown("The plots show the mean cell counts over time. They predict using the doubling formula on the last two most recent counts.")
    st.markdown("Note: The plots are only meaningful when all the inputted folders are from the same colony.")

    # Input folder path using text input widget
    st.write("Enter the folder path:")
    folder_path = st.text_input("Folder Path")

    # Checkbox to enable quality control
    enable_quality_control = st.checkbox("Enable Quality Control")

    # Button to start processing
    if st.button("Start Processing"):
        if folder_path:
            folder_path = folder_path.strip('\"')
            input_folders = get_subfolders(folder_path)
            
            # Count total photos
            total_photos = count_photos_in_folders(input_folders)
            st.write(f"Total Photos to Analyze: {total_photos}")

            # Estimate total time required
            time_per_photo = 60  # seconds
            total_time = ((total_photos * time_per_photo)/60)
            st.write(f"Estimated Time Required: {total_time} minutes")

            # Perform cell count prediction
            with st.spinner('Processing...'):
                predicted_cell_counts = predict_cell_counts(input_folders)

                # Display the predicted cell counts
                for i, folder_counts in enumerate(predicted_cell_counts, start=1):
                    st.write(f"Folder {i} Cell Counts: {folder_counts[:-1]}, Mean: {folder_counts[-1]}")

                # Create DataFrame for cell counts
                cell_count_df = create_cell_count_dataframe(predicted_cell_counts, input_folders)
                st.write("Cell Count DataFrame:")
                st.write(cell_count_df)

                
                # Show progress
                photos_analyzed = sum(len(folder_counts) - 1 for folder_counts in predicted_cell_counts)
                st.write(f"Progress: {photos_analyzed} out of {total_photos} photos analyzed")

                calculate_doubling_rate_from_dataframe(cell_count_df)
                fig = plot_mean_cell_count_over_time(cell_count_df, days_to_predict=5)
                st.pyplot(fig)

                # Create DataFrame for quality control if enabled
                if enable_quality_control:
                    quality_control_df = create_quality_control_dataframe(input_folders)
                    st.write("Quality Control DataFrame:")
                    st.write(quality_control_df)

if __name__ == "__main__":
    main()

