import cv2
import numpy as np

def estimate_waterbody(image, depth_map, percentile=5, scale_factor=1.2, region='upper'):
    # Convert image to float and normalize
    image_float = image.astype(np.float32) / 255.0
    depth_map = cv2.medianBlur(depth_map, 7)
    depth_flat = depth_map.flatten()
    image_flat = image_float.reshape(-1, 3)

    # Compute depth threshold dynamically based on image characteristics
    depth_threshold = np.percentile(depth_flat, percentile)
    farthest_depth_indices = np.where(depth_flat >= depth_threshold)[0]
    farthest_pixels = image_flat[farthest_depth_indices]

    if region == "entire":
        # Region-based selection: Entire image region
        # Create a mask that selects the entire image
        entire_region_mask = np.ones(image.shape[:2], dtype=bool) 
        # Flatten the entire region mask
        entire_region_mask_flat = entire_region_mask.flatten()
        # Apply the region-based mask to select only farthest pixels in the entire region
        farthest_pixels_entire_region = image_flat[np.intersect1d(farthest_depth_indices, np.where(entire_region_mask_flat)[0])]
        # Since you're selecting from the entire region, no need to combine from different corners
        farthest_pixels_combined = farthest_pixels_entire_region

    elif region == "upper":
        #Optionally Sometimes if waterbody prminent in upper regions
        # Region-based selection: Top-left and top-right corners (20% width, 20% height)
        height, width, _ = image.shape
        top_left_mask = np.zeros(image.shape[:2], dtype=bool)
        top_right_mask = np.zeros(image.shape[:2], dtype=bool)
        # Define the regions in the top left and top right corners
        top_left_mask[:int(0.2 * height), :int(0.2 * width)] = True
        top_right_mask[:int(0.2 * height), int(0.8 * width):] = True
        # Flatten the masks
        top_left_mask_flat = top_left_mask.flatten()
        top_right_mask_flat = top_right_mask.flatten()
        # Apply the region-based masks to select only farthest pixels in the corners
        farthest_pixels_top_left = image_flat[np.intersect1d(farthest_depth_indices, np.where(top_left_mask_flat)[0])]
        farthest_pixels_top_right = image_flat[np.intersect1d(farthest_depth_indices, np.where(top_right_mask_flat)[0])]
        # Combine the selected pixels from both corners
        farthest_pixels_combined = np.vstack([farthest_pixels_top_left, farthest_pixels_top_right])

    elif region == "bottom":
        # Optionally Sometimes if waterbody prminent in bottom regions
        # Region-based selection: Bottom-left and bottom-right corners (20% width, 20% height)
        height, width, _ = image.shape
        bottom_left_mask = np.zeros(image.shape[:2], dtype=bool)
        bottom_right_mask = np.zeros(image.shape[:2], dtype=bool)
        # Define the regions in the bottom left and bottom right corners
        bottom_left_mask[int(0.8 * height):, :int(0.2 * width)] = True
        bottom_right_mask[int(0.8 * height):, int(0.8 * width):] = True
        # Flatten the masks
        bottom_left_mask_flat = bottom_left_mask.flatten()
        bottom_right_mask_flat = bottom_right_mask.flatten()
        # Apply the region-based masks to select only farthest pixels in the corners
        farthest_pixels_top_left = image_flat[np.intersect1d(farthest_depth_indices, np.where(bottom_left_mask_flat)[0])]
        farthest_pixels_top_right = image_flat[np.intersect1d(farthest_depth_indices, np.where(bottom_right_mask_flat)[0])]
        # Combine the selected pixels from both corners
        farthest_pixels_combined = np.vstack([farthest_pixels_top_left, farthest_pixels_top_right])
    

    else:
        raise ValueError(f"Invalid region specified: {region}. Choose 'entire', 'upper', or 'bottom'.")

    # If not enough farthest pixels from the corners, fall back to the entire farthest pixel set
    if len(farthest_pixels_combined) < 100:
        farthest_pixels_combined = farthest_pixels
    # Apply a color filter to remove non-background colors
    lower_color_bound = np.array([0.0, 0.2, 0.3])  # Lower bound: low red, moderate green, more blue
    upper_color_bound = np.array([0.9, 1.0, 1.0])  # Upper bound: higher red to include yellows, max green and blue

    color_filtered_pixels = farthest_pixels_combined[
        np.all((farthest_pixels_combined >= lower_color_bound) & (farthest_pixels_combined <= upper_color_bound), axis=1)
    ]

    # If not enough color-filtered pixels, relax the threshold or fallback to a broader range
    if len(color_filtered_pixels) < 50:
        color_filtered_pixels = farthest_pixels_combined

    # If still not enough pixels, fallback to average color of the whole image
    if len(color_filtered_pixels) == 0:
        return np.mean(image_float, axis=(0, 1))

    # Estimate the background light as the median color of farthest pixels
    waterbody = np.median(color_filtered_pixels, axis=0)

        # Scale the background light to make it more prominent
    waterbody_scaled = waterbody * scale_factor

    return waterbody_scaled