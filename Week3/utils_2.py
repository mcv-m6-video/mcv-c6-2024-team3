import cv2
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter


def reset_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)

def save_frames(video_path, output_folder):
    # Check if the folder already exists
    reset_folder(output_folder)
    # Open the video
    video = cv2.VideoCapture(video_path)
    # Read the first frame
    success, image = video.read()
    # Counter to save the frames
    count = 0
    # While there are frames to read
    while success:
        # Save the frame
        cv2.imwrite(os.path.join(output_folder, 'frame' + f'{str(count).zfill(5)}.png'), image)
        # Read the next frame
        success, image = video.read()
        # Increase the counter
        count += 1
    # Close the video
    video.release()


def read_of(flow_path):

    flow_raw = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED).astype(np.double)

    # Transform data (DevKit Stereo Flow - KITTI)
    flow_u = (flow_raw[:,:,2] - 2**15) / 64.0
    flow_v = (flow_raw[:,:,1] - 2**15) / 64.0
    flow_valid = flow_raw[:,:,0] == 1

    # Set to 0 the points where the flow is not valid
    flow_u[~flow_valid] = 0
    flow_v[~flow_valid] = 0

    # Reorder channels
    return np.stack((flow_u, flow_v, flow_valid), axis=2)

def get_metrics(gt, flow):

    square_error_matrix = (flow[:,:,0:2] - gt[:,:,0:2]) ** 2
    square_error_matrix_valid = square_error_matrix*np.stack((gt[:,:,2],gt[:,:,2]),axis=2)

    non_occluded_pixels = np.sum(gt[:,:,2] != 0)

    # Compute MSEN
    pixel_error_matrix = np.sqrt(np.sum(square_error_matrix_valid, axis= 2)) # Pixel error for both u and v
    msen = (1/non_occluded_pixels) * np.sum(pixel_error_matrix) # Average error for all non-occluded pixels
    # Compute PEPN
    erroneous_pixels = np.sum(pixel_error_matrix > 3)
    pepn = (erroneous_pixels/non_occluded_pixels)*100
    
    return msen, pepn

def plot_dense_flow(flow, filename):

    #Calculate OF magnitude and angles
    flow_magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
    flow_direction = np.arctan2(flow[:,:,1], flow[:,:,0]) + np.pi

    #Clip the highest magnitude values according to the 0.95 quantile
    clip_th = np.quantile(flow_magnitude,0.95)
    flow_magnitude = np.clip(flow_magnitude,0,clip_th) 

    #Scale the magnitude so that it takes values within [0,255]
    flow_scaled_magnitude = (flow_magnitude/np.max(flow_magnitude))*255

    #Generate a visualization of the OF in the HSV space
    flow_hsv = np.zeros(flow.shape, dtype=np.uint8)
    flow_hsv[:,:,0] = flow_direction/(2*np.pi)*179
    flow_hsv[:,:,1] = flow_scaled_magnitude
    flow_hsv[:,:,2] = 255
    flow_bgr = cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2BGR)

    #Write images
    cv2.imwrite("figures/CV2/dense_" + filename + ".png", flow_bgr)

    
def plot_flow(flow, img, filename, step): 
    height = np.shape(flow[:,:,0])[0]
    width = np.shape(flow[:,:,0])[1]

    X, Y = np.meshgrid(np.arange(0, width, step), np.arange(0, height, step))

    U = flow[::step,::step,0]
    V = flow[::step,::step,1]
    M = np.hypot(U, V)
    
    # Plot
    plt.figure()
    plt.axis('off')
    plt.quiver(X, Y, U, V, M, color='red', angles='xy')
    plt.imshow(img)
    plt.savefig("figures/CV2/arrows_" + filename + ".png", bbox_inches='tight', pad_inches=0, dpi=250)

def error(image1, image2, function):
    
    if function == 'SSD':
        # Calculate SSD
        error = np.sum((image1 - image2)**2)
    elif function == 'SAD':
        # Calculate SAD
        error = np.sum(np.abs(image1 - image2))

    elif function == 'MSE':
        # Calculate MSE
        error = np.mean((image1 - image2) ** 2)

    elif function == 'MAD':
        # Calculate MAD
        error = np.mean(np.abs(image1 - image2))
    
    elif function == 'NCC':
        # Compute the means of the images
        mean_image1 = np.mean(image1)
        mean_image2 = np.mean(image2)

        # Compute the standard deviations of the images
        std_image1 = np.std(image1)
        std_image2 = np.std(image2)

        # Compute the cross-correlation between the images
        cross_corr = np.sum((image1 - mean_image1) * (image2 - mean_image2))

        # Compute the normalized cross-correlation coefficient
        error = -(cross_corr / (std_image1 * std_image2)) #Add the negative sign because the more similar the images, the higher NCC is, so we want to make it the opposite

    
    
    return error

def postprocessing(flow_met, block_size, smoothing = True):
    #Iterate through the blocks
    for i in range(0, flow_met.shape[0], block_size):
        for j in range(0, flow_met.shape[1], block_size):
            #Extract the current block
            block_x = flow_met[i:i+block_size, j:j+block_size, 0]
            block_y = flow_met[i:i+block_size, j:j+block_size, 1]

            #Get the OF value of the current block (as all the pixels have the same, we can take any position)
            block_median_x = block_x[0,0]
            block_median_y = block_y[0,0]

            # Compute the median values of neighboring blocks in both directions
            neighboring_blocks_x = []
            neighboring_blocks_y = []
            for m in range(max(0, i - block_size), min(flow_met.shape[0], i + block_size + 1), block_size):
                for n in range(max(0, j - block_size), min(flow_met.shape[1], j + block_size + 1), block_size):
                    if m != i or n != j:

                        neighboring_block_x = flow_met[m:m+block_size, n:n+block_size, 0]
                        neighboring_blocks_x.append(neighboring_block_x[0,0])

                        neighboring_block_y = flow_met[m:m+block_size, n:n+block_size, 1]
                        neighboring_blocks_y.append(neighboring_block_y[0,0])

            #Check if the block's value is significantly different from its neighbors median value in both directions
            if np.abs(block_median_x - np.median(neighboring_blocks_x)) > 5:
                # Interpolate the value of the current block based on its neighbors
                interpolated_value = np.mean(neighboring_blocks_x)
                flow_met[i:i+block_size, j:j+block_size, 0] = interpolated_value

            if np.abs(block_median_y - np.median(neighboring_blocks_y)) > 5:
                # Interpolate the value of the current block based on its neighbors
                interpolated_value = np.mean(neighboring_blocks_y)
                flow_met[i:i+block_size, j:j+block_size, 1] = interpolated_value
        
    if smoothing == True:
        flow_blurred = np.zeros_like(flow_met)
        for i in range(2):  # Apply median blur to each component of the flow separately
            flow_blurred[..., i] = median_filter(flow_met[..., i], 9)
            flow_blurred[..., i] = gaussian_filter(flow_blurred[..., i], sigma=9)
        # flow_met = median_filter(flow_met, size=5)
        flow_met = flow_blurred
    
    return flow_met