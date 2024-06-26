from utils_2 import *
import time


def block_matching(path_cur, path_pre, block_size = 16, mode = 'backward', aos = 32, error_function = "cv2_TM_CCORR_NORMED", step_size = 1):
    prev_img = cv2.imread(path_pre)
    prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY) 
    # copy = prev_img.copy()
    curr_img = cv2.imread(path_cur)
    curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY) 

    if mode == "backward":
        im_ref = curr_img_gray
        im_ref_rgb = curr_img
        im_target = prev_img_gray
        # im_target_rgb = prev_img
    elif mode == "forward":
        im_ref = prev_img_gray
        im_ref_rgb = prev_img
        im_target = curr_img_gray
        # im_target_rgb = curr_img

    # Initialize the optical flow algorithm
    flow_met = np.zeros((im_ref.shape[0], im_ref.shape[1],3), dtype=float) 
    flow_plot = np.zeros((im_ref.shape[0], im_ref.shape[1],3), dtype=float) #We want this to plot the real optical flow

    start_time = time.time()
    #We iterate through all the blocks in  the previous frame:
    for y in range(0, im_ref.shape[0]-block_size, block_size):
        for x in range(0, im_ref.shape[1]-block_size, block_size):
            
            block = im_ref[y:y+block_size, x:x+block_size]

            min_dist = None
            flowx, flowy = 0, 0

            #Define the search window
            x_search_left = max(x-aos, 0)
            x_search_right = min(x+block_size+aos, im_ref.shape[1])

            y_search_up = max(y-aos, 0)
            y_search_down = min(y+block_size+aos, im_ref.shape[0])

            #Compare the reference block to those inside the area of search, find the best match and store the coordinates
            if 'cv2' not in error_function:
                for j in range(y_search_up, y_search_down-block_size, step_size):
                    for i in range(x_search_left, x_search_right-block_size, step_size):
                        # print(i,j)
                        block_post = im_target[j:j+block_size, i:i+block_size]
                        
                        #Compute the distance and update minimum.
                        dist = error(block, block_post, error_function)
                        if not min_dist or dist < min_dist:
                            min_dist = dist

                            #Adjust position to image scale
                            flowx = i - x 
                            flowy = j - y
            else:
                search_area = im_target[y_search_up:y_search_down, x_search_left:x_search_right]
                
                if error_function == "cv2_TM_CCOEFF": 
                    diff = cv2.matchTemplate(search_area, block, cv2.TM_CCOEFF)
                elif error_function == "cv2_TM_CCOEFF_NORMED": 
                    diff = cv2.matchTemplate(search_area, block, cv2.TM_CCOEFF_NORMED)
                elif error_function == "cv2_TM_CCORR": 
                    diff = cv2.matchTemplate(search_area, block, cv2.TM_CCORR)
                elif error_function == "cv2_TM_CCORR_NORMED": 
                    diff = cv2.matchTemplate(search_area, block, cv2.TM_CCORR_NORMED)
                elif error_function == "cv2_TM_SQDIFF": 
                    diff = cv2.matchTemplate(search_area, block, cv2.TM_SQDIFF)
                elif error_function == "cv2_TM_SQDIFF_NORMED": 
                    diff = cv2.matchTemplate(search_area, block, cv2.TM_SQDIFF_NORMED)
                
                
                #Find the minimum difference and its position
                _, _, min_loc, max_loc = cv2.minMaxLoc(diff)

                if "cv2_TM_SQDIFF" in error_function:
                    loc = min_loc
                else: 
                    loc = max_loc

                #Adjust position to image scale
                flowx = loc[0] - (x-x_search_left) 
                flowy = loc[1] - (y-y_search_up)

            #Store the coordinates of the optical flow
            for dy in range(y, y+block_size):
                for dx in range(x, x+block_size):
                    if mode == 'backward':
                        flow_met[dy, dx, 0] = -flowx
                        flow_met[dy, dx, 1] = -flowy
                        flow_met[dy, dx, 2] = 1
                    else:
                        flow_met[dy, dx, 0] = flowx
                        flow_met[dy, dx, 1] = flowy
                        flow_met[dy, dx, 2] = 1 
            
            # Draw the displacement vector
            # cv2.arrowedLine(copy, (x + block_size // 2, y + block_size // 2),
            #                 (x + flowx + block_size // 2, y + flowy + block_size // 2),
            #                 (0, 255, 0), 1)

    flow_met = postprocessing(flow_met, block_size)
    
    if mode == 'backward': #Get the plotting values (real ones)
        flow_plot = -flow_met

    return flow_met


if __name__ == "__main__":
    
    block_matching()