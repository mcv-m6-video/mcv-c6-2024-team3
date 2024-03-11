from utils import *
import wandb
import time

wandb.login(key='4127c8a2b851657f629b6f8f83ddc2e3415493f2')  # IKER

sweep_config = {
    'method': 'random',
    'metric': {'goal': 'minimize', 'name': 'MSE'},
    'parameters': {
        'mode' : {
            'values': ['backward', 'forward']
        },

        'block_size': {
            'values': [4,8,16,24,48,64,128]
        },

        'aos': {
            'values': [4,8,16,24,48,64,128]
        },

        'step_size': {
            'values': [1,3,5,7,9]
        },

        'error_function': {
            'values': ["cv2_TM_CCOEFF", "cv2_TM_CCOEFF_NORMED", "cv2_TM_CCORR", "cv2_TM_CCORR_NORMED", "cv2_TM_SQDIFF", "cv2_TM_SQDIFF_NORMED"]
        },
    }
 }

# ['SSD', 'SAD', 'MSE', 'MAD','NCC']

def block_matching(config=None):
    
    with wandb.init(config=config):
        config = wandb.config

        block_size = config.block_size
        mode = config.mode
        aos = config.aos
        error_function = config.error_function
        step_size = config.step_size

        prev_img = cv2.imread('/ghome/group02/C6/Week3/data_stereo_flow/training/colored_0/000045_10.png')
        prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY) 
        # copy = prev_img.copy()
        curr_img = cv2.imread('/ghome/group02/C6/Week3/data_stereo_flow/training/colored_0/000045_11.png')
        curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY) 


        gt = read_of('/ghome/group02/C6/Week3/data_stereo_flow/training/flow_noc/000045_10.png')


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
        flow_plot = np.zeros((im_ref.shape[0], im_ref.shape[1],3), dtype=float)
        # flow_extra = np.zeros_like(prev_img)

        start_time = time.time()
        #We iterate through all the blocks in  the previous frame:
        for y in range(0, im_ref.shape[0]-block_size, block_size):
            for x in range(0, im_ref.shape[1]-block_size, block_size):
                
                block = im_ref[y:y+block_size, x:x+block_size]

                min_dist = None
                flowx, flowy = 0, 0

                # Define the search window
                x_search_left = max(x-aos, 0)
                x_search_right = min(x+block_size+aos, im_ref.shape[1])

                y_search_up = max(y-aos, 0)
                y_search_down = min(y+block_size+aos, im_ref.shape[0])

                # Compare each block of the next frame to each block from a greater
                # region with the same center in the previous frame.
                if 'cv2' not in error_function:
                    for j in range(y_search_up, y_search_down-block_size, step_size):
                        for i in range(x_search_left, x_search_right-block_size, step_size):
                            # print(i,j)
                            block_post = im_target[j:j+block_size, i:i+block_size]
                            
                            # Compute the distance and update minimum.
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
                    
                    
                    # Find the minimum difference and its position
                    _, _, min_loc, max_loc = cv2.minMaxLoc(diff)

                    if "cv2_TM_SQDIFF" in error_function:
                        loc = min_loc
                    else: 
                        loc = max_loc


                    flowx = loc[0] - (x-x_search_left) 
                    flowy = loc[1] - (y-y_search_up)

                if mode == 'backward':
                    flow_met[y:y + block_size, x:x + block_size, :] = [-flowx, -flowy, 1]
                else:
                    flow_met[y:y + block_size, x:x + block_size, :] = [flowx, flowy, 1]
                
                flow_plot[y:y + block_size, x:x + block_size, :] = [flowx, flowy, 1]

                
                
                # Draw the displacement vector
                # cv2.arrowedLine(copy, (x + block_size // 2, y + block_size // 2),
                #                 (x + flowx + block_size // 2, y + flowy + block_size // 2),
                #                 (0, 255, 0), 1)

        end_time = time.time()
        execution_time = end_time - start_time

        filename = f"mode_{mode}_bs_{block_size}_aos_{aos}_ef_{error_function}"
        plot_dense_flow(flow_plot, filename)
        plot_flow(flow_plot, im_ref_rgb, filename, block_size)

        mse,pepn = get_metrics(gt, flow_met)
        print(f"Block_flow metrics, MSE:{mse}, PEPN:{pepn}, TIME:{execution_time}")

        wandb.log({'MSEN': mse, 'PEPN': pepn, 'TIME': execution_time})
        # flow = flow.astype(original_image.dtype)
        # cv2.imwrite('flow_2.jpg', copy)
        print('Finished!')

if __name__ == "__main__":
    
    sweep_id = wandb.sweep(sweep_config, project="C6-Week3-CV2")
    wandb.agent(sweep_id, function=block_matching, count=300)