from get_gps_coordinates import get_gps_coordinates
from sklearn.metrics.pairwise import haversine_distances
import pandas as pd
import numpy as np
import cv2
import os
from itertools import chain
import time
import pickle  
from pathlib import Path


"""
Projectes de millora:
- fer reid inter-frame appart from intra-frame
- velocity difference
- optimitzar els bins del histogram
"""

def parse(df):
    print("saving individual files")
    for unique_value in df.iloc[:, 1].unique():
        # Create a new file for each unique value
        filename = f"s04_{unique_value}.txt"  # Or any other file extension you prefer
        with open(filename, 'w') as file:
            # Write rows corresponding to the current unique value to the file
            for index, row in df[df.iloc[:, 1] == unique_value].iterrows():
                file.write(','.join(map(str, row)) + '\n')




def appearance_distance(row1, row2):
    #tlx, tly, w, h --> suposem qeu el format es aixi, si peta pot ser que sigui el problema

    bb1, path1 = row1
    bb2, path2 = row2
    
    img1 = cv2.imread(str(path1))
    img2 = cv2.imread(str(path2))

    bb1_vals = img1[int(bb1[1]):int(bb1[1]+bb1[3]), int(bb1[0]):int(bb1[0]+bb1[2])] 
    histogram1 = cv2.calcHist([bb1_vals], [0, 1, 2], None, [32,32,32], [0, 256, 0, 256, 0, 256])
    bb2_vals = img2[int(bb2[1]):int(bb2[1]+bb2[3]), int(bb2[0]):int(bb2[0]+bb2[2])]
    histogram2 = cv2.calcHist([bb2_vals], [0, 1, 2], None, [32,32,32], [0, 256, 0, 256, 0, 256])
    if np.sum(histogram1) == 0:
        print("problema joder! 1")
        print(bb1)
        cv2.imwrite("bb1_vals_image.jpg", bb1_vals)
    if np.sum(histogram2) == 0:
        print(bb2)
        print("problema joder! 2")
        cv2.imwrite("bb2_vals_image.jpg", bb2_vals)

    histogram1 /= np.sum(histogram1)
    histogram2 /= np.sum(histogram2)

    return cv2.compareHist(histogram1, histogram2, cv2.HISTCMP_INTERSECT)



def reid(sequence_tracks, dist_thr, appearance_thr):
    """
    Input: pandas df
    FIlters by appearance and distance, if combi in both, i assume they are the same car
    Output: tracking in mot challenge
    """
    min_frame = sequence_tracks['frame'].min()
    max_frame = sequence_tracks['frame'].max()
    seqs_no_remove = sequence_tracks.copy()
    
    for frame in range(min_frame,max_frame +1):
        group = sequence_tracks[sequence_tracks['frame'] == frame]

        possible_unions = {}
        print("FRAME", frame)
        for i, row1 in group.iterrows():
            for j, row2 in group.iterrows():
                if (i != j) and (row1['camera'] != row2['camera']) and (row1['new_id'] != -1) and (row2['new_id'] != -1) and (row2['new_id'] != row1['new_id'] ): # if not same element and from different cameras
                    
                    dist_spa = haversine_distances([row1['gps'], row2['gps']])[0][1] # calculate spatial distance

                    if dist_spa < dist_thr: # first calc the spatial distance, because appearance costs more and then we filter out
                        dist_app = appearance_distance(row1[['bbox', 'path_name']], row2[['bbox', 'path_name']]) # calculate appearance distance
                        

                        if (dist_app > appearance_thr):
                            if ((i,j) not in possible_unions) and ((j,i) not in possible_unions):
                                
                                possible_unions[(i,j)] = [dist_spa, dist_app]

        if len(possible_unions) != 0:
            # UPDATE DF WITH NEW MERGINGS
            
            # 1. match: since several cameras can have the same car, we consider there can be a lot of mergings into 1
            # In furher research, this could be improved
            # 2. merge: given sets of indices, AND REMOVE THE ONES FROM THE OVERLAPPING

            id_already_used = []
            for k in possible_unions.keys():
                id1, id2 = k
                new_id_1 = sequence_tracks.at[id1, 'new_id']
                camera_1 = sequence_tracks.at[id1, 'camera']

                new_id_2 = sequence_tracks.at[id2, 'new_id']
                camera_2 = sequence_tracks.at[id2, 'camera']


                if (new_id_1 not in id_already_used) and (new_id_2 not in id_already_used):
                    id_already_used.append(new_id_1)
                    id_already_used.append(new_id_2)

                    rows_1 = sequence_tracks[(sequence_tracks['new_id'] == new_id_1) & (sequence_tracks['camera'] == camera_1)]
                    rows_2 = sequence_tracks[(sequence_tracks['new_id'] == new_id_2) & (sequence_tracks['camera'] == camera_2)]

                    if len(rows_1) > len(rows_2):
                        # keep the detections from the first and remove the others, well, check the ids if there is some place where they don't have it
                        sequence_tracks.loc[rows_2.index, 'new_id'] = -1
                        # Get the values of the column "frame_id" for rows_1 and rows_2
                        frame_ids_1 = set(rows_1['frame'])

                        # Find the indices where values in rows_2 are not in rows_1
                        indices_not_in_rows_1 = rows_2[~rows_2['frame'].isin(frame_ids_1)].index
                        sequence_tracks.loc[indices_not_in_rows_1, 'new_id'] = new_id_1
                        seqs_no_remove.loc[indices_not_in_rows_1, 'new_id'] = new_id_1
                    else:
                        sequence_tracks.loc[rows_1.index, 'new_id'] = -1
                        
                        frame_ids_2 = set(rows_2['frame'])
                        indices_not_in_rows_2 = rows_1[~rows_1['frame'].isin(frame_ids_2)].index
                        sequence_tracks.loc[indices_not_in_rows_2, 'new_id'] = new_id_2
                        seqs_no_remove.loc[indices_not_in_rows_2, 'new_id'] = new_id_2



    filtered_df = sequence_tracks[sequence_tracks['new_id'] != -1]
    sorted_df = filtered_df.sort_values(by='frame', ascending=True)

    s_to_rem_sorted = seqs_no_remove.sort_values(by='frame', ascending=True)

    parse(s_to_rem_sorted)
    return sorted_df

def to_mot(df):
    """
    Input: pd df of frame, camera, track_id, gps, bbox, path_name, new_id
    Output: single txt file in mot format
    """
    df.to_csv('s04_all.txt', sep='\t', index=False, header=False)
    # df.to_csv('seq1_all_def.txt', sep='\t', index=False, header=False)



if __name__ == '__main__':

    start_time = time.time()
    sequence = "S04Track"
    sequence_tracks = get_gps_coordinates(sequence)
    sequence_tracks= pd.DataFrame(sequence_tracks)

    # these thresolds have to be optimized
    dist_thr = 0.0001
    appearance_thr = 0.1

    print("Doing reid")
    start_time = time.time()

    dataframe = reid(sequence_tracks, dist_thr, appearance_thr)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time taken:", elapsed_time, "seconds")
    
    print("Reid finnished. Saving df as mot")

    to_mot(dataframe)

    print("Mot saved")

    