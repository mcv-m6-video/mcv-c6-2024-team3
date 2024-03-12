"""
Reads a tracking file in the MOTchallenge format, reads the tracked video and plots the rectangles over the corresponding objects, then saves the video.

Usage:
  plot_rectangles_video.py <trackFile> <inputVideo> <outputVideo>  [--downsampleVideo=<dv>] [--startFrame=<sf>] [--maxFrame=<mf>] [--frameStep=<fs>]
  plot_rectangles_video.py -h | --help

Example:
python plot_bb_trail.py /home/user/Documents/MASTER/C6/mcv-c6-2024-team3/Week2/tracking/tracking_DeepSort.txt /home/user/Documents/MASTER/C6/mcv-c6-2024-team3/Week2/vdo.avi /home/user/Documents/MASTER/C6/mcv-c6-2024-team3/Week2/video.mp4

Options:
  --downsampleVideo=<dv>         Downsample the output video by a factor of 2 [default: False]
  --startFrame=<sf>              First frame to process [default: 1]
  --maxFrame=<mf>                Stop processing at this frame [default: -1]
  --frameStep=<fs>               When using  a sequence of frames, the number of frames to skip [default: 0]
"""

from docopt import docopt
import cv2 as cv
import numpy as np
import pandas as pd
from distutils.util import strtobool
import sys
from moviepy.editor import VideoFileClip

def convert_to_gif(input_file, output_file, fps=5, scale=None):
    # Load the video clip
    clip = VideoFileClip(input_file)
    
    # Adjust the scale if provided
    if scale:
        clip = clip.resize(scale)
    
    # Write the GIF file with loop set to 0 for infinite looping
    clip.write_gif(output_file, fps=fps, loop=0)

# tested OK
def trackinfo (df:pd.DataFrame) -> pd.DataFrame:
    '''
    For each track, store the info of the start and end frames of the track. Return a pandas DataFrame with this info
    Parameters: 
      df             : DataFrame with the tracking information
    Return value:
      DataFrame with the info about the starting and ending frames of each track
    '''
    # list of unique tracks
    tracks = list(set(df['trackId']))

    # Create an empty table that will contain, for each track, the list of frames and the list of bboxes
    dftrack = pd.DataFrame(columns = ['trackId', 'frames', 'bboxes'])

    # For each track ...
    for ii,track in enumerate(tracks):
        # Create a new dataframe with all the frames that contain this track
        dft = df.loc[df.loc[:,'trackId']==track]

        frames = dft['frameId'].tolist()
        # List of bboxes of this track
        bboxes = dft[['tlx','tly','width','height']].to_numpy().tolist()
        
        # Add a new row to the table
        dftrack.loc[ii] = [track, frames, bboxes]

    return (dftrack)
'''
def read_rectangles2(df):

    # Co vert from x1y1wh to x1y1x2w2
    df[4] = df[4] + df[2]
    df[5] = df[5] + df[3]

    for ii in range(len(df)):
        tlx, tly, brx, bry = df.iloc[ii][2], df.iloc[ii][3], df.iloc[ii][4], df.iloc[ii][5]
        frame = df.iloc[ii][0]
        tid = df.iloc[ii][1]
        if frame not in rectangles:
            rectangles[frame] = list()

        rectangles[frame].append((tid,tlx,tly,brx,bry))
    
    return rectangles
'''

def read_rectangles(file_path):
    rectangles = {}
    with open(file_path, 'r') as f:
        for line in f:
            # Split line into columns
            columns = line.split(',')
            
            # Extract frame number
            try:
                frame = int(columns[0])
            except:
                frame = int(float(columns[0]))
            
            try:
                tid   = int(columns[1])
            except:
                tid   = int(float(columns[1]))
            
            # Extract rectangle coordinates
            x1, y1, w, h = map(int, map(round, map(float, columns[2:6])))
            x2 = x1 + w
            y2 = y1 + h
            
            if frame not in rectangles:
                rectangles[frame] = []
                
            rectangles[frame].append((tid, x1, y1, x2, y2))
            
    return rectangles

def bbox_center (bbox):
    return (bbox[0]+bbox[2]//2, bbox[1]+bbox[3]//2)


def plot_rectangles(img, rectangles, fr, dftrack=None):
    # Plot rectangles
    
    for rectangle in rectangles:
        x1, y1, x2, y2 = rectangle[1:]
        tid = rectangle[0]
        cv.rectangle(img, (x1, y1), (x2, y2), colours[tid%32], 3)

    # Write image to disk
    return img


#colours = np.random.rand(32, 3) #used only for display
#print (colours)

colours = [
    [0, 0, 255],
    [255, 0, 0],
    [0, 255, 0],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
    [192, 192, 192],
    [128, 128, 128],
    [128, 0, 0],
    [128, 128, 0],
    [0, 128, 0],
    [128, 0, 128],
    [0, 128, 128],
    [0, 0, 128],
    [255, 165, 0],
    [255, 215, 0],
    [184, 134, 11],
    [218, 165, 32],
    [189, 183, 107],
    [0, 100, 0],
    [70, 130, 180],
    [95, 158, 160],
    [30, 144, 255],
    [255, 250, 205],
    [173, 216, 230],
    [255, 192, 203],
    [240, 230, 140],
    [210, 180, 140],
    [255, 99, 71],
    [250, 128, 114],
    [233, 150, 122],
    [244, 164, 96]
]


if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)
    tracking_file    = args['<trackFile>']
    input_video      = args['<inputVideo>']
    out_video        = args['<outputVideo>']
    downsample_video = bool(strtobool(args['--downsampleVideo']))
    start_frame      = int(args['--startFrame'])
    max_frame        = int(args['--maxFrame'])
    frame_step       = int(args['--frameStep'])
    
    # A directory with frames extracted from a video can be used instead of a video. In this case, the frames
    # should be named as 000001.png, 000002.png, etc. and the name will be a template like ./imgs/%06d.jpg
    capture = cv.VideoCapture(input_video)
    if not capture.isOpened():
        print('Unable to open: ' + args.input, file=sys.stderr)
        exit(0)

    # Define the codec and create a video writer object
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    ww,hh = int(capture.get(3)), int(capture.get(4))
    if downsample_video:
        ww, hh = int(np.round(ww/2.0)), int(np.round(hh/2.0))

    out = cv.VideoWriter(out_video, fourcc, 25.0, (ww, hh))


    df = pd.read_csv(tracking_file, sep=',', header=None)
    df.columns= ['frameId','trackId','tlx','tly','width','height','conf','a','b','c']

    # info about tracks init-end
    dftrack = trackinfo(df)

    # Read rectangles from text file
    #rectangles = read_rectangles2(df)
    
    # Read rectangles from text file
    rectangles = read_rectangles(tracking_file)

    fr = start_frame
    while True:
        if fr == max_frame:
            break

        if fr % 500 == 0:
            print (f'Processing frame {fr}', file=sys.stderr)
            
        ret, frame = capture.read()
        if frame is None:
            break

        if fr in rectangles:
            # Plot rectangles on image
            frame = plot_rectangles(frame, rectangles[fr], fr, dftrack)

        if downsample_video:
            out.write(cv.pyrDown(frame))
        else:
            out.write(frame)
            
        fr = fr + 1 + frame_step
        
    # Release the video capture and writer objects
    capture.release()
    out.release()

    convert_to_gif(out_video, out_video.replace('.mp4', '.gif'), fps=2, scale=0.25)