from pyexpat import model
from webbrowser import get
import cv2
import os
import shutil
import math
import numpy as np
from tqdm import tqdm
import pickle

COLOR_CHANGES = {
    'grayscale' : cv2.COLOR_BGR2GRAY,
    'hsv' : cv2.COLOR_BGR2HSV
}
class BackgroundRemoval:

    def __init__(self, pathInput, newHeight = None, newWidth = None, colorSpace = 'grayscale'):
        self.framesInPath = pathInput
        self.frames = []
        self.fps = None
        self.width = newWidth
        self.height = newHeight
        self.channels = None
        self.colorSpace = colorSpace
        self.totalFrames = 0

    
    def apply_changes(self, frames):

        newFrames = cv2.cvtColor(frames, COLOR_CHANGES[self.colorSpace])

        if self.width is not None and self.height is not None:
            newFrames = cv2.resize(newFrames, (self.width, self.height))

        return newFrames


        
    def getFrames(self, preload=False):

        if preload and not os.path.exists('frames.pkl'):
            print('The pkl file is not there, preload is not going to be used.')
            preload = False

        if preload:
            print('Loading the pkl file.')
            with open('frames.pkl', 'rb') as f:
                self.frames = pickle.load(f)

            print(self.frames.shape)
            self.totalFrames, self.height, self.width, self.channels = self.frames.shape

            print(self.totalFrames, self.width, self.height)
            print('Loading done.')
        
        else:

            cap = cv2.VideoCapture(self.framesInPath)

            self.totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f'Getting the frames. There are {self.totalFrames}.')

            self.fps = cap.get(cv2.CAP_PROP_FPS)

            for _ in tqdm(range(self.totalFrames)):
                ret, frame = cap.read()
                if ret == False:
                    break
                
                self.frames.append(frame)

            cap.release()

            self.frames = np.array(self.frames)

            with open('frames.pkl', 'wb') as f:
                pickle.dump(self.frames, f)

            print('Frames obtained.')


    def train(self, percentageFrames = 0.25):
        lastFrame = int(math.floor(self.totalFrames * percentageFrames))

        newFrames = self.apply_changes(self.frames)

        self.mean = np.mean(newFrames[:lastFrame, :, :], axis=0) / 255

        self.std = np.std(newFrames[:lastFrame, :, :], axis=0) / 255

        cv2.imshow('Frame', self.mean)
        cv2.waitKey(0)

    def test(self, percentageTrainFrames = 0.25):
        pass

            
            

if __name__ == '__main__':
    # Read a video and save the frames on a folder
    inFrames = 'c010\\vdo.avi'
    
    modelo = BackgroundRemoval(inFrames)

    modelo.getFrames(preload=True)
    modelo.train()



    