from webbrowser import get
import cv2
import os
import shutil

def getFrames(pathInput, pathOutput):
        # Uncomment this if you want to introduce another different video from the test
        if not os.path.exists(pathOutput):
            cap = cv2.VideoCapture(pathInput)

            if os.path.exists(pathOutput):  
                shutil.rmtree(pathOutput)

            os.mkdir(pathOutput)

            totalFrames = 0
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                    break
                cv2.imwrite(pathOutput + '/frame%d.jpg' % cap.get(1), frame)
                totalFrames += 1

            return totalFrames
        
        else:
            return len([name for name in os.listdir(pathOutput) if os.path.isfile(os.path.join(pathOutput, name))])

class BackgorundRemoval:

    def __init__(self, pathInput, pathOutput):
        self.framesInPath = pathInput
        self.framesOutPath = pathOutput
        self.numFrames = getFrames(self.framesInPath, self.framesOutPath)

    def getFrames(self):
        return self.numFrames
    
    def train(self, numberOfFrames):
        pass

if __name__ == '__main__':
    # Read a video and save the frames on a folder
    inFrames = 'c010\\vdo.avi'
    outFrames = 'frames'
    
    modelo = BackgorundRemoval(inFrames, outFrames)
    print(modelo.getFrames())



    