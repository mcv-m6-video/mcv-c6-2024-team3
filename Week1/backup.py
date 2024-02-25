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
                cv2.imwrite(pathOutput + '/frame' + str(totalFrames).zfill(5) + '.jpg', frame)
                totalFrames += 1

            return totalFrames, os.listdir(pathOutput)
        
        else:
            return len([name for name in os.listdir(pathOutput) if os.path.isfile(os.path.join(pathOutput, name))]), os.listdir(pathOutput)

class BackgorundRemoval:

    def __init__(self, pathInput, pathOutput):
        self.framesInPath = pathInput
        self.framesOutPath = pathOutput
        self.numFrames, self.listFrames = getFrames(self.framesInPath, self.framesOutPath)

    def getNumFrames(self):
        return self.numFrames
    
    def getFrames(self):
        return self.listFrames
    
    def train(self, percentageOfFrames=0.25):
        lastFrame = math.floor(self.numFrames * percentageOfFrames)

        print('Using ' + str(lastFrame) + ' frames to train the model.')
        
        frames = []
        for i in tqdm(range(0, lastFrame)):
            frame = cv2.imread(self.framesOutPath + '/' + self.listFrames[i], cv2.IMREAD_GRAYSCALE)
            frames.append(frame)

        frames = np.array(frames)

        self.meanImage = np.mean(frames, axis=0) / 255
        self.stadImage = np.std(frames, axis=0) / 255

        # cv2.imshow('Mean Image', self.meanImage)
        # cv2.waitKey(0)

    def test(self, alpha = 2.5):

        videoIn = cv2.VideoCapture(self.framesInPath)
        fps = videoIn.get(cv2.CAP_PROP_FPS)
        frame_width = int(videoIn.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(videoIn.get(cv2.CAP_PROP_FRAME_HEIGHT))
        videoIn.release()

        video_name = 'Task1_1.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        video = cv2.VideoWriter(video_name, fourcc, fps, (frame_width, frame_height))

        for i in tqdm(range(0, self.numFrames)):
            frame = cv2.imread(self.framesOutPath + '/' + self.listFrames[i], cv2.IMREAD_GRAYSCALE)
            frame = frame * (1. / 255)

            frame = np.abs(frame - self.meanImage)

            result = frame >=  alpha * (alpha * self.stadImage + 2/255)
            result = result.astype(np.uint8) * 255
            
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

            video.write(result)
        
        video.release()