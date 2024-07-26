import cv2
import sys
import time
import numpy as np
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PlotModule import LivePlot
# dimensions for displayed frame
realWidth = 640   
realHeight = 480  
# dimensions for processed frame
videoWidth = 160
videoHeight = 120

videoChannels = 3            #no of color channels(3 for rgb)
videoFrameRate = 15

# Webcam Parameters
webcam = cv2.VideoCapture(0)       #initializes the webcam for capturing video in OpenCV.  When you call cv2.VideoCapture(0), it attempts to open the default webcam on your device. If successful, it returns a VideoCapture object, which can then be used to read frames from the webcam.
detector = FaceDetector()           # initializes a face detector object from the FaceDetectionModule.it initializes the necessary components to detect faces using pre-trained models. 

webcam.set(3, realWidth)        # 3: This is the property identifier for the frame width.realWidth: The desired width of the video frames.
webcam.set(4, realHeight)

# Color Magnification Parameters
levels = 3
alpha = 170             #This parameter sets the amplification factor for the color magnification.
minFrequency = 1.0      #This sets the minimum frequency (in Hz) of the bandpass filter.Purpose: The bandpass filter is used to isolate the frequency range that corresponds to the heart rate. The minimum frequency here represents the lowest expected heart rate. For example, 1.0 Hz corresponds to 60 beats per minute (bpm).
maxFrequency = 2.0
bufferSize = 150        # This parameter defines the number of frames to store in the buffer.
bufferIndex = 0

plotY = LivePlot(realWidth,realHeight,[60,120],invert=True)

# Helper Methods  Applies a Gaussian blur and downscaling operation to the current image frame, effectively reducing its size and smoothing it.
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid
#reverse of gaussian pyramid. reconstructs image from guassian pyramid
def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame

# Output Display Parameters to set up various configurations for displaying text and graphics on the video frames
font = cv2.FONT_HERSHEY_SIMPLEX     #This specifies the font type used for text rendering in OpenCV. 
loadingTextLocation = (30, 40)      # where the text to be displayed
bpmTextLocation = (videoWidth//2, 40)   #Specifies the position where the heart rate in BPM (beats per minute) will be displayed.
fpsTextLoaction = (500,600)              #Specifies the position where the frames per second (FPS) will be displayed.


fontScale = 1       #size of font 1 - normal
fontColor = (255,255,255)
lineType = 2        #Determines the thickness of the text outline.Value: 2 specifies a thickness of 2 pixels
boxColor = (0, 255, 0)
boxWeight = 3

# Initialize Gaussian Pyramid
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))        #Creates an empty (all-zero) image with dimensions specified by videoHeight, videoWidth, and videoChannels.
firstGauss = buildGauss(firstFrame, levels+1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))   #firstGauss.shape[0]: The height of each frame in the pyramid. [1] means width
fourierTransformAvg = np.zeros((bufferSize))

# Bandpass Filter for Specified Frequencies
frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)


# Heart Rate Calculation Variables
bpmCalculationFrequency = 10   #15 determines how often the BPM value is updated. For example, if set to 10, BPM is calculated every 10 frames. 
bpmBufferIndex = 0
bpmBufferSize = 10
bpmBuffer = np.zeros((bpmBufferSize))

i = 0
ptime = 0
ftime = 0
while (True):
    ret, frame = webcam.read()      # Reads a frame from the webcam. ret is a boolean indicating if the frame was successfully captured. frame conta
    if ret == False:                   #Checks if the frame was not successfully captured.
        break       

    frame, bboxs = detector.findFaces(frame,draw=False)     #Uses the FaceDetector to detect faces in the frame. bboxs is a list of bounding boxes around detected faces. draw=False means no annotations are drawn on the frame.
    frameDraw = frame.copy()        # Creates a copy of the frame to draw additional information (e.g., FPS, BPM) without altering the original frame.
    ftime = time.time()
    fps = 1 / (ftime - ptime)
    ptime = ftime

    cv2.putText(frameDraw, f'FPS: {int(fps)}', (30, 440), 0, 1, [0, 0, 0], thickness=2, lineType=cv2.LINE_AA)
    if bboxs:
        x1, y1, w1, h1 = bboxs[0]['bbox']
        cv2.rectangle(frameDraw, bboxs[0]['bbox'], (255, 0, 255), 2)
        detectionFrame = frame[y1:y1 + h1, x1:x1 + w1]
        detectionFrame = cv2.resize(detectionFrame,(videoWidth,videoHeight))

        # Construct Gaussian Pyramid
        videoGauss[bufferIndex] = buildGauss(detectionFrame, levels+1)[levels]
        fourierTransform = np.fft.fft(videoGauss, axis=0)

        # Bandpass Filter
        fourierTransform[mask == False] = 0

        # Grab a Pulse
        if bufferIndex % bpmCalculationFrequency == 0:
            i = i + 1
            for buf in range(bufferSize):
                fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            hz = frequencies[np.argmax(fourierTransformAvg)]
            bpm = 60.0 * hz
            bpmBuffer[bpmBufferIndex] = bpm
            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

        # Amplify
        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        filtered = filtered * alpha

        # Reconstruct Resulting Frame
        filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
        outputFrame = detectionFrame + filteredFrame
        outputFrame = cv2.convertScaleAbs(outputFrame)

        bufferIndex = (bufferIndex + 1) % bufferSize
        outputFrame_show = cv2.resize(outputFrame,(videoWidth//2,videoHeight//2))
        frameDraw[0:videoHeight // 2, (realWidth-videoWidth//2):realWidth] = outputFrame_show

        bpm_value = bpmBuffer.mean()
        imgPlot = plotY.update(float(bpm_value))

        if i > bpmBufferSize:
            cvzone.putTextRect(frameDraw,f'BPM: {bpm_value}',bpmTextLocation,scale=2)
        else:
            cvzone.putTextRect(frameDraw, "Calculating BPM...", loadingTextLocation,scale=2)

        if len(sys.argv) != 2:
            imgStack = cvzone.stackImages([frameDraw,imgPlot],2,1)
            cv2.imshow("Heart Rate Monitor", imgStack)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        imgStack = cvzone.stackImages([frameDraw, frameDraw], 2, 1)
        cv2.imshow("Heart Rate Monitor", imgStack)
webcam.release()
cv2.destroyAllWindows()

# webcam = cv2.VideoCapture(0)

# # Use this
# video_file = 'path_to_your_video_file.mp4'  # Replace with your video file path
# webcam = cv2.VideoCapture(video_file)
# if not webcam.isOpened():
#     print("Error: Could not open video file.")
#     exit()
# The process of calculating heart rate from a detected face in the video involves several steps. The key idea is to use subtle color variations in the face, caused by blood flow, to estimate the heart rate.