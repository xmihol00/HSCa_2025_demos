from threading import Thread
import cv2
import socket
import time
import numpy as np

from picamera2 import Picamera2, MappedArray, libcamera
from picamera2.encoders import LibavH264Encoder
from picamera2.outputs import FileOutput
from picamera2.devices import IMX500
from picamera2.devices import Hailo
import av

from exposureModeHandling import ExposureModes, ExposureModeHandler
from imxDetectionParsers import ImxDetectionParser

SKELETON = [
    (0, 1),   (1, 3),   # nose to left eye, left eye to left ear
    (0, 2),   (2, 4),   # nose to right eye, right eye to right ear
    (5, 6),             # reft shoulder to right shoulder
    (5, 7),   (7, 9),   # reft shoulder to left elbow, left elbow to left wrist
    (6, 8),   (8, 10),  # right shoulder to right elbow, right elbow to right wrist
    (5, 11),  (6, 12),  # left shoulder to left hip, right shoulder to right hip
    (11, 12),           # left hip to right hip
    (11, 13), (13, 15), # left hip to left knee, left knee to left ankle
    (12, 14), (14, 16), # right hip to right knee, right knee to right ankle
]

frameWidth = 1920
frameHeight = 1080
targetFrameWidth = 768
targetFrameHeight = 1024
hailoFrameWidth = 192
hailoFrameHeight = 256
showCenter = True
fps = 25
imxModel = "yolov8n.rpk"
hailoModel = "vit_pose_small_bn.hef"
loresFrameWidth = 640
loresFrameHeight = 640
confidence = 0.4

gLastDetection = [0, targetFrameHeight, 0, targetFrameWidth, (targetFrameWidth // 2, targetFrameHeight // 2)]
gLastTimestamp = time.time()
gFrameCount = 0
gLastFPS = 0
gHailo = None
gImx500 = IMX500(imxModel)
gIntrinsics = gImx500.network_intrinsics
gLastFrame = np.zeros((loresFrameHeight, loresFrameWidth, 3), dtype=np.uint8)
gCurrentFrame = np.zeros((loresFrameHeight, loresFrameWidth, 3), dtype=np.uint8)
gLastXs = np.zeros(17)
gLastYs = np.zeros(17)
gLastScores = np.zeros(17)
gHeatmap = np.zeros((hailoFrameHeight // 4, hailoFrameWidth // 4, 17))
gResized = np.zeros((hailoFrameHeight, hailoFrameWidth, 3), dtype=np.uint8)
gPicam2 = Picamera2()
gImxDetectionParser = ImxDetectionParser(gImx500, gPicam2, frameWidth, frameHeight, confidence=0.5, minHeighWidthRatio=0.0, minRoiEdgeSize=0)

class H264Encoder(LibavH264Encoder):
    def __init__(self, bitrate, framerate):
        super().__init__(bitrate=bitrate, framerate=framerate)
    
    def _encode(self, stream, request):
        global gLastDetection, gLastTimestamp, gFrameCount, gLastFPS, gHailo, gHeatmap, gLastScores, gLastXs, gLastYs, gLastFrame, gCurrentFrame, gResized
        topLeft_y, bottomRight_y, topLeft_x, bottomRight_x, closestCenter = gLastDetection

        timestamp_us = self._timestamp(request)
        with MappedArray(request, stream) as m:
            if showCenter:
                cv2.rectangle(m.array, (int(closestCenter[0]) - 3, int(closestCenter[1]) - 3), (int(closestCenter[0]) + 3, int(closestCenter[1]) + 3), (0, 0, 255), 3)

            gCurrentFrame = m.array[topLeft_y:bottomRight_y, topLeft_x:bottomRight_x].copy()
            gResized = cv2.resize(gCurrentFrame, (hailoFrameWidth, hailoFrameHeight))

            for x, y, score in zip(gLastXs, gLastYs, gLastScores):
                if score > confidence:
                    x = int(x * gLastFrame.shape[1] / gHeatmap.shape[1])
                    y = int(y * gLastFrame.shape[0] / gHeatmap.shape[0])
                    cv2.circle(gLastFrame, (x, y), 8, (0, 255, 0), -1)
                
            for pair in SKELETON:
                if gLastScores[pair[0]] > confidence and gLastScores[pair[1]] > confidence:
                    x1 = int(gLastXs[pair[0]] * gLastFrame.shape[1] / gHeatmap.shape[1])
                    y1 = int(gLastYs[pair[0]] * gLastFrame.shape[0] / gHeatmap.shape[0])
                    x2 = int(gLastXs[pair[1]] * gLastFrame.shape[1] / gHeatmap.shape[1])
                    y2 = int(gLastYs[pair[1]] * gLastFrame.shape[0] / gHeatmap.shape[0])
                    cv2.line(gLastFrame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
            currentTimestamp = time.time()
            gFrameCount += 1
            if currentTimestamp - gLastTimestamp > 1:
                gLastFPS = gFrameCount / (currentTimestamp - gLastTimestamp)
                gFrameCount = 0
                print(f"FPS: {gLastFPS}")
                gLastTimestamp = currentTimestamp

            cv2.putText(gLastFrame, f"FPS: {gLastFPS:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            frame = av.VideoFrame.from_ndarray(gLastFrame, format=self._av_input_format, width=targetFrameWidth)
            frame.pts = timestamp_us
            for packet in self._stream.encode(frame):
                self._lasttimestamp = (time.monotonic_ns(), packet.pts)
                self.outputframe(bytes(packet), packet.is_keyframe, timestamp=packet.pts, packet=packet)
            
            gLastFrame = gCurrentFrame.copy()

class TrackCropping:
    def __init__(self, targetWidth=targetFrameWidth, targetHeight=targetFrameHeight, currentWidth=frameWidth, 
                       currentHeight=frameHeight, averageCount=10, minMovement=1000):
        self.targetWidth = targetWidth
        self.halfTargetWidth = targetWidth // 2
        self.targetHeight = targetHeight
        self.halfTargetHeight = targetHeight // 2
        self.averageCount = averageCount
        self.color = (0, 0, 255)
        self.currentWidth = currentWidth
        self.currentHeight = currentHeight
        self.minMovement = minMovement
        
        self.centers = np.array([(currentWidth // 2, currentHeight // 2)] * averageCount)
        self.nextCenterIdx = 1
        self.lastCenterIdx = 0
        self.moving = False
        self.stationaryCountThreshold = 15
        self.currentStationaryCount = 0
    
    def getCropCoordinates(self, metadata):
        global gImx500
        boxes = gImxDetectionParser.getCropCoordinatesYolo(metadata, [0])
        centers = np.zeros((len(boxes), 2))
        for i, box in enumerate(boxes):
            centers[i] = [box[0] + box[2] * 0.5, box[1] + box[3] * 0.5]

        if centers is not None and len(centers) > 0:
            lastCenter = self.centers[self.lastCenterIdx % self.averageCount]
            closestCenter = centers[np.argmin(np.sum((centers - lastCenter) ** 2, axis=1))]
            move = (closestCenter[0] - lastCenter[0]) ** 2 > self.minMovement or (closestCenter[1] - lastCenter[1]) ** 2 > self.minMovement
            if self.moving or move:
                self.centers[self.nextCenterIdx % self.averageCount] = closestCenter
                self.lastCenterIdx += 1
                self.nextCenterIdx += 1
                self.currentStationaryCount += 1
                if self.currentStationaryCount > self.stationaryCountThreshold:
                    self.moving = False
                if move:
                    self.currentStationaryCount = 0
                    self.moving = True
            else:
                closestCenter = lastCenter

        elif centers is not None:
            print("No detection available")
            closestCenter = self.centers[self.lastCenterIdx % self.averageCount]

        average_center = np.average(self.centers, axis=0)
        topLeft_x = int(max(average_center[0] - self.halfTargetWidth, 0))
        topLeft_y = int(max(average_center[1] - self.halfTargetHeight, 0))
        bottomRight_x = int(min(average_center[0] + self.halfTargetWidth, self.currentWidth))
        bottomRight_y = int(min(average_center[1] + self.halfTargetHeight, self.currentHeight))

        if topLeft_x == 0:
            bottomRight_x = self.targetWidth
        elif bottomRight_x == self.currentWidth:
            topLeft_x = self.currentWidth - self.targetWidth

        if topLeft_y == 0:
            bottomRight_y = self.targetHeight
        elif bottomRight_y == self.currentHeight:
            topLeft_y = self.currentHeight - self.targetHeight       

        return topLeft_y, bottomRight_y, topLeft_x, bottomRight_x, closestCenter

trackCropper = TrackCropping()
encoder = H264Encoder(bitrate=1000000, framerate=fps)
ExposureModeHandler = ExposureModeHandler(gPicam2)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 10001))
    sock.listen()

    gPicam2.encoders = encoder

    conn, addr = sock.accept()
    stream = conn.makefile("wb")
    encoder.output = FileOutput(stream)
    video_config = gPicam2.create_video_configuration(
        main={"size": (targetFrameWidth, targetFrameHeight), "format": "RGB888"}
    ) # first, use a dummy configuration to configure the encoder
    gPicam2.configure(video_config)
    gPicam2.start_encoder(encoder)
    video_config = gPicam2.create_video_configuration(
        main={"size": (frameWidth, frameHeight), "format": "RGB888"}, 
        transform=libcamera.Transform(hflip=1, vflip=1),
        controls={'FrameRate': fps}
    ) # second, use a proper configuration to start the camera
    gPicam2.configure(video_config)
    gPicam2.start()
    
    with Hailo(hailoModel) as hailo:
        gHailo = hailo
        try:
            while True:
                metadata = gPicam2.capture_metadata()
                gLastDetection = trackCropper.getCropCoordinates(metadata)
                lux = metadata["Lux"]

                gHeatmap = gHailo.run(gResized)
                max_indices = np.argmax(gHeatmap.reshape(-1, gHeatmap.shape[-1]), axis=0)
                gLastYs, gLastXs = np.unravel_index(max_indices, gHeatmap.shape[:2])
                gLastScores = gHeatmap[gLastYs, gLastXs, range(gHeatmap.shape[-1])]

        except KeyboardInterrupt:
            pass

    gPicam2.stop()
    gPicam2.stop_encoder()
    conn.close()
