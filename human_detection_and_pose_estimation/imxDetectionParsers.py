import numpy as np

class Detection:
    def __init__(self, boxCoords, roiCoords, scale, cropImage=None, feature=None):
        self.boxCoords = boxCoords
        self.roiCoords = roiCoords
        self.scale = scale
        self.cropImage = cropImage
        self.feature = feature

class VehicleDetection(Detection):
    def __init__(self, boxCoords, roiCoords, scale, cropImage=None, feature=None, plateLabels=[], valid=False):
        super().__init__(boxCoords, roiCoords, scale, cropImage, feature)
        self.plateLabels = plateLabels
        self.valid = valid

class ImxDetectionParser:
    def __init__(self, imx500, camera, frameWidth, frameHeight, confidence=0.5, sensorRoi=False, minHeighWidthRatio=0.6, minRoiEdgeSize=200):
        self.imx500 = imx500
        self.camera = camera
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight
        self.confidence = confidence
        self.minHeighWidthRatio = minHeighWidthRatio
        self.minRoiEdgeSize = minRoiEdgeSize

        if sensorRoi:
            self.widthScale = int(frameWidth * 0.75)
            self.offsetX = int(frameWidth * 0.125)
        else:
            self.widthScale = frameWidth
            self.offsetX = 0
    
    def getCropCoordinatesMobileNet(self, metadata) -> list[tuple[int, int, int, int]]:
        """
        Retrieves the coordinates of the bounding boxes of detected objects and prepares regions of interest for further processing.

        :param metadata: metadata of the frame produced by a MobileNet model trained on the COCO dataset
        """

        npOutputs = self.imx500.get_outputs(metadata, add_batch=True)
        if npOutputs:
            boxes, scores = npOutputs[0][0], npOutputs[1][0]
            selectedBoxes = np.array([boxCoords for boxCoords, score 
                                          in zip(boxes, scores) 
                                          if score > self.confidence])
            
            return self._getCropCoordinates(selectedBoxes, metadata)
        else:
            return []

    def getCropCoordinatesYolo(self, metadata, classes) -> list[tuple[int, int, int, int]]:
        """
        Retrieves the coordinates of the bounding boxes of detected vehicles and prepares regions of interest for further processing.

        :param metadata: metadata of the frame produced by a YOLOv8 model 
        """

        npOutputs = self.imx500.get_outputs(metadata, add_batch=True)
        size, _ = self.imx500.get_input_size()
        if npOutputs:
            boxes, scores, classes = npOutputs[0][0], npOutputs[1][0], npOutputs[2][0]
            boxes = boxes / size
            boxes = boxes[:, [1, 0, 3, 2]]
            boxes = np.array_split(boxes, 4, axis=1)
            boxes = zip(*boxes)

            selectedBoxes = np.array([boxCoords for boxCoords, score, cls 
                                        in zip(boxes, scores, classes) 
                                        if cls in classes and score > self.confidence])
            
            return self._getCropCoordinates(selectedBoxes, metadata)
        else:
            return []

    def _getCropCoordinates(self, selectedBoxes, metadata) -> list[tuple[int, int, int, int]]:
        boxesCrops = []
        for boxCoords in selectedBoxes:
            boxCoords = self.imx500.convert_inference_coords(boxCoords, metadata, self.camera)
            boxesCrops.append(boxCoords)
        
        return boxesCrops
