#####
# Author:  David Mihola
# Contact: mihola.dejv@gmail.com
# Date:    2025
# About:   See Section 5.1 Image Signal Processing Focused Deep Learning Pipeline in the thesis,
#          the script implements the Table 5.1: Image signal processing pipeline exposure times and analog gain configuration given lighting conditions. 
#####

import numpy as np

class ExposureModes:
    # values for the ISP pipeline configuration
    Short = 1
    Normal = 0
    Long = 2
    
    def toString(value):
        if ExposureModes.Short == value:
            return "Short"
        if ExposureModes.Normal == value:
            return "Normal"
        if ExposureModes.Long == value:
            return "Long"
        
class ExposureModeHandler:    
    # values as per the Table 5.1
    ShortToNormalExposureLuxThreshold = 500
    NormalToShortExposureLuxThreshold = 525
    NormalToLongExposureLuxThreshold = 60
    LongToNormalExposureLuxThreshold = 75
    LongToNightExposureLuxThreshold = 10
    NightToLongExposureLuxThreshold = 15
    
    def __init__(self, piCamera, bufferSize=512):
        self.luxIdx = 0
        self.luxIdxMask = bufferSize - 1
        self.luxDivider = self.luxIdxMask + 1
        self.luxRunningAverage = 525
        self.lastLuxMeasurements = np.zeros(bufferSize, np.float32) + 525 / bufferSize
        self.currentExposureMode = ExposureModes.Short
        self.lowLightPipelineActive = False
        self.piCamera = piCamera
        piCamera.set_controls({"AeExposureMode": self.currentExposureMode})

    def updateLux(self, lux, lowLightSwitchCallback=lambda x: None, highLightSwitchCallback=lambda x: None):
        # update the running average of the lux values
        luxDivided = lux / self.luxDivider
        self.luxRunningAverage -= self.lastLuxMeasurements[self.luxIdx]
        self.luxRunningAverage += luxDivided
        self.lastLuxMeasurements[self.luxIdx] = luxDivided
        self.luxIdx = (self.luxIdx + 1) & self.luxIdxMask
        
        # update the exposure mode based on the lux value if it changes enough
        newExposureMode = self.currentExposureMode
        if ExposureModes.Short == self.currentExposureMode and self.luxRunningAverage < self.ShortToNormalExposureLuxThreshold:
            newExposureMode = ExposureModes.Normal
        elif ExposureModes.Normal == self.currentExposureMode and self.luxRunningAverage < self.NormalToLongExposureLuxThreshold:
            newExposureMode = ExposureModes.Long
        elif ExposureModes.Normal == self.currentExposureMode and self.luxRunningAverage > self.NormalToShortExposureLuxThreshold:
            newExposureMode = ExposureModes.Short
        elif ExposureModes.Long == self.currentExposureMode and self.luxRunningAverage > self.LongToNormalExposureLuxThreshold:
            newExposureMode = ExposureModes.Normal
        elif ExposureModes.Long == self.currentExposureMode and self.luxRunningAverage < self.LongToNightExposureLuxThreshold:
            if not self.lowLightPipelineActive:
                lowLightSwitchCallback(True)
                print("Switching to low light pipeline")
            self.lowLightPipelineActive = True
        elif ExposureModes.Long == self.currentExposureMode and self.luxRunningAverage > self.NightToLongExposureLuxThreshold:
            if self.lowLightPipelineActive:
                highLightSwitchCallback(False)
                print("Switching to normal pipeline")
            self.lowLightPipelineActive = False
        
        # change the exposure mode of the camera if the lux running average crosses any of the thresholds
        if newExposureMode != self.currentExposureMode:
            print(f"Changing exposure mode from '{ExposureModes.toString(self.currentExposureMode)}' to '{ExposureModes.toString(newExposureMode)}'")
            self.currentExposureMode = newExposureMode
            self.piCamera.set_controls({"AeExposureMode": newExposureMode})

        return self.currentExposureMode
        
