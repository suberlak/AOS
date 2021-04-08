import numpy as np
import updatePhosimFunctions as up

import lsst.geom as geom
import lsst.afw.cameraGeom as cameraGeom
from lsst.obs.lsst import LsstCam

camera = LsstCam().getCamera()


###########################################
#                                         # 
#  Update focalplanelayout for lsstCam    #
#                                         #
###########################################



# Need to store each line of the other focalplanelayout to convey the zernike perturbations ...
headerLines, contentLinesZer = up.readFile('/project/scichris/aos/phosim_syseng4/data/lsst/focalplanelayout_old.txt')


splitLinesZerDic = {}
for line in contentLinesZer:
    splitLinesZerDic[line.split()[0]] = line.split() # sensor name is the key 
    
  
    
# read the file and split into two lists
headerLines, contentLines = up.readFile('/project/scichris/aos/phosim_syseng4/data/lsst/focalplanelayout_old.txt')

# update the content 
newContentLines = []
for line in contentLines:
    splitContent = line.split()
    newSplitContent = splitContent.copy()
    
    # update the sensor name 
    sensorName  = splitContent[0]
    newName = up.getNewSensorName(sensorName)
    newSplitContent[0] = newName
    
    # update the defocal shift 
    deformationCoeff = splitContent[19]
    defCoeff = float(deformationCoeff)
    if abs(defCoeff)>10:
        sgn = np.sign(defCoeff) # change +/- 1000 to +/- 1500 
        newDeformationCoeff = str(defCoeff + sgn*500.0)
        #print(newDeformationCoeff)
        newSplitContent[19]  = newDeformationCoeff
    
    # update the x,y position 
    detector = camera.get(newName)
    bbox = detector.getBBox()
    xc_px, yc_px = bbox.centerX, bbox.centerY
    xc_mm, yc_mm = detector.transform(geom.PointD(xc_px, yc_px), 
                                                cameraGeom.PIXELS, 
                                                cameraGeom.FOCAL_PLANE)
    
    xc_microns , yc_microns = 1000*xc_mm , 1000*yc_mm
    xpos, ypos = float(splitContent[1]), float(splitContent[2])
    #print(xpos, ypos, '-->', 
    #      yp_microns, xp_microns, 
    #      'dx:', xpos - yp_microns, 
    #      'dy:', ypos-xp_microns)
    newSplitContent[1] = str(round(yc_microns,2))
    newSplitContent[2] = str(round(xc_microns,2))
    
    
    # update the number of  x px, y px 
    xpx, ypx = splitContent[4], splitContent[5]
    
    xnew, ynew = bbox.getHeight(), bbox.getWidth() # rotated wrt lsstCam ... 
    newSplitContent[4] = str(xnew)
    newSplitContent[5] = str(ynew)
    
    # update the dx, dy - since we know the actual position, set these to 0 ...
    dx,dy,dz = float(splitContent[15]), float(splitContent[16]), float(splitContent[17])
    newSplitContent[15] = '0' # <-- set dx to 0 
    newSplitContent[16] = '0' # <-- set dy to 0 

    
    # change sensor.txt to zern, and append all parts that follow...
    # that way I can also use the 'new' format of focalplanelayout,
    # changing all after 'sensor.txt' to zern and the coefficients ... 
    newSplitContent[18]  = 'zern'
    newSplitContent[20:] = splitLinesZerDic[sensorName][20:]
    
    # append updated line 
    newContentLines.append(' '.join(newSplitContent)+'\n')
    
# store the updated version: the header and  updated  content lines 
fname = "/project/scichris/aos/phosim_syseng4/data/lsst/focalplanelayout_upd.txt"
f = open(fname, "w")
f.writelines(headerLines)
f.writelines(newContentLines)
f.close()      
print('Saved as %s'%fname)