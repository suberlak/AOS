import lsst.geom as geom
import lsst.afw.cameraGeom as cameraGeom
from lsst.obs.lsst import LsstComCam
import updatePhosimFunctions as up
camera = LsstComCam().getCamera()

###########################################
#                                         # 
#  Update focalplanelayout for comCam     #
#                                         #
###########################################

    
# read the file and split into two lists
headerLines, contentLines = up.readFile('/project/scichris/aos/phosim_syseng4/data/comcam/focalplanelayout_old.txt')

# store sensors in the mapper in  a variable
mapperSensors = list(camera.getNameMap().keys())

# DEROTATE PHOSIM? 
derotate_phosim = True

if derotate_phosim:
    print('Derotating phosim')
    ticket_number= 'DM-30367' # both: obs_lsst orientation
else:
    print('Keeping phosim orientation')
    ticket_number='DM-29264' # comcam 
    
# update the content 
newContentLines = []
for line in contentLines:
    splitContent = line.split()
    newSplitContent = splitContent.copy()
    
    # no need to update the sensor name 
    # comcam raft names are the same 
    sensorName  = splitContent[0]
    
    # save only the lsstComCam sensors - no need to simulate anything else here.... 
    if sensorName in mapperSensors:  

        # update the defocal shift 
        deformationCoeff = splitContent[19]
        defCoeff = float(deformationCoeff)
        if abs(defCoeff)>10:
            sgn = np.sign(defCoeff) # change +/- 1000 to +/- 1500 
            newDeformationCoeff = str(defCoeff + sgn*500.0)
            #print(newDeformationCoeff)
            newSplitContent[19]  = newDeformationCoeff

        # update the x,y position 
        detector = camera.get(sensorName)
        bbox = detector.getBBox()
        xc_px, yc_px = bbox.centerX, bbox.centerY
        xc_mm, yc_mm = detector.transform(geom.PointD(xc_px, yc_px), 
                                                    cameraGeom.PIXELS, 
                                                    cameraGeom.FOCAL_PLANE)

        xc_microns , yc_microns = 1000*xc_mm , 1000*yc_mm
        #xpos, ypos = float(splitContent[1]), float(splitContent[2])
        #print(xpos, ypos, '-->', 
        #      yp_microns, xp_microns, 
        #      'dx:', xpos - yp_microns, 
        #      'dy:', ypos-xp_microns)
        
        # [1] is x-pos, [2] is y-pos 
        if derotate_phosim:  # x_phosim <-- x_lsst, i.e. no transpose
            newSplitContent[1] = str(round(xc_microns,2))
            newSplitContent[2] = str(round(yc_microns,2))
            
        else: # x_phosim <-- y_obslsst , i.e. apply transpose
            newSplitContent[1] = str(round(yc_microns,2))
            newSplitContent[2] = str(round(xc_microns,2))

        # update the number of  x px, y px 
        # old values 
        #xpx, ypx = splitContent[4], splitContent[5]
        
        if derotate_phosim:
            xnew, ynew = bbox.getWidth(), bbox.getHeight() # same as lsstCam  
        else:
            xnew, ynew = bbox.getHeight(), bbox.getWidth() # rotated wrt lsstCam ... 
            
        newSplitContent[4] = str(xnew)
        newSplitContent[5] = str(ynew)

        # update the dx, dy - since we know the actual position, set these to 0 ...
        dx,dy,dz = float(splitContent[15]), float(splitContent[16]), float(splitContent[17])
        newSplitContent[15] = '0' # <-- set dx to 0 
        newSplitContent[16] = '0' # <-- set dy to 0 

        # append updated line 
        newContentLines.append(' '.join(newSplitContent)+'\n')
    
# store the updated version: the header and  updated  content lines 
fname = f"/project/scichris/aos/phosim_syseng4/data/comcam/focalplanelayout_{ticket_number}.txt"
f = open(fname, "w")
f.writelines(headerLines)
f.writelines(newContentLines)
f.close()      
print('Saved as %s'%fname)