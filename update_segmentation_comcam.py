import updatePhosimFunctions as up
from lsst.obs.lsst import LsstComCam
import numpy as np 


# read all the lines to a list : 
pathToFile = '/project/scichris/aos/phosim_syseng4/data/comcam/segmentation_old.txt'

# read the content of segmentation.txt file
# and store in a  dictionary
headerLines, contentLines = up.readFile(pathToFile)
sensorData = up.storeSensorDataAsDictionary(contentLines)

# initialize the camera 
camera = LsstComCam().getCamera()
    
# Update the data 
newSensorData  = {}
  
# running only over CCDs in the mapper,
# even though originally that comcam segmentation.txt 
# had data for guide and corner sensors ... (basically it was 
# the same as for data/lsst/segmentation.txt... )

for sensorName in camera.getNameIter():
    print('Running %s'%sensorName)
    newName = up.getNewSensorName(sensorName)
    
    # get the lsstCam data for that sensor 
    #print(newName)
    lsstDetectors = camera.get(newName)
    
    # initialize a list for the new data 
    newSensorData[newName] = []
    
    for content in sensorData[sensorName]:
        splitContent = content.split() # by default splits by whitespace, omitting empty strings 
        newSplitContent = splitContent.copy()
        
        # first line contains sensor name, number of amps, and ccd dimensions, eg.
        # 'R00_S12', '16', '4000', '4072'
        if len(content)<40:
            # update sensor name 
            newSplitContent[0] = newName
            
            # update sensor px_x, px_y 
            bbox = lsstDetectors.getBBox()
            px_y = bbox.getDimensions()[0]
            px_x = bbox.getDimensions()[1]
            newSplitContent[2] = str(px_x)
            newSplitContent[3] = str(px_y)
            
            # print information...
            old_px_x = splitContent[2]
            old_px_y = splitContent[3]
            #if old_px_x !=  str(px_x) : 
                #print(sensorName, old_px_x, old_px_y,  '--> ', newName, px_x, px_y)
            
        if len(content)>40:
            #continue
            #print(len(content), content)
            
            sensorAmpName = splitContent[0]
            ampName = sensorAmpName.split('_')[2]
            #ampGain = splitContent[7]   <--  in the mapper as amp.getGain()
            #ampGainVariance = splitContent[8] <-- change from 3% to 0
            #ampBias  = splitContent[9]   
            #ampBiasVariance = splitContent[10]
            #ampReadNoise = splitContent[11]  <-- in the mapper as amp.getReadNoise()
            #ampReadNoiseVariance = splitContent[12]
            #ampDarkCurrent = splitContent[13]

            #print('%s: gain %s, bias %s, noise %s, dc %s'% (sensorAmpName, ampGain, ampBias, ampReadNoise, ampDarkCurrent ))
            #print(content)
            #newContent = splitContent.copy()
            
                
            # for the main raft the amp names are correct 
            newSensorAmpName = '%s_%s'%(newName, ampName)
            newSplitContent[0] = newSensorAmpName
            
            # read the mapper information for that sensor / amp ... 
            for amp in lsstDetectors:
                if amp.getName() == ampName:
                    
                    # update gain and readNoise 
                    newAmpGain = str(amp.getGain())
                    if amp.getGain() == 0:
                        print(newName, ampName, amp.getGain())
                    newAmpReadNoise = str(amp.getReadNoise()) 
                    #print('%s: gain %s --> %s, noise %s --> %s'%(ampName, ampGain, newAmpGain, 
                    #                                             ampReadNoise, newAmpReadNoise))
                    #if float(newAmpGain) > 2 :
                        #print(newName)
                        #print('   %s: gain %s --> %s, suspicious value '%(ampName, ampGain, newAmpGain))
                        
                    # replace the content values ... 
                    newSplitContent[7] = newAmpGain
                    newSplitContent[11] = newAmpReadNoise
                    
                    # change the gain variance from 3% to 0 
                    newSplitContent[8] = str(0)  
                    
                    # update xlo, xhi, ylo, yhi ...
                    # amp.getBBox() is the only one that explains the extent of the amplifier 
                    # in sensor  coords ... 
                    bbox = amp.getBBox()

                    # here I assign them to match the phoSim Transposed system 
                    xlo = bbox.getMinY()
                    xhi = bbox.getMaxY()
                    ylo = bbox.getMinX()
                    yhi = bbox.getMaxX()
                    
                    #print(amp.getName(), xlo,xhi,ylo,yhi)
                    xlo_old = splitContent[1]
                    xhi_old = splitContent[2]
                    ylo_old = splitContent[3]
                    yhi_old = splitContent[4]
                    
                    
                    #print('Updating', amp.getName(), xlo_old, xhi_old, ylo_old, yhi_old, 
                    #          '-->', xlo,xhi,ylo,yhi)
                    #if xlo_old != str(xlo): # wrong filter   - just update all ! 
                        #print('Updating', amp.getName(), xlo_old, xhi_old, ylo_old, yhi_old, 
                        #      '-->', xlo,xhi,ylo,yhi)
                    
                    # replace the content values 
                    # for ITL these are unchanged,
                    #      xlo xhi ylo  yhi 
                    # i.e. 0  1999 0     508  
                    #      . .     509  1017
                    #      0  1999 3563 4071
                    #      . . 
                    # 
                    # for E2V these are changed 
                    # i.e.  0  1999 0     508  --> 0     2001   0    511
                    #       . .     509  1017
                    #       0  1999 3563 4071  --> 2000  3999  3563  4071 
                    

                    newSplitContent[1] = str(xlo)
                    newSplitContent[2] = str(xhi)
                    newSplitContent[3] = str(ylo)
                    newSplitContent[4] = str(yhi)

                    
                    # update serialread / parallelread for E2V.
                    # for ITL serialread is always 1, 
                    # and parallelread is 1 or -1 
                    # for E2V, serialread is -1 when parallelread is -1 
                    serialread = splitContent[5]
                    parallelread = splitContent[6]
                    
                    if lsstDetectors.getPhysicalType() == 'E2V':
                        #print('%s is E2V'%newSensorAmpName)
                        #print('%s %s'%(serialread, parallelread))
                        if parallelread == '-1':
                            serialread = '-1'
                            #print('--> %s %s'%(serialread, parallelread))
                        newSplitContent[5] = serialread
                        
                     
                    
                    # update the overscan / prescan values 
                    # preserving here phosim's rotation of 
                    # 90 degrees to the left wrt lsstCam ... 
                    
                    #      phosim        < ---        lsstCam 
                    #
                    # serial overscan              parallel prescan
                    # serial prescan               parallel overscan
                    # parallel overscan            serial overscan
                    # parallel prescan             serial prescan 
                    
                    # eg.      3             0            10            10 
                    #     parallel pre | serial over | serial pre | parallel over 
                    #          A             B            C              D 
                    
                    # ITL      3             0            48            32
                    # E2V     10             0            46            54
                    
                    # these numbers can be achieved from :
                    #oldA,oldB,oldC,oldD = splitContent[15], splitContent[16], splitContent[17], splitContent[18]
                    
                    bbox = amp.getRawSerialPrescanBBox()
                    A = bbox.getWidth()
                    B = '0'

                    bbox = amp.getRawParallelOverscanBBox()
                    C = bbox.getHeight()

                    bbox = amp.getRawSerialOverscanBBox()
                    D = bbox.getWidth()
                    #print(A,B,C,D)
                    newSplitContent[15] = str(A) # parallel prescan for phosim
                    newSplitContent[16] = B
                    newSplitContent[17] = str(C) # serial prescan for phosim
                    newSplitContent[18] = str(D) # parallel overscan for phosim 
                    
        # either way, make new content by joining the elements of 
        # the updated split content: 
        newContent = ' '.join(newSplitContent)+'\n'
        #print(newContent)
        
        
        # add the new content line to the new dictionary 
        newSensorData[newName].append(newContent)
        
# combine the new content as a long list of lines 
newContentLines = []
for sensorName in newSensorData.keys():
    for line in newSensorData[sensorName]:
        newContentLines.append(line)
        
# write the unchanged header, and the new content,
# into a new segmentation.txt file 
fname = "/project/scichris/aos/phosim_syseng4/data/comcam/segmentation_DM-29843.txt"
f = open(fname, "w")
f.writelines(headerLines)
f.writelines(newContentLines)
f.close()      
print('Saved as %s'%fname)