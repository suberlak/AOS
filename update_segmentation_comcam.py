import updatePhosimFunctions as up
from lsst.obs.lsst import LsstComCam
import numpy as np 

# read all the lines to a list.
# as a base for all changes is the copy of segmentation.txt from before I started 
# to change anything that had to do with orientation,
# i.e. as of v1.0.4 https://github.com/lsst-ts/phosim_syseng4/releases/tag/v1.0.4 

# The geometry update for LSSTCam was https://github.com/lsst-ts/phosim_syseng4/releases/tag/v1.0.5
# The geometry update for LSSTComCam was https://github.com/lsst-ts/phosim_syseng4/releases/tag/v1.0.6
# Fix gain, variance for LSSTComCam https://github.com/lsst-ts/phosim_syseng4/releases/tag/v1.0.7 

pathToFile = '/project/scichris/aos/phosim_syseng4/data/comcam/segmentation_old.txt'

# read the content of segmentation.txt file
# and store in a  dictionary
headerLines, contentLines = up.readFile(pathToFile)
sensorData = up.storeSensorDataAsDictionary(contentLines)

# initialize the camera 
camera = LsstComCam().getCamera()
    
# Update the data 
newSensorData  = {}
  
    
# DEROTATE PHOSIM? 
derotate_phosim = False


# CHANGE SERIAL?
change_serial = True

# REMAP CHANNELS?
remap_channels = True 

if derotate_phosim:
    print('Derotating phosim')
    ticket_number = 'DM-30367' # both: obs_lsst orientation
else:
    print('Keeping phosim orientation')
    ticket_number = 'DM-29264' # comcam 
    
    
    
# running only over CCDs in the mapper,
# even though originally that comcam segmentation.txt 
# had data for guide and corner sensors ... (basically it was 
# the same as for data/lsst/segmentation.txt... )

for sensorName in camera.getNameIter():
    print('Running %s'%sensorName)
    newSensorName = up.getNewSensorName(sensorName)
    
    # get the lsstComCam data for that sensor 
    #print(newSensorName)
    lsstDetectors = camera.get(newSensorName)
    
    # initialize a list for the new data 
    newSensorData[newSensorName] = []
    
    for content in sensorData[sensorName]:
        splitContent = content.split() # by default splits by whitespace, omitting empty strings 
        newSplitContent = splitContent.copy()
        
        # first line contains sensor name, number of amps, and ccd dimensions, eg.
        # 'R00_S12', '16', '4000', '4072'
        if len(content)<40:
            # update sensor name 
            newSplitContent[0] = newSensorName
            
            # update sensor px_x, px_y 
            bbox = lsstDetectors.getBBox()
            if derotate_phosim:
                print('Rotation x,y dimension to 4072x4000')
                px_x = bbox.getDimensions()[0]
                px_y = bbox.getDimensions()[1]
                
            else:
                print("Keeping original dimension x,y of 4000x4072")
                px_y = bbox.getDimensions()[0]
                px_x = bbox.getDimensions()[1]
                
            newSplitContent[2] = str(px_x)
            newSplitContent[3] = str(px_y)
            
            # print information...
            #old_px_x = splitContent[2]
            #old_px_y = splitContent[3]
            #if old_px_x !=  str(px_x) : 
                #print(sensorName, old_px_x, old_px_y,  '--> ', newSensorName, px_x, px_y)
            
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
            #  below,  newSensorName is eg. R22_S02
            # and ampName is eg. C02 
            newSensorAmpName = '%s_%s'%(newSensorName, ampName)
            newSplitContent[0] = newSensorAmpName
            
            # read the mapper information for that sensor / amp ... 
            # looping over the information in the mapper 
            # this gets executed only once 
            # i.e. once we find the amp.getName() which has 
            # the same name as the amp in that content line 
            for amp in lsstDetectors:
                if amp.getName() == ampName:
                    
                    # update gain and readNoise 
                    newAmpGain = str(amp.getGain())
                    if amp.getGain() == 0:
                        print(newSensorName, ampName, amp.getGain())
                    newAmpReadNoise = str(amp.getReadNoise()) 
                    #print('%s: gain %s --> %s, noise %s --> %s'%(ampName, ampGain, newAmpGain, 
                    #                                             ampReadNoise, newAmpReadNoise))
                    #if float(newAmpGain) > 2 :
                        #print(newSensorName)
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
                    
                    if derotate_phosim:
                        # make phosim system the same as obs_lsst , i.e. 
                        # X,Y (obs_lsst) --> X,Y (phosim)
                        xlo = bbox.getMinX()
                        xhi = bbox.getMaxX()
                        ylo = bbox.getMinY()
                        yhi = bbox.getMaxY()
                        
                    else:
                        # assign the dimensions to match the phoSim Transposed system ,i.e.
                        # X,Y (obs_lsst) --> Y,X (phosim) 
                        xlo = bbox.getMinY()
                        xhi = bbox.getMaxY()
                        ylo = bbox.getMinX()
                        yhi = bbox.getMaxX()
                    
                    #print(amp.getName(), xlo,xhi,ylo,yhi)
#                     xlo_old = splitContent[1]
#                     xhi_old = splitContent[2]
#                     ylo_old = splitContent[3]
#                     yhi_old = splitContent[4]
                    
                    
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
                    if change_serial:
                        serialread = '-1'
                        newSplitContent[5] = serialread
                        
                    if lsstDetectors.getPhysicalType() == 'E2V':
                        print('%s is E2V'%newSensorAmpName)
                        print('%s %s'%(serialread, parallelread))
                        if parallelread == '-1':
                            serialread = '-1'
                            #print('--> %s %s'%(serialread, parallelread))
                        newSplitContent[5] = serialread
                        
                     
                    if derotate_phosim:
                        # update the overscan / prescan values,
                        # keeping the lsstCam orientation
                        
                        # phosim          < --  obs_lsst 
                        # parallel prescan      parallel prescan 
                        # parallel overscan     parallel overscan
                        # serial prescan        serial prescan 
                        # serial overscan       serial overscan 
                        
                        
                        # eg.   3             0            10            10 
                        #     parallel pre | serial over | serial pre | parallel over 
                        #       A             B            C              D 
                        # 
                        # ITL   0          |  32          | 3         | 48 
                        
                        # parallel prescan 
                        A = '0'
                        
                        # serial overscan
                        bbox = amp.getRawSerialOverscanBBox()
                        B = bbox.getWidth()
                        
                        # serial prescan
                        bbox = amp.getRawSerialPrescanBBox()
                        C = bbox.getWidth()
                        
                        # parallel overscan
                        bbox = amp.getRawParallelOverscanBBox()
                        D = bbox.getHeight()
                        
                    else:
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

                        # parallel prescan from serial prescan 
                        bbox = amp.getRawSerialPrescanBBox()
                        A = bbox.getWidth()
                        B = '0'

                        bbox = amp.getRawParallelOverscanBBox()
                        C = bbox.getHeight()

                        bbox = amp.getRawSerialOverscanBBox()
                        D = bbox.getWidth()

                    #print(A,B,C,D)
                    newSplitContent[15] = str(A) # parallel prescan for phosim
                    newSplitContent[16] = str(B)
                    newSplitContent[17] = str(C) # serial prescan for phosim
                    newSplitContent[18] = str(D) # parallel overscan for phosim 
                    
                    
                    
                
            if remap_channels: 
                # change names from 17,16.. to 10,11.. 
                # and 07,06... to 00, 01 ... ,
                # which amounts to flipping along x-axis 

                mapping_dict = {# top row
                               'C00':'C07',
                               'C01':'C06',
                               'C02':'C05',
                               'C03':'C04',
                               'C04':'C03',
                               'C05':'C02',
                               'C06':'C01',
                               'C07':'C00',
                                # bottom row 
                               'C10':'C17',
                               'C11':'C16',
                               'C12':'C15',
                               'C13':'C14',
                               'C14':'C13',
                               'C15':'C12',
                               'C16':'C11',
                               'C17':'C10'
                               }
                newAmpName = mapping_dict[ampName]
                print(f'Changing {ampName} to {newAmpName}')
                newSensorAmpName = '%s_%s'%(newSensorName, newAmpName)
                newSplitContent[0] = newSensorAmpName

                        
        # either way, make new content by joining the elements of 
        # the updated split content: 
        newContent = ' '.join(newSplitContent)+'\n'
        #print(newContent)
        
        
        # add the new content line to the new dictionary 
        newSensorData[newSensorName].append(newContent)
        
# combine the new content as a long list of lines 
newContentLines = []
for sensorName in newSensorData.keys():
    for line in newSensorData[sensorName]:
        newContentLines.append(line)
        
# write the unchanged header, and the new content,
# into a new segmentation.txt file 
fname = f"/project/scichris/aos/phosim_syseng4/data/comcam/segmentation_{ticket_number}.txt"
f = open(fname, "w")
f.writelines(headerLines)
f.writelines(newContentLines)
f.close()      
print('Saved as %s'%fname)