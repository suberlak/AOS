import updatePhosimFunctions as up
from lsst.obs.lsst import LsstCam
import numpy as np 


###########################################
#                                         # 
#  Update segmentation for lsstCam        #
#                                         #
###########################################

cornerSensors = ['R04_S20_C0',
               'R04_S20_C1',
               'R44_S00_C0',
               'R44_S00_C1',
               'R00_S22_C0',
               'R00_S22_C1',
               'R40_S02_C0',
               'R40_S02_C1']
        
        
# phosim corner wavefront
# amplifier names to lsst cam
ch_map = {'00': '17',
        '01': '16',
        '02': '15',
        '03': '14',
        '04': '13',
        '05': '12',
        '06': '11',
        '07': '10',
        '10': '17',
        '11': '16',
        '12': '15',
        '13': '14',
        '14': '13',
        '15': '12',
        '16': '11',
        '17': '10'}

# read all the lines to a list : 
pathToFile = '/project/scichris/aos/phosim_syseng4/data/lsst/segmentation_old.txt'

# read the content of segmentation.txt file
# and store in a  dictionary
headerLines, contentLines = up.readFile(pathToFile)
sensorData = up.storeSensorDataAsDictionary(contentLines)

# initialize the camera 
camera = LsstCam().getCamera()
    
# Update the data 
newSensorData  = {}
  
# CHANGE SERIAL?
change_serial = True

# REMAP CHANNELS?
remap_channels = True 


# the default:
ticket_number = 'DM-28557' # lsstCam 
    
    
if change_serial :
    print('Changing serial direction ')
    ticket_number = 'DM-30367' # both: obs_lsst orientation
else:
    print('Keeping  original serial direction')
    
if remap_channels:
    print('Remapping channel names')
    ticket_number = 'DM-30367' # both: obs_lsst orientation
else:
    print('Keeping original amp channel names ')
    
    
for sensorName in list(sensorData.keys()): # [:4] to test a few ... 
    #print('Running %s'%sensorName)
    newSensorName = up.getNewSensorName(sensorName)
    
    # get the lsstCam data for that sensor 
    #print(newSensorName)
    lsstCamDetectors = camera.get(newSensorName)
    
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
            bbox = lsstCamDetectors.getBBox()
            px_y = bbox.getDimensions()[0]
            px_x = bbox.getDimensions()[1]
            newSplitContent[2] = str(px_x)
            newSplitContent[3] = str(px_y)
            
            # print information...
            old_px_x = splitContent[2]
            old_px_y = splitContent[3]
            #if old_px_x !=  str(px_x) : 
                #print(sensorName, old_px_x, old_px_y,  '--> ', newSensorName, px_x, px_y)
            
        if len(content)>40:
            #continue
            #print(len(content), content)
            
            sensorAmpName = splitContent[0]
            ampName = sensorAmpName.split('_')[2]
            ampGain = splitContent[7]
            #ampBias  = splitContent[9]
            ampReadNoise = splitContent[11]
            #ampDarkCurrent = splitContent[13]

            #print('%s: gain %s, bias %s, noise %s, dc %s'% (sensorAmpName, ampGain, ampBias, ampReadNoise, ampDarkCurrent ))
            #print(content)
            #newContent = splitContent.copy()
            
            # change the name of corner sensor amplifiers 
            if sensorName in cornerSensors:
                ampName = 'C%s'%ch_map[ampName[1:]]
                
            # for the main raft the amp names are correct 
            newSensorAmpName = '%s_%s'%(newSensorName, ampName)
            newSplitContent[0] = newSensorAmpName
            
            #
            # begin loop lsstCam mapper amp info 
            #
            # read the lsstCam information for that sensor / amp ... 
            for amp in lsstCamDetectors:
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
                    
                    if lsstCamDetectors.getPhysicalType() == 'E2V':
                        #print('%s is E2V'%newSensorAmpName)
                        #print('%s %s'%(serialread, parallelread))
                        if parallelread == '-1':
                            serialread = '-1'
                            #print('--> %s %s'%(serialread, parallelread))
                        newSplitContent[5] = serialread
                        
                     
                    # switch from 1 -1 to  1  1  so that no up-down flips needed ...
                    if lsstCamDetectors.getPhysicalType() == 'ITL_WF':
                        #print('%s is ITL_WF'%newSensorAmpName)
                        #print('%s %s'%(serialread, parallelread))
                        if parallelread == '-1':
                            parallelread = '1'
                            #print('--> %s %s'%(serialread, parallelread))
                        # update the value ...
                        newSplitContent[6] = parallelread
                    
                    if change_serial: # multiply by -1 
                        newSplitContent[5] = str(int(serialread) * -1)

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
                    #B = '0'

                    bbox = amp.getRawParallelOverscanBBox()
                    C = bbox.getHeight()

                    bbox = amp.getRawSerialOverscanBBox()
                    D = bbox.getWidth()
                    #print(A,B,C,D)
                    newSplitContent[15] = str(A) # parallel prescan for phosim
                    newSplitContent[16] = '0'
                    newSplitContent[17] = str(C) # serial prescan for phosim
                    newSplitContent[18] = str(D) # parallel overscan for phosim 
            #
            # end loop lsstCam mapper amp info 
            #
            
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
fname = f"/project/scichris/aos/phosim_syseng4/data/lsst/segmentation_{ticket_number}.txt"
f = open(fname, "w")
f.writelines(headerLines)
f.writelines(newContentLines)
f.close()      
print('Saved as %s'%fname)