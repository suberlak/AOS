###########################################
#                                         # 
#         updatePhosimFunctions.py        #
#                                         #
###########################################


'''
Functions used by 

update_focalplanelayout_comcam.py
update_focalplanelayou_lsst.py

update_segmentation_comcam.py
update_segmentation_lsst.py
'''

def readFile(pathToFile, headerString = '#'):
    ''' Read file line-by-line  as two lists,
    one containing the header lines, 
    the other the content lines 
    
    Parameters:
    -----------
    pathToFile: `str`
        absolute path to file 
        
    Returns:
    ----------
    `list`, `list` 
        lists containing the header lines 
        and the content lines 
    '''
    with open(pathToFile) as f:
        allLines = f.readlines()

    headerLines=[]
    contentLines=[]

    # store the header part and content separately 
    for line in allLines:
        if line.startswith('#'):
            headerLines.append(line)
        else:
            contentLines.append(line)

    return headerLines, contentLines 

    
def getNewSensorName(sensorName):
    '''Get updated sensor names
    
    Parameters:
    -----------
    sensorName: `str` 
        input sensor name to translate 
    
    
    Returns:
    --------
    `str` 
        translated sensor name 
    '''
    # mapping 
    guideSensorMapping = {'R04_S10':'R04_SG0', 
                          'R04_S21':'R04_SG1', 
                          'R44_S01':'R44_SG0', 
                          'R44_S10':'R44_SG1', 
                          'R00_S12':'R00_SG0', 
                          'R00_S21':'R00_SG1', 
                          'R40_S01':'R40_SG0', 
                          'R40_S12':'R40_SG1'
                         }

    # C0 : intra-focal  - SW1 
    # C1 : extra-focal  - SW0 
    cornerSensorMapping = {'R04_S20_C0':'R04_SW1',
                           'R04_S20_C1':'R04_SW0',
                           'R44_S00_C0':'R44_SW1',
                           'R44_S00_C1':'R44_SW0',
                           'R00_S22_C0':'R00_SW1',
                           'R00_S22_C1':'R00_SW0',
                           'R40_S02_C0':'R40_SW1',
                           'R40_S02_C1':'R40_SW0'
                          }
    
    # alter the name of the corner sensors to be able to
    # access lsstCam data 
    if sensorName in cornerSensorMapping.keys():
        newName = cornerSensorMapping[sensorName]
        #print('corner: %s --> %s'%(sensorName, newName))
        
    elif sensorName in guideSensorMapping.keys():
        newName = guideSensorMapping[sensorName]
        #print('guide: %s --> %s ' %(sensorName, newName))
        
    # otherwise, name remains unchanged 
    else: 
        newName = sensorName
    
    return newName 
    
    
def storeSensorDataAsDictionary(contentLines):
    '''Store the content of phoSim segmentation
    file as dictionary, with sensor names as keys
    
    Parameters:
    ----------
    contentLines: `list` 
        list of content lines (not starting with #)
        
    Returns:
    ----------
    sensorData: `dict` '''
    
    # store the segmentation.txt data into a dictionary...
    # it makes it easier to access the data for 
    # each element and compare to the old one ... 

    sensorData = {}
    # gather data from all sensors  into a dictionary  
    for i in range(len(contentLines)):
        # iterate over lines
        line = contentLines[i]

        # the beginning of the short line is a sensor name.
        # store the name 
        # if the line is longer, it continues using that name
        # until the next short line 
        if len(line)<50:

            sensorName = line.split(' ')[0]
            #print('Short line! Updating name to %s'%sensorName)
            # initialize the dict for that sensor...
            sensorData[sensorName] = []

        # this continues with each line that starts with that
        # sensor name ... 
        # it also works for corner sensors, because 
        # _C0  contain C0x amps,
        # _C1 contain C1x amps... 
        if line.split(' ')[0].startswith(sensorName):
            #print(line)
            sensorData[sensorName].append(line)
            
    return sensorData 