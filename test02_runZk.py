import os
from lsst.ts.phosim.utils.Utility import getPhoSimPath
import subprocess
import shutil
from lsst.ts.wep.Utility import runProgram
from lsst.daf import butler as dafButler
import astropy.io.ascii
import numpy as np
import argparse


def writeInstFile(
    pertDir,
    obsId=9006001,
    dof=-1500.0,
    defocal="Extra",
    rotSkyPos=0,
    rotTelPos=0,
    mjd=59580.0,
    ra=0.0,
    dec=0.0,
    simSeed=1000,
):
    # inst file
    filePath = os.path.join(pertDir, f"star{defocal}.inst")
    # obsId = 9006001
    idx = 10
    # dof = -1500.0
    # Observation Parameters
    content = ""
    content += "Opsim_obshistid %d \n" % obsId
    content += "Opsim_filter 1 \n"
    content += "mjd %.10f \n" % mjd
    content += "SIM_SEED %d \n" % simSeed

    # Add the sky information
    content += "rightascension %.6f \n" % ra
    content += "declination %.6f \n" % dec
    # content += "rotskypos %.6f \n" % rotSkyPos
    content += "rottelpos  %.6f \n" % rotTelPos
    content += "SIM_VISTIME 15.0 \n"
    content += "SIM_NSNAP 1 \n"
    content += "Opsim_rawseeing 0.69 \n"
    content += "move %d %7.4f \n" % (idx, dof)
    content += "camconfig 3 \n"
    content += "object  0  0.000000  0.000000 15.000000  ../sky/sed_500.txt 0.0 0.0 0.0 0.0 0.0 0.0 star 0.0 none none \n"
    mode = "w"

    fid = open(filePath, mode)
    fid.write(content)

    return filePath


def writeCmdFile(pertDir, surfaceId=0, zernikeNumber=4, zernikeInNm=250):

    zernikeInMilimeters = zernikeInNm * 1e-6  # nm to mm
    filePath = os.path.join(pertDir, f"star.cmd")

    content = ""
    content += "backgroundmode 0 \n"
    content += "raydensity 0.0 \n"
    content += "perturbationmode 1 \n"
    content += "trackingmode 0 \n"
    content += "cleartracking \n"
    content += "clearclouds \n"
    content += "lascatprob 0.0 \n"
    content += "contaminationmode 0 \n"
    content += "diffractionmode 1 \n"
    content += "straylight 0 \n"
    content += "detectormode 0 \n"
    content += "centroidfile 1 \n"
    content += f"izernike {surfaceId} {zernikeNumber-1} {zernikeInMilimeters} \n"

    mode = "w"

    fid = open(filePath, mode)
    fid.write(content)

    return filePath


def getPhosimArgs(
    imageDir,
    instFilePath,
    cmdFilePath,
    numPro=55,
    instrument="comcam",
    e2ADC=1,
    sensorName="R22_S11",
    defocal="Extra",
):
    """Get the arguments needed to run the PhoSim.

    Parameters
    ----------
    e2ADC : int, optional
        Whether to generate amplifier images (1 = true, 0 = false). (the
        default is 1.)


    """
    outputImgDir = os.path.join(imageDir, defocal.lower())
    if not os.path.exists(outputImgDir):
        os.makedirs(outputImgDir)
    logFileName = f"star{defocal}Phosim.log"

    logFilePath = os.path.join(outputImgDir, logFileName)

    # Prepare the argument list
    argString = "%s -i %s -e %d" % (instFilePath, instrument, e2ADC)

    # if extraCommandFile is not None:
    argString += " -c %s" % cmdFilePath

    if numPro > 1:
        argString += " -p %d" % numPro

    #     if numThread > 1:
    #         argString += " -t %d" % numThread
    argString += " -s %s" % sensorName
    argString += " -o %s" % outputImgDir
    argString += " -w %s" % outputImgDir
    argString += " > %s 2>&1" % logFilePath

    return argString


def runPhoSim(argString):

    # Path of phosim.py script
    phosimRunPath = os.path.join(getPhoSimPath(), "phosim.py")

    # Command to execute the python
    command = " ".join(["python", phosimRunPath])

    # Arguments for the program
    command += " " + argString

    print("Running ", command)
    # Call the program w/o arguments
    if subprocess.call(command, shell=True) != 0:
        raise RuntimeError("Error running: %s" % command)


def repackagePistonCamImgs(
    outputImgDir, instName="comcam", isEimg=False, defocalDistInMm=1500.0
):
    """Repackage the images of piston camera (ComCam and LSST FAM) from
    PhoSim for processing.
    FAM: Full-array mode.
    Parameters
    ----------
    instName : `str`
        Instrument name.
    isEimg : bool, optional
        Is eimage or not. (the default is False.)
    """

    # Make a temporary directory
    tmpDirPath = os.path.join(outputImgDir, "tmp")
    if not os.path.exists(tmpDirPath):
        os.makedirs(tmpDirPath)

    intraFocalDirName = "intra"
    extraFocalDirName = "extra"
    for imgType in (intraFocalDirName, extraFocalDirName):

        # Repackage the images to the temporary directory
        command = "phosim_repackager.py"
        phosimImgDir = os.path.join(outputImgDir, imgType)
        argstring = "%s --out_dir=%s" % (phosimImgDir, tmpDirPath)
        argstring += f" --inst {instName} "
        if isEimg:
            argstring += " --eimage"
        focusz = defocalDistInMm * 1e3 * (-1.0 if imgType == intraFocalDirName else 1.0)
        argstring += f" --focusz {focusz}"

        runProgram(command, argstring=argstring)

        # Remove the image data in the original directory
        argstring = "-rf %s/*.fits*" % phosimImgDir
        runProgram("rm", argstring=argstring)

        # Put the repackaged data into the image directory
        argstring = "%s/*.fits %s" % (tmpDirPath, phosimImgDir)
        runProgram("mv", argstring=argstring)

    # Remove the temporary directory
    shutil.rmtree(tmpDirPath)


def ingestData(
    butlerRootPath,
    outputImgDir,
    seqNumExtra,
    seqNumIntra,
    instName="comcam",
):
    """Ingest data into a gen3 data Butler.
    Parameters
    ----------
    butlerRootPath : str
        Path to the butler repository.
    instName : str
        Instrument name.
    """

    intraRawExpDir = os.path.join(outputImgDir, "intra")

    extraRawExpDir = os.path.join(outputImgDir, "extra")

    runProgram(f"butler ingest-raws {butlerRootPath} {intraRawExpDir}")
    runProgram(f"butler ingest-raws {butlerRootPath} {extraRawExpDir}")
    if instName == "comcam":
        cmd = f"butler define-visits {butlerRootPath} lsst.obs.lsst.LsstComCam"
        cmd += f' --where "exposure.seq_num IN ({seqNumExtra},{seqNumIntra})"'
        runProgram(cmd)

    else:
        runProgram(f"butler define-visits {butlerRootPath} lsst.obs.lsst.LsstCam")


def writeWepConfiguration(instName, pipelineYamlPath):
    """Write wavefront estimation pipeline task configuration.
    Parameters
    ----------
    instName: `str`
        Name of the instrument this configuration is intended for.
    pipelineYamlPath: `str`
        Path where the pipeline task configuration yaml file
        should be saved.
    """

    butlerInstName = "ComCam" if instName == "comcam" else "Cam"

    with open(pipelineYamlPath, "w") as fp:
        fp.write(
            f"""# This yaml file is used to define the tasks and configuration of
# a Gen 3 pipeline used for testing in ts_wep.
description: wep basic processing test pipeline
# Here we specify the corresponding instrument for the data we
# will be using.
instrument: lsst.obs.lsst.Lsst{butlerInstName}
# Then we can specify each task in our pipeline by a name
# and then specify the class name corresponding to that task
tasks:
  isr:
    class: lsst.ip.isr.isrTask.IsrTask
      # Below we specify the configuration settings we want to use
      # when running the task in this pipeline. Since our data doesn't
      # include bias or flats we only want to use doApplyGains and
      # doOverscan in our isr task.
    config:
      connections.outputExposure: 'postISRCCD'
      doBias: False
      doVariance: False
      doLinearize: False
      doCrosstalk: False
      doDefect: False
      doNanMasking: False
      doInterpolate: False
      doBrighterFatter: False
      doDark: False
      doFlat: False
      doApplyGains: True
      doFringe: False
      doOverscan: True
  generateDonutCatalogWcsTask:
    class: lsst.ts.wep.task.GenerateDonutCatalogWcsTask.GenerateDonutCatalogWcsTask
    # Here we specify the configurations for pointing that we added into the class
    # GenerateDonutCatalogWcsTaskConfig.
    config:
      filterName: 'g'
      referenceSelector.doMagLimit: True
      referenceSelector.magLimit.maximum: 15.90
      referenceSelector.magLimit.minimum: 8.74
      referenceSelector.magLimit.fluxField: 'g_flux'
      doDonutSelection: True
      donutSelector.fluxField: 'g_flux'
  estimateZernikesScienceSensorTask:
    class: lsst.ts.wep.task.EstimateZernikesScienceSensorTask.EstimateZernikesScienceSensorTask
    config:
      # And here we specify the configuration settings originally defined in
      # EstimateZernikesScienceSensorTaskConfig.
      donutTemplateSize: 160
      donutStampSize: 160
      initialCutoutPadding: 40
"""
        )


def runWep(extraObsId, intraObsId, butlerRootPath, instName="comcam", numPro=20):
    """Run wavefront estimation pipeline task.
    Parameters
    ----------
    extraObsId : `int`
        Extra observation id.
    intraObsId : `int`
        Intra observation id.
    butlerRootPath : `str`
        Path to the butler gen3 repos.
    instName : `str`
        Instrument name.
    numPro : int, optional
        Number of processor to run DM pipeline. (the default is 1.)
    Returns
    -------
    listOfWfErr : `list` of `SensorWavefrontError`
        List of SensorWavefrontError with the results of the wavefront
        estimation pipeline for each sensor.
    """
    visitIdOffset = 4021114100000
    butlerInstName = "ComCam" if instName == "comcam" else "Cam"
    pipelineYaml = f"{instName}Pipeline.yaml"
    pipelineYamlPath = os.path.join(butlerRootPath, pipelineYaml)

    butler = dafButler.Butler(butlerRootPath)

    if f"LSST{butlerInstName}/calib" not in butler.registry.queryCollections():

        print("Ingesting curated calibrations.")

        runProgram(
            f"butler write-curated-calibrations {butlerRootPath} lsst.obs.lsst.Lsst{butlerInstName}"
        )

    writeWepConfiguration(instName, pipelineYamlPath)

    runProgram(
        f"pipetask run -b {butlerRootPath} "
        f"-i refcats,LSST{butlerInstName}/raw/all,LSST{butlerInstName}/calib/unbounded "
        f"--instrument lsst.obs.lsst.Lsst{butlerInstName} "
        f"--register-dataset-types --output-run ts_phosim_{extraObsId} -p {pipelineYamlPath} -d "
        f'"exposure IN ({visitIdOffset+extraObsId}, {visitIdOffset+intraObsId})" -j {numPro} --clobber-outputs'
    )


def generateButler(butlerRootPath, instName="comcam"):
    """Generate butler gen3.
    Parameters
    ----------
    butlerRootPath: `str`
        Path to where the butler repository should be created.
    instName: `str`
        Name of the instrument.
    """

    print(f"Generating butler gen3 in {butlerRootPath} for {instName}")

    runProgram(f"butler create {butlerRootPath}")

    if instName == "comcam":
        print("Registering LsstComCam")
        runProgram(
            f"butler register-instrument {butlerRootPath} lsst.obs.lsst.LsstComCam"
        )
    else:
        print("Registering LsstCam")
        runProgram(f"butler register-instrument {butlerRootPath} lsst.obs.lsst.LsstCam")


def generateRefCatalog(
    butlerRootPath,
    instName="comcam",
    pathSkyFile="/project/scichris/aos/rotation_DM-34065/singleStarBoresight00.txt",
):
    """Generate reference star catalog.
    Parameters
    ----------
    instName: `str`
        Name of the instrument.
    butlerRootPath: `str`
        Path to the butler gen3 repository.
    pathSkyFile: `str`
        Path to the catalog star file.
    """
    print("Creating reference catalog.")

    catDir = os.path.join(butlerRootPath, "skydata")
    skyFilename = os.path.join(catDir, "sky_data.csv")
    catConfigFilename = os.path.join(catDir, "cat.cfg")
    catRefConfigFilename = os.path.join(catDir, "convertRefCat.cfg")

    os.mkdir(catDir)

    # Read sky file and convert it to csv
    skyData = astropy.io.ascii.read(pathSkyFile)
    # Constructing the catalog of stars to use in the wavefront estimation
    # pipeline. Here it assigns the g filter. Since this is only for target
    # selection it really doesn't matter which filter we select, as long
    # as it is a valid one.
    skyData.rename_column("Mag", "g")

    skyData.write(skyFilename, format="csv", overwrite=True)

    with open(os.path.join(catDir, "_mapper"), "w") as fp:
        fp.write("lsst.obs.lsst.LsstCamMapper\n")

    with open(catConfigFilename, "w") as fp:
        fp.write(
            """config.ra_name='Ra'
config.dec_name='Decl'
config.id_name='Id'
config.mag_column_list=['g']
"""
        )
    with open(catRefConfigFilename, "w") as fp:
        fp.write('config.datasetIncludePatterns = ["ref_cat", ]\n')
        fp.write('config.refCats = ["cal_ref_cat"]\n')

    runProgram(
        f"ingestReferenceCatalog.py {catDir} {skyFilename} --configfile {catConfigFilename}"
    )

    runProgram(
        f"butler convert --gen2root {catDir} --config-file {catRefConfigFilename} {butlerRootPath}"
    )


def makeJcountDic():
    # Make the dict of zk to jcount
    jcount = 0
    jcountDic = {}
    for zkNumber in np.arange(4, 23):
        # print(zkNumber, jcount)
        jcountDic[zkNumber] = jcount
        jcount += 1
    return jcountDic


def main(
    baseOutputDir,
    rotCamInDeg,
    zernikeInNm,
    numPro,
    zkMin,
    zkMax,
    dryrun,
    sensorName="R22_S11",
):

    if dryrun:
        print(
            "\n Will use the following arguments:",
            f"\nbaseOutputDir = {baseOutputDir}",
            f"\nrotCamInDeg={rotCamInDeg}",
            f"\nzernikeInNm={zernikeInNm}",
            f"\nnumPro={numPro}",
            f"\nzkMin={zkMin}",
            f"\nzkMax={zkMax}",
        )
    else:  # run the actual program

        jcountDic = makeJcountDic()
        butlerRootPath = os.path.join(baseOutputDir, "phosimData")
        butlerFilePath = os.path.join(butlerRootPath, "butler.yaml")

        # only run once
        if not os.path.exists(butlerFilePath):
            # generate butler gen3 repo and refCat once
            generateButler(butlerRootPath)
            generateRefCatalog(butlerRootPath)

        for zkNumber in np.arange(zkMin, zkMax + 1):

            # jcount = jcountDic[zkNumber]
            # print(jcount, zkNumber)

            if rotCamInDeg < 0:  # so -60 becomes 60n
                titleDeg = f"{str(rotCamInDeg)[1:]}n"
            else:
                titleDeg = f"{rotCamInDeg}"

            rotZerDir = os.path.join(baseOutputDir, f"rot{titleDeg}_zk{zkNumber}")
            pertDir = os.path.join(rotZerDir, "pert")
            imageDir = os.path.join(rotZerDir, "img")

            # this will change with each rotCam / izernike combination
            # eg. 9004512 for 45 deg zk 12 ...

            obsId = int(f"90{str(rotCamInDeg).zfill(3)}{str(zkNumber).zfill(2)}")
            # str(rotCamInDeg).zfill(3) makes 45 --> 045
            # while str(zkNumber).zfill(2) makes zk3 --> 03
            # keeping it that way ensures that SEQNUM is < 100k, i.e.
            # for rotCam of 120 and zk 13 seqnum is 12013.
            # this is a hardcoded limit - can't have more than 100k obs in a day,
            # because SEQNUM is understood as a sequential counter of observations
            # taken in a given day....

            print(f"zkNumber={zkNumber}, rotCamInDeg={rotCamInDeg}, obsId={obsId}")

            # that way with identical rotCamInDeg eg 105
            # intra-exposures span the last two digits of 105 34:52,
            # and extra-exposures span 105 14:32
            # this works because there are 19 zk values to go through
            extraObsId = obsId + 10
            intraObsId = obsId + 30

            seqNumExtra = int(str(extraObsId)[-5:])
            seqNumIntra = int(str(intraObsId)[-5:])
            print(f"seqNumExtra={seqNumExtra}, seqNumIntra={seqNumIntra}")

            if not os.path.exists(pertDir):
                os.makedirs(pertDir)

            obsIdList = {-1: extraObsId, 1: intraObsId}
            defocals = {-1: "Extra", 1: "Intra"}
            moveTenInMm = {-1: -1500.0, 1: 1500.0}
            argStringList = []

            # iterate over extra and intra-focal arguments...
            for ii in (-1, 1):

                # write inst file
                instFilePath = writeInstFile(
                    pertDir,
                    obsId=obsIdList[ii],
                    dof=moveTenInMm[ii],
                    defocal=defocals[ii],
                    rotTelPos=rotCamInDeg,
                )

                # write cmd file
                cmdFilePath = writeCmdFile(
                    pertDir,
                    surfaceId=0,
                    zernikeNumber=zkNumber,
                    zernikeInNm=zernikeInNm,
                )

                # get phosim args
                argString = getPhosimArgs(
                    imageDir,
                    instFilePath=instFilePath,
                    cmdFilePath=cmdFilePath,
                    defocal=defocals[ii],
                    instrument="comcam",
                    e2ADC=1,
                    sensorName=sensorName,
                    numPro=numPro,
                )

                argStringList.append(argString)

            # run  phosim
            for argString in argStringList:
                runPhoSim(argString)

            # Repackage the images based on the image type
            repackagePistonCamImgs(outputImgDir=imageDir)

            # Ingest images into butler gen3
            ingestData(
                butlerRootPath=butlerRootPath,
                outputImgDir=imageDir,
                seqNumExtra=seqNumExtra,
                seqNumIntra=seqNumIntra,
            )

            # run WEP
            runWep(extraObsId, intraObsId, butlerRootPath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run phosim for test02, input zk for star on the boresight"
    )
    parser.add_argument(
        "--baseOutputDir",
        "-o",
        type=str,
        default="/project/scichris/aos/rotation_DM-34065/test02_c3",
        help="Output directory",
    )
    parser.add_argument(
        "--rotCamInDeg",
        "-r",
        nargs=1,
        type=int,
        default=[0],
        help="Rotation angle in degrees (default is 0.)",
    )
    parser.add_argument(
        "--zkMin",
        type=int,
        nargs=1,
        default=[4],
        help="Min zk (defualt is 4)",
    )
    parser.add_argument(
        "--zkMax",
        type=int,
        nargs=1,
        default=[22],
        help="Max zk (defualt is 22)",
    )
    parser.add_argument(
        "--zernikeInNm",
        type=int,
        nargs=1,
        default=[250],
        help="Zernike value in nanometers (defualt is 250 nm)",
    )
    parser.add_argument(
        "--numPro",
        type=int,
        nargs=1,
        default=[55],
        help="Number of cores, passed as -p to phosim (defualt is 55)",
    )
    parser.add_argument(
        "--dryRun",
        default=False,
        action="store_true",
        help="Do not run any simulation, just print commands used. (default: False)",
    )
    args = parser.parse_args()
    main(
        args.baseOutputDir,
        rotCamInDeg=args.rotCamInDeg[0],
        zernikeInNm=args.zernikeInNm[0],
        numPro=args.numPro[0],
        zkMin=args.zkMin[0],
        zkMax=args.zkMax[0],
        dryrun=args.dryRun,
    )
