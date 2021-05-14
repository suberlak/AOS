#!/bin/bash
butler create DATA
butler register-instrument DATA/ lsst.obs.lsst.LsstComCam
butler ingest-raws DATA repackaged/
butler define-visits DATA/ lsst.obs.lsst.LsstComCam
butler write-curated-calibrations DATA/ lsst.obs.lsst.LsstComCam
pipetask run -j 9 -b DATA/ -i LSSTComCam/raw/all,LSSTComCam/calib         -p testPipeline.yaml --register-dataset-types --output-run run1