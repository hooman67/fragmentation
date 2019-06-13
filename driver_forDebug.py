###############################################################################
##################  FDMLAlgo DRIVER (for debugging)  ##########################
###############################################################################
#Put the path to fmdl_algo project here
import sys
sys.path.append('/media/hooman/hsSsdPartUbuntu/FM_PROJECT/FMDL_3.0/fmdl_algo')

import json
def run_FMDLAlgo():
    global fmdlAlgo
    global results
    global boxDetectorNetworkPath
    global roiDelineatorNetworkPath
    global saveResults
    global testFilePath
    global testFileName
    global saveResultsPath
    global shovelConfig
    global displayResults


    if "jpg" in testFilePath or "png" in testFilePath:
        with open(testFilePath, "rb") as imageFile:
                f = imageFile.read()
                b = bytearray(f)
                
                
        results = fmdlAlgo.execute(b, shovelConfig)


        print('\n\n\n**********************')
        print('results')
        print(results)
        print('ser')
        ser = json.dumps(results)
        print('**********************\n\n\n')

        
        print("\n\nDriver received this results:\n")
        print("resultsAreValid: " + str(results['valid']))
        print("imageWidthPx: " + str(results['imageWidthPx']))
        print("imageHeightPx: " + str(results['imageHeightPx']))
        print("Px2CmConversionFactor: " + str(results['pixel2CM_conversion_factor']))
        print("detected_bucketWidth_inPixels: " + str(results['detected_bucketWidth_inPixels']))
        print("approximated_roi_boundary: " + str(results['approximated_roi_boundary']))
        print("approximated_bucket_left_line: " + str(results['approximated_bucket_left_line']))
        print("approximated_bucket_right_line: " + str(results['approximated_bucket_right_line']))
        print("approximated_bucket_mid_line: " + str(results['approximated_bucket_mid_line']))
        print("effective_width_calculations_valid: " + str(results['effective_width_calculations_valid']))

        
        inputImage = PIL.Image.open(io.BytesIO(b))
        
        if displayResults:
            plt.title("inputImage")
            plt.imshow(inputImage)
            plt.show()

                
        if 'debug' in results:
            for res in results['debug']:
                imgdata = base64.b64decode(res['image'])
                img = PIL.Image.open(io.BytesIO(imgdata))

                if displayResults:
                    plt.title(res['description'])
                    plt.imshow(img)
                    plt.show()

                if res['description'] == 'output of roiDelineator, ouput of"                " boxDetector, and the approximated roi boundary points all overlayed on the input image':

                        imgdata = base64.b64decode(res['image'])
                        img = PIL.Image.open(io.BytesIO(imgdata))
                        
                        if saveResults:
                            img.save(saveResultsPath + testFileName)



if __name__ == "__main__":
    from fmdlAlgo.FMDLAlgo import FMDLAlgo


    runOnSingleImage = False
    saveResults = True
    displayResults = False
    

    ############################# FMDL3.0 Networks ####################################
    #boxDetectorNetworkPath = '/media/hooman/hsSsdPartUbuntu/FM_PROJECT/FMDL_3.0/ssdMobileNet_multiClass_bucket_rockInside_FineInside_NoTeeth_NoCase/try6-bbDetectorVersion-1/result_minScore09_ckpt-111269_bbDetectorVersion-1/output_inference_graph_chkpt-111269.pb/frozen_inference_graph.pb'
    #roiDelineatorNetworkPath = '/media/hooman/hsSsdPartUbuntu/FM_PROJECT/FMDL_3.0/UNet_Hydraulics/hsUnet_try21_1chan_roiDelineatorVersion-1/model-hsUNet-try21-1chan_roiDelineatorVersion-1.h5'
    #boxDetectorNetworkPath = '/home/hooman/randd/MachineLearning/Projects/FM-Cloud/BucyrusAndPnH-shovels/BucyrusAndPnH_Release-1_roiDelineatorVersion-1_bbDetectorVersion-1/bbDetector.pb'
    #roiDelineatorNetworkPath = '/home/hooman/randd/MachineLearning/Projects/FM-Cloud/BucyrusAndPnH-shovels/BucyrusAndPnH_Release-1_roiDelineatorVersion-1_bbDetectorVersion-1/roiDelineator.h5'
    #boxDetectorNetworkPath = '/home/hooman/mmHome/hooman/FMDL/FMDL_3.0/backhoeOpticalScene/boxDetectors/try2_ssdMultiClass_withInappInMatInside_bbDetectorVersion-1/result_defaultConfig_ckpt-660384/output_inference_graph/frozen_inference_graph.pb'
    #roiDelineatorNetworkPath = '/home/hooman/mmHome/hooman/FMDL/FMDL_3.0/backhoeOpticalScene/roiDelineators/try4-csvFrom-ssdTry3-withInapp-reducedNumberOfempyBuckets-BatchSize4_roiDelineatorVersion-1/model-hsUNet-Backhoe-try4-1chan.h5'
   #####################################################################################
    


    ############################### FMDL3.1 Networks ####################################
    boxDetectorNetworkPath = '/home/hooman/Desktop/currentFMDL3.1Nets/BucyrusAndPnH_Release-2_roiDelineatorVersion-1_bbDetectorVersion-1/bbDetector.pb'
    roiDelineatorNetworkPath = '/home/hooman/Desktop/currentFMDL3.1Nets/BucyrusAndPnH_Release-2_roiDelineatorVersion-1_bbDetectorVersion-1/roiDelineator.h5'
    #####################################################################################


    testdirPath = '/media/hooman/New Volume/FM_PROJECT_STORAGE/productionBugs/releaseingFML3.2Bugs/Frame/'
    saveResultsPath = '/media/hooman/New Volume/FM_PROJECT_STORAGE/productionBugs/releaseingFML3.2Bugs/preds/'
    

    testFilePath = '/media/hooman/New Volume/FM_PROJECT_STORAGE/productionBugs/releaseingFML3.2Bugs/FMDL_2018.07.31_11.15.15.png'
    testFileName = 'FMDL_2018.07.31_11.15.15.png'


    shovelConfig = {
        "measuredBucketWidthCM":              300,
        "boxDetectorScoreThresholdBucket":    0.5,
        "boxDetectorScoreThresholdMatInside": 0.5,
        "boxDetectorScoreThresholdCase":      0.5,
        "roiDelineatorScoreThreshold":        0.2,
        "minContourArea":                     12000,
        "closingKernelSize":                  7,
        "closingIterations":                  4,
        "erosionKernelSize":                  7,
        "erosionIterations":                  4,
        "roiBoundaryPointsReductionFactor":   0.01,
        "minBoundingBoxAspectRatio":          1.2,
        "maxBoundingBoxAspectRatio":          3,
        "minObjectsRequired":                 [["bucket"]],
        "intersectingRoiMaxIterations":       5,
        "intersectingRoiStepSize":            0.001,
        "effectiveWidthYcoordMultiplier":     0.5,
        "maxDiffBetweenAbsBucketEdgeSlopes":  3
    }


    '''
    shovelConfig = {
        "measuredBucketWidthCM":              300,
        "boxDetectorScoreThresholdBucket":    0.5,
        "boxDetectorScoreThresholdMatInside": 0.5,
        "boxDetectorScoreThresholdCase":      0.5,
        "roiDelineatorScoreThreshold":        0.5,
        "minContourArea":                     13000,
        "closingKernelSize":                  7,
        "closingIterations":                  0,
        "erosionKernelSize":                  7,
        "erosionIterations":                  0,
        "roiBoundaryPointsReductionFactor":   0.01,
        "maxBoundingBoxAspectRatio":          3,
        "minObjectsRequired":                 [["bucket"]],
        "minBoundingBoxAspectRatio":          1,
        "intersectingRoiMaxIterations":       5,
        "intersectingRoiStepSize":            0.001,
        "effectiveWidthYcoordMultiplier":     0.5,
        "maxDiffBetweenAbsBucketEdgeSlopes":  3
     }
     '''
    
    
    '''
    #configs 0 restriction
    shovelConfig = {
        "measuredBucketWidthCM":              300,
        "boxDetectorScoreThresholdBucket":    0.2,
        "boxDetectorScoreThresholdMatInside": 0.2,
        "roiDelineatorScoreThreshold":        0.2,
        "minContourArea":                     10,
        "closingKernelSize":                  1,
        "closingIterations":                  0,
        "erosionKernelSize":                  1,
        "erosionIterations":                  0,
        "roiBoundaryPointsReductionFactor":   0.01,
        "minBoundingBoxAspectRatio":          0.2,
        "maxBoundingBoxAspectRatio":          9,
        "minObjectsRequired":                 [["matInside"]],
    }


	############################# FMDL3.2 Configs ####################################
    #Default Backhoe FMDL3.2
    shovelConfig = {
        "measuredBucketWidthCM":              300,
        "boxDetectorScoreThresholdBucket":    0.5,
        "boxDetectorScoreThresholdMatInside": 0.5,
        "boxDetectorScoreThresholdCase":      0.5,
        "roiDelineatorScoreThreshold":        0.5,
        "minContourArea":                     13000,
        "closingKernelSize":                  7,
        "closingIterations":                  1,
        "erosionKernelSize":                  7,
        "erosionIterations":                  1,
        "roiBoundaryPointsReductionFactor":   0.01,
        "minBoundingBoxAspectRatio":          1.5,
        "maxBoundingBoxAspectRatio":          3,
        "minObjectsRequired":                 [["bucket"]],
        "minBoundingBoxAspectRatio":          1,
        "maxBoundingBoxAspectRatio":          5,
        "intersectingRoiMaxIterations":       5,
        "intersectingRoiStepSize":            0.001,
        "effectiveWidthYcoordMultiplier":     0.5,
     }
	##################################################################################


    ############################# FMDL3.1 Configs ####################################
    #Default Hydraulic FMDL3.1
    shovelConfig = {
        "measuredBucketWidthCM":              300,
        "boxDetectorScoreThresholdBucket":    0.5,
        "boxDetectorScoreThresholdMatInside": 0.5,
        "roiDelineatorScoreThreshold":        0.2,
        "minContourArea":                     17000,
        "closingKernelSize":                  7,
        "closingIterations":                  2,
        "erosionKernelSize":                  7,
        "erosionIterations":                  2,
        "roiBoundaryPointsReductionFactor":   0.01,
        "minBoundingBoxAspectRatio":          1,
        "maxBoundingBoxAspectRatio":          4,
        "minObjectsRequired":                 [["matInside"]],
        "intersectingRoiMaxIterations":          50,
        "intersectingRoiStepSize":            0.001
    }

    #Default Backhoe FMDL3.1
    shovelConfig = {
        "measuredBucketWidthCM":              300,
        "boxDetectorScoreThresholdBucket":    0.5,
        "boxDetectorScoreThresholdMatInside": 0.5,
        "roiDelineatorScoreThreshold":        0.5,
        "minContourArea":                     13000,
        "closingKernelSize":                  7,
        "closingIterations":                  1,
        "erosionKernelSize":                  7,
        "erosionIterations":                  1,
        "roiBoundaryPointsReductionFactor":   0.01,
        "minBoundingBoxAspectRatio":          1,
        "maxBoundingBoxAspectRatio":          4,
        "minObjectsRequired":                 [["matInside"]],
        "intersectingRoiMaxIterations":          50,
        "intersectingRoiStepSize":            0.001
    }

    #Default Cable FMDL3.1
    shovelConfig = {
        "measuredBucketWidthCM":              300,
        "boxDetectorScoreThresholdBucket":    0.5,
        "boxDetectorScoreThresholdMatInside": 0.5,
        "roiDelineatorScoreThreshold":        0.2,
        "minContourArea":                     12000,
        "closingKernelSize":                  7,
        "closingIterations":                  4,
        "erosionKernelSize":                  7,
        "erosionIterations":                  4,
        "roiBoundaryPointsReductionFactor":   0.01,
        "minBoundingBoxAspectRatio":          1.2,
        "maxBoundingBoxAspectRatio":          3,
        "minObjectsRequired":                 [["bucket"]],
        "intersectingRoiMaxIterations":          50,
        "intersectingRoiStepSize":            0.001
    }

    ##################################################################################


    ############################# FMDL3.0 Configs ####################################
    #Default Hydraulic FMDL3.0
    shovelConfig = {
        "measuredBucketWidthCM":              300,
        "boxDetectorScoreThresholdBucket":    0.8,
        "boxDetectorScoreThresholdMatInside": 0.5,
        "roiDelineatorScoreThreshold":        0.5,
        "minContourArea":                     13000,
        "closingKernelSize":                  7,
        "closingIterations":                  1,
        "erosionKernelSize":                  7,
        "erosionIterations":                  1,
        "roiBoundaryPointsReductionFactor":   0.01,
        "minBoundingBoxAspectRatio":          1.5,
        "maxBoundingBoxAspectRatio":          3,
        "minObjectsRequired":                 [["bucket"]]
    }
    
    
    #Default Bucyrus FMDL3.0
    shovelConfig = {
        "measuredBucketWidthCM":              300,
        "boxDetectorScoreThresholdBucket":    0.5,
        "boxDetectorScoreThresholdMatInside": 0.5,
        "roiDelineatorScoreThreshold":        0.2,
        "minContourArea":                     12000,
        "closingKernelSize":                  7,
        "closingIterations":                  4,
        "erosionKernelSize":                  7,
        "erosionIterations":                  4,
        "roiBoundaryPointsReductionFactor":   0.01,
        "minBoundingBoxAspectRatio":          1.2,
        "maxBoundingBoxAspectRatio":          3,
        "minObjectsRequired":                 [["bucket"]],
    }


    #Default Backhoe FMDL3.0
    shovelConfig = {
        "measuredBucketWidthCM":              300,
        "boxDetectorScoreThresholdBucket":    0.6,
        "boxDetectorScoreThresholdMatInside": 0.5,
        "roiDelineatorScoreThreshold":        0.5,
        "minContourArea":                     13000,
        "closingKernelSize":                  7,
        "closingIterations":                  1,
        "erosionKernelSize":                  7,
        "erosionIterations":                  1,
        "roiBoundaryPointsReductionFactor":   0.01,
        "minBoundingBoxAspectRatio":          1,
        "maxBoundingBoxAspectRatio":          4,
        "minObjectsRequired":                 [["bucket"]]
    }
    '''
    ##############################################################################



    import PIL
    import base64
    from matplotlib import pyplot as plt
    import io
    import os

    fmdlAlgo = FMDLAlgo(
            boxDetectorNetworkPath,
            roiDelineatorNetworkPath,
            debugMode=True)
    
    
    if runOnSingleImage:
        run_FMDLAlgo()
        
    else:
        TEST_IMAGE_PATHS = []
        TEST_IMAGE_IDS = []
        for fileName in os.listdir(testdirPath):
            TEST_IMAGE_IDS.append(fileName)
            TEST_IMAGE_PATHS.append(testdirPath + fileName)
            
        results = {}
        failedCounter = 0
        successCounter = 0
        
    
        for testFilePath, testFileName in zip(TEST_IMAGE_PATHS, TEST_IMAGE_IDS):
            print("processing: " + testFilePath + "\n")
            
            run_FMDLAlgo()
            
            if results:
                if not results['valid']:
                    print("FAILED")
                    failedCounter += 1
                else:
                    print("SUCESS")
                    successCounter +=1 
            
            
        print("Processed  " + str(len(TEST_IMAGE_PATHS))+ "  files.  " +
              str(failedCounter) + "   files failed.  " + 
              str(successCounter) + "  files were successful")