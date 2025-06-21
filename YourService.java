package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.util.Log;

import gov.nasa.arc.astrobee.Kinematics;
import gov.nasa.arc.astrobee.Result;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import org.opencv.android.Utils;
import org.opencv.aruco.Aruco;
import org.opencv.aruco.DetectorParameters;
import org.opencv.aruco.Dictionary;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point3;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import android.graphics.Bitmap;

import org.opencv.photo.Photo;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.MappedByteBuffer;
/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */

public class YourService extends KiboRpcService {
    private final String TAG = this.getClass().getSimpleName();
    int callpreimg = 1;
    List<double[]> pointTarget = new ArrayList<>();
    List<float[]> quaternionTarget = new ArrayList<>();
    List<String> EachTarget = new ArrayList<>();
    @Override
    protected void runPlan1() {
        // Initialize TensorFlow Lite model first
        if (tfliteInterpreter == null) {
            initializeTFLiteModel();
        }
        Mat image = null;
        Full_process image_processor = new Full_process();
        List<String> result = null;
        final int LOOP_MAX = 5;
        Result res;

        api.startMission();
        Log.i(TAG, "Start mission");

        Point point = new Point(10.9d, -9.92284d, 5.195d);
        Quaternion quaternion = new Quaternion(0f, 0f, -0.707f, 0.707f);
        int loopCounter = 0;
        res = api.moveTo(point, quaternion, false);
        while (!res.hasSucceeded() && loopCounter < LOOP_MAX) {
            res = api.moveTo(point, quaternion, false);
            loopCounter++;
        }
        image = api.getMatNavCam();
        result = image_processor.process(image, callpreimg);
        callpreimg++;
        Point pointArea2_1 = new Point(10.95d, -9.54d, 4.745d);
        Quaternion quaternionArea2_1 = new Quaternion(0f, 0.707f, 0f, 0.707f);
        api.moveTo(pointArea2_1, quaternionArea2_1, false);
        // Area 2
        Point pointArea2 = new Point(10.95d, -9.2, 4.745d);  // Midpoint x = (10.3 + 11.55) / 2, y = (−9.25 + −8.5) / 2
        Quaternion quaternionArea2 = new Quaternion(0f, 0.707f, 0f, 0.707f);
        loopCounter = 0;
        res = api.moveTo(pointArea2, quaternionArea2, false);
        while (!res.hasSucceeded() && loopCounter < LOOP_MAX) {
            res = api.moveTo(pointArea2, quaternionArea2, false);
            loopCounter++;
        }
        image = api.getMatNavCam();
        result = image_processor.process(image, callpreimg);
        callpreimg++;

        // Area 3
        Point pointArea3 = new Point(10.925, -7.445d, 4.52d);  // Midpoint y = (−8.4 + −7.45) / 2
        Quaternion quaternionArea3 = new Quaternion(0f, 0.707f, 0f, 0.707f);// Assuming same orientation
        api.moveTo(pointArea3, quaternionArea3, false);
        loopCounter = 0;
        res = api.moveTo(pointArea3, quaternionArea3, false);
        while (!res.hasSucceeded() && loopCounter < LOOP_MAX) {
            res = api.moveTo(pointArea3, quaternionArea3, false);
            loopCounter++;
        }
        image = api.getMatNavCam();
        result = image_processor.process(image, callpreimg);
        callpreimg++;
//
        Point pointArea4_1 = new Point(11.137, -7.21d, 4.815d); // y = (−7.34 + −6.365)/2, z = (4.32 + 5.57)/2
        Quaternion quaternionArea4_1 = new Quaternion(-0.707f, 0f, 0f, 0.707f);
        api.moveTo(pointArea4_1, quaternionArea4_1, false);
//        // Area 4
        Point pointArea4 = new Point(11.137, -7.01d, 4.815d); // y = (−7.34 + −6.365)/2, z = (4.32 + 5.57)/2
        Quaternion quaternionArea4 = new Quaternion(-0.707f, 0f, 0f, 0.707f);
        loopCounter = 0;
        res = api.moveTo(pointArea4, quaternionArea4, false);
        while (!res.hasSucceeded() && loopCounter < LOOP_MAX) {
            res = api.moveTo(pointArea4, quaternionArea4, false);
            loopCounter++;
        }
        image = api.getMatDockCam();
        result = image_processor.process(image, callpreimg);
        callpreimg++;

        point = new Point(11.143d, -6.7607d, 4.9654d);
        quaternion = new Quaternion(0f, 0f, 0.707f, 0.707f);
        api.moveTo(point, quaternion, false);
        api.reportRoundingCompletion();
        try {
            Thread.sleep(1700); // Pauses the current thread
        } catch (InterruptedException e) {
            // Handle the exception if the thread is interrupted while sleeping
            Thread.currentThread().interrupt(); // Re-interrupt the current thread
        }
        image = api.getMatNavCam();
        result = image_processor.process(image, callpreimg);
    }

    @Override
    protected void runPlan2(){
        // write your plan 2 here
    }

    @Override
    protected void runPlan3(){
        // write your plan 3 here.
    }

    public class Full_process {
        public List<String> process(Mat inputImage, int callpreimg) {
            double[][] cameraParam = null;
            if (callpreimg == 4) {
                cameraParam = api.getDockCamIntrinsics();
            } else {
                cameraParam = api.getNavCamIntrinsics();
            }
            Mat cameraM = new Mat(3, 3, CvType.CV_64F);
            cameraM.put(0, 0, cameraParam[0]);
            Mat cameraCoeff = new Mat(1, 5, CvType.CV_64F);
            cameraCoeff.put(0, 0, cameraParam[1]);
            cameraCoeff.convertTo(cameraCoeff, CvType.CV_64F);

            Mat undistort = new Mat();
            Calib3d.undistort(inputImage, undistort, cameraM, cameraCoeff);
            api.saveMatImage(undistort, callpreimg + "undistort.png");
            Log.i(TAG, "Undistort image size: " + undistort.size() + ", channels: " + undistort.channels());

            Log.i(TAG, "Start cropping image");
            DocumentScanner scanner = new DocumentScanner();
            Mat A4Crop = scanner.processDocument(undistort);
            api.saveMatImage(A4Crop, callpreimg + "A4Crop.png");
            if (!scanner.detectAruco(A4Crop)) {
                A4Crop = ManualA4Cropper(undistort, cameraM);
                api.saveMatImage(A4Crop, callpreimg + "ManualA4Crop.png");
                Log.i(TAG, "[WARNING] No ArUco markers detected by DocumentScanner. Using ManualA4Cropper instead.");
            } else {
                A4Crop = scanner.cropRightCmRatio(A4Crop);
            }

            List<Mat> croppedImages = cropObjects(A4Crop);
            int validObjectCount = 0;
            int landmarkCount = 0;
            List<String> allPredictedClasses = new ArrayList<>();

            for (int i = 0; i < croppedImages.size(); i++) {
                Mat croppedImage = croppedImages.get(i);
                if (croppedImage != null && !croppedImage.empty()) {
                    validObjectCount++;

                    // Classify the object
                    String predictedClass = classifyObject(croppedImage);
                    allPredictedClasses.add(predictedClass);
                    Log.i(TAG, "Object " + i + " classified as: " + predictedClass);

                    api.saveMatImage(croppedImage, callpreimg + "object_" + i + ".png");
                    if (predictedClass != "crystal" && predictedClass != "diamond" && predictedClass != "emerald") {
                        landmarkCount++;
                        api.setAreaInfo(callpreimg, predictedClass, landmarkCount);
                    } else {
                        if (callpreimg < 5) {
                            double MARKER_SIZE = 0.045;
                            List<double[]> positions = new ArrayList<>();
                            Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
                            MatOfInt ids = new MatOfInt();
                            List<Mat> corners = new ArrayList<>();
                            DetectorParameters parameters = DetectorParameters.create();
                            Aruco.detectMarkers(undistort, dictionary, corners, ids, parameters);
                            // Pose estimation
                            Mat rvecs = new Mat();
                            Mat tvecs = new Mat();

                            try {
                                Aruco.estimatePoseSingleMarkers(corners, (float) MARKER_SIZE, cameraM, cameraCoeff, rvecs, tvecs);
                                Log.i(TAG, "Pose estimation completed");
                                Log.i(TAG, "rvecs size: " + rvecs.rows() + "x" + rvecs.cols());
                                Log.i(TAG, "tvecs size: " + tvecs.rows() + "x" + tvecs.cols());
                            } catch (Exception e) {
                                Log.e(TAG, "Error in pose estimation: " + e.getMessage());
                            }

                            double[] robotPos = new double[]{0.0, 0.0, 0.0}; // Default initialization
                            try {
                                Kinematics k = api.getRobotKinematics();
                                Point robotPoint = k.getPosition();
                                robotPos = new double[]{robotPoint.getX(), robotPoint.getY(), robotPoint.getZ()};
                                Log.i(TAG, "Robot position: [" + robotPos[0] + ", " + robotPos[1] + ", " + robotPos[2] + "]");
                            } catch (Exception e) {
                                Log.e(TAG, "Error getting robot position: " + e.getMessage());
                                // robotPos will use default values [0.0, 0.0, 0.0]
                            }


                            double[] tvec = tvecs.get(0, 0);
                            Log.i(TAG, "tvec position: [" + tvec[0] + ", " + tvec[1] + ", " + tvec[2] + "]");
                            double[] targetInCamera = new double[3];
                            if (callpreimg == 1){
                                targetInCamera[0] = tvec[0] + robotPos[0];
                                targetInCamera[1] = -9.88d;
                                targetInCamera[2] = robotPos[2] + tvec[1];
                                quaternionTarget.add(new float[]{0.0f, 0.0f, -0.707f, 0.707f});
                            } else if (callpreimg == 2 || callpreimg == 3) {
                                targetInCamera[0] = robotPos[0] + tvec[1];
                                targetInCamera[1] = robotPos[1] + tvec[0];
                                targetInCamera[2] = 4.46203d;
                                quaternionTarget.add(new float[]{0.0f, 0.707f, 0.0f, 0.707f});
                            } else if (callpreimg == 4) {
                                targetInCamera[0] = 10.566984d;
                                targetInCamera[1] = robotPos[1] + tvec[1];
                                targetInCamera[2] = robotPos[2] + tvec[0];
                                quaternionTarget.add(new float[]{-0.707f, 0.0f, 0.0f, 0.707f});
                            }
                            pointTarget.add(new double[]{targetInCamera[0], targetInCamera[1], targetInCamera[2]});
                            EachTarget.add(predictedClass);
                            Log.i(TAG, "Target point in robot coordinates (constrained to KIZ 1): [" + targetInCamera[0] + ", " + targetInCamera[1] + ", " + targetInCamera[2] + "]");
                        } else {
                            int indexTarget = -1;
                            for (int j = 0; j < EachTarget.size(); j++) {
                                if (predictedClass == EachTarget.get(j)) {
                                    indexTarget = j;
                                    api.notifyRecognitionItem();
                                    break;
                                }
                            }
                            Point point = new Point(pointTarget.get(indexTarget)[0], pointTarget.get(indexTarget)[1], pointTarget.get(indexTarget)[2]);
                            Quaternion quaternion = new Quaternion(quaternionTarget.get(indexTarget)[0], quaternionTarget.get(indexTarget)[1], quaternionTarget.get(indexTarget)[2], quaternionTarget.get(indexTarget)[3]);
                            Result result = api.moveTo(point, quaternion, false);
                            int loopCounter = 0;
                            while (!result.hasSucceeded() && loopCounter < 3) {
                                result = api.moveTo(point, quaternion, false);
                                loopCounter++;
                            }
                            api.takeTargetItemSnapshot();
                            break;
                        }
                    }
                }
            }

            Log.i(TAG, "Total objects detected: " + validObjectCount);
            Log.i(TAG, "All predicted classes: " + allPredictedClasses.toString());
            Log.i(TAG, "Classify image done!!");
            return allPredictedClasses;
        }
    }

    private Interpreter tfliteInterpreter;
    private String[] classNames = {"coin", "compass", "coral", "crystal", "diamond",
            "emerald", "fossil", "key", "letter", "shell", "treasure_box"};
    private void initializeTFLiteModel() {
        try {
            // Load the TFLite model from assets
            MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(getApplicationContext(),
                    "final_icon_classifier_model.tflite");

            // Create and configure interpreter options
            Interpreter.Options options = new Interpreter.Options();
//            options.setNumThreads(4); // Optional: set number of threads

            // Initialize the interpreter
            tfliteInterpreter = new Interpreter(tfliteModel, options);

            Log.i(TAG, "TensorFlow Lite model loaded successfully");
        } catch (IOException e) {
            Log.e(TAG, "Error loading TFLite model: " + e.getMessage());
        }
    }

    int image_size = 96;
    private String classifyObject(Mat croppedImage) {
        if (tfliteInterpreter == null) {
            Log.e(TAG, "TensorFlow Lite interpreter not initialized");
            return "unknown";
        }

        try {
            // Resize the image to image_sizeximage_size (model input size)
            Mat resized = new Mat();
            Imgproc.resize(croppedImage, resized, new Size(image_size, image_size));

            // Convert to RGB if needed
            Mat rgb = new Mat();
            if (resized.channels() == 3) {
                Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_BGR2RGB);
            } else if (resized.channels() == 1) {
                Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_GRAY2RGB);
            } else {
                rgb = resized.clone();
            }

            // Convert Mat to Bitmap
            Bitmap bitmap = Bitmap.createBitmap(rgb.cols(), rgb.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(rgb, bitmap);

            // Convert Bitmap to ByteBuffer for TensorFlow Lite
            ByteBuffer inputBuffer = convertBitmapToByteBuffer(bitmap);

            // Prepare output array
            float[][] output = new float[1][11]; // 1 batch, 11 classes

            // Run inference
            tfliteInterpreter.run(inputBuffer, output);

            // Process output
            float[] predictions = output[0];
            int predictedClass = argmax(predictions);
            float confidence = predictions[predictedClass];

            String predictedClassName = classNames[predictedClass];

            Log.i(TAG, "Predicted class: " + predictedClass + " (" + predictedClassName +
                    ") with confidence: " + String.format("%.4f", confidence));

            // Clean up
            resized.release();
            rgb.release();
            bitmap.recycle();

            return predictedClassName;

        } catch (Exception e) {
            Log.e(TAG, "Error during classification: " + e.getMessage());
            return "unknown";
        }
    }

    // Helper method to convert Bitmap to ByteBuffer
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * image_size * image_size * 3);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[image_size * image_size];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < image_size; ++i) {
            for (int j = 0; j < image_size; ++j) {
                final int val = intValues[pixel++];

                // FIXED: Remove the /255.0f normalization
                byteBuffer.putFloat(((val >> 16) & 0xFF)); // Red: 0-255 range
                byteBuffer.putFloat(((val >> 8) & 0xFF));  // Green: 0-255 range
                byteBuffer.putFloat((val & 0xFF));         // Blue: 0-255 range
            }
        }
        return byteBuffer;
    }

    // Helper method to find argmax
    private int argmax(float[] array) {
        int maxIndex = 0;
        float maxValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public class DocumentScanner {
        private static final String TAG = "DocumentScanner";

        public Mat processDocument(Mat inputImage) {
            if (inputImage.empty()) {
                Log.i(TAG, "[ERROR] Input image is empty");
                return new Mat();
            }

            Log.i(TAG, "[DEBUG] Input image size: " + inputImage.size() + ", type: " + inputImage.type() + ", channels: " + inputImage.channels());

            Mat gray = new Mat();
            if (inputImage.channels() > 1) {
                Imgproc.cvtColor(inputImage, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = inputImage.clone();
            }

            Mat imgCopy = new Mat();
            if (inputImage.channels() == 1) {
                Imgproc.cvtColor(inputImage, imgCopy, Imgproc.COLOR_GRAY2BGR);
            } else {
                imgCopy = inputImage.clone();
            }

            Log.i(TAG, "[DEBUG] gray type: " + gray.type() + ", channels: " + gray.channels());
            Log.i(TAG, "[DEBUG] imgCopy type: " + imgCopy.type() + ", channels: " + imgCopy.channels());

            boolean arucoDetected = detectAruco(imgCopy);

            Mat edged = preprocessEdges(gray);
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(edged, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

            if (contours.isEmpty()) {
                Log.i(TAG, "[ERROR] No contours found");
                return inputImage;
            }

            MatOfPoint2f biggest = findContourWithAruco(contours, imgCopy);
            saveContourDetectionImage(imgCopy, contours, biggest, callpreimg);
            if (biggest == null) {
                Log.i(TAG, "[ERROR] No valid rectangle found");
                return inputImage;
            }

            org.opencv.core.Point[] points = biggest.toArray();
            Log.i(TAG, "[DEBUG] Raw contour points found: " + points.length);

            if (!validateCorners(points)) {
                Log.i(TAG, "[ERROR] Corner validation failed");
                return inputImage;
            }

            Log.i(TAG, "[SUCCESS] Corners validated");

            try {
                Mat warped = perspectiveTransform(imgCopy, points);
                Mat rotated = correctImageOrientation(warped);
                // Mat cropped = cropRightCmRatio(rotated);

                Log.i(TAG, "[SUCCESS] Document processing completed");
                return rotated;
            } catch (Exception e) {
                Log.i(TAG, "[ERROR] Processing failed: " + e.getMessage());
                return inputImage;
            } finally {
                // Release temporary matrices
                gray.release();
                imgCopy.release();
            }
        }

        private boolean detectAruco(Mat imgDraw) {
            Mat gray1 = new Mat();
            if (imgDraw.channels() > 1) {
                Imgproc.cvtColor(imgDraw, gray1, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray1 = imgDraw.clone();
            }
            Mat gray = gray1.clone();
            Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
            DetectorParameters parameters = DetectorParameters.create();

            List<Mat> corners = new ArrayList<>();
            Mat ids = new Mat();
            List<Mat> rejected = new ArrayList<>();

            // Ensure gray is CV_8U
            Mat gray8U = new Mat();
            if (gray.type() != CvType.CV_8U) {
                Log.i(TAG, "[DEBUG] Converting gray matrix to CV_8U, original type: " + gray.type());
                gray.convertTo(gray8U, CvType.CV_8U);
            } else {
                gray8U = gray.clone();
            }

            Log.i(TAG, "[DEBUG] gray8U type: " + gray8U.type() + ", channels: " + gray8U.channels());

            Aruco.detectMarkers(gray8U, dictionary, corners, ids, parameters, rejected);

            if (!ids.empty() && corners.size() > 0) {
                Log.i(TAG, "[SUCCESS] ArUco marker detected: " + corners.size());
                // Ensure imgDraw is CV_8UC3 for drawing
                Mat drawMat = new Mat();
                if (imgDraw.type() != CvType.CV_8UC3) {
                    Log.i(TAG, "[DEBUG] Converting imgDraw to CV_8UC3, original type: " + imgDraw.type());
                    Imgproc.cvtColor(imgDraw, drawMat, Imgproc.COLOR_GRAY2BGR);
                } else {
                    drawMat = imgDraw.clone();
                }

                Aruco.drawDetectedMarkers(drawMat, corners, ids);
                drawMat.release();

                // Process corner points
                Mat corner = corners.get(0);
                Log.i(TAG, "[DEBUG] Corner matrix type: " + corner.type() + ", size: " + corner.size() + ", channels: " + corner.channels());

                // Convert corner data to float array for processing
                Mat corner32F = new Mat();
                if (corner.type() == CvType.CV_16SC2) {
                    Log.i(TAG, "[DEBUG] Handling CV_16SC2 corner matrix");
                    // For CV_16SC2, we need to handle it differently
                    corner.convertTo(corner32F, CvType.CV_32FC2);
                } else if (corner.type() != CvType.CV_32FC2) {
                    Log.i(TAG, "[DEBUG] Converting corner matrix to CV_32FC2, original type: " + corner.type());
                    corner.convertTo(corner32F, CvType.CV_32FC2);
                } else {
                    corner32F = corner.clone();
                }

                Log.i(TAG, "[DEBUG] corner32F type: " + corner32F.type() + ", size: " + corner32F.size() + ", channels: " + corner32F.channels());

                // Use float array instead of double array for CV_32FC2
                float[] floatData = new float[8];
                try {
                    corner32F.get(0, 0, floatData);
                    Log.i(TAG, "[DEBUG] corner32F data: " + Arrays.toString(floatData));

                    // Convert to Point array
                    org.opencv.core.Point[] pts = new org.opencv.core.Point[4];
                    for (int i = 0; i < 4; i++) {
                        pts[i] = new org.opencv.core.Point(floatData[i * 2], floatData[i * 2 + 1]);
                    }

                    // Calculate rotation angle
                    double dx = pts[1].x - pts[0].x;
                    double dy = pts[1].y - pts[0].y;
                    double angle = Math.toDegrees(Math.atan2(dy, dx));
                    Log.i(TAG, "[INFO] ArUco rotation angle: " + String.format("%.2f", angle) + "°");

                } catch (Exception e) {
                    Log.i(TAG, "[ERROR] Failed to extract corner data: " + e.getMessage());
                    // Try alternative approach with double array
                    try {
                        double[] doubleData = new double[8];
                        corner32F.get(0, 0, doubleData);
                        Log.i(TAG, "[DEBUG] Successfully extracted with double array: " + Arrays.toString(doubleData));
                    } catch (Exception e2) {
                        Log.i(TAG, "[ERROR] Both float and double extraction failed: " + e2.getMessage());
                        gray8U.release();
                        corner32F.release();
                        return false;
                    }
                }

                // Release temporary matrices
                gray8U.release();
                corner32F.release();
                return true;
            }

            Log.i(TAG, "[WARNING] No ArUco markers detected");
            gray8U.release();
            return false;
        }

        private Mat preprocessEdges(Mat gray) {
            Mat blur = new Mat();
            Imgproc.bilateralFilter(gray, blur, 20, 30, 30);

            Mat edged = new Mat();
            Imgproc.Canny(blur, edged, 10, 20);

            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
            Imgproc.dilate(edged, edged, kernel);

            return edged;
        }

        private MatOfPoint2f findContourWithAruco(List<MatOfPoint> contours, Mat image) {
            Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250); // Match your ArUco markers
            MatOfPoint2f biggestWithAruco = null;

            for (MatOfPoint contour : contours) {
                double area = Imgproc.contourArea(contour);
                if (area > 1000 && area < 45000) { // Minimum area threshold
                    MatOfPoint2f contour2f = new MatOfPoint2f();
                    contour.convertTo(contour2f, CvType.CV_32FC2);

                    // Create a mask for the current contour
                    Mat mask = Mat.zeros(image.size(), CvType.CV_8UC1);
                    Imgproc.drawContours(mask, Collections.singletonList(contour), -1, new Scalar(255), -1);

                    // Extract region of interest (ROI) based on the contour
                    Mat roi = new Mat(image.size(), CvType.CV_8UC1);
                    image.copyTo(roi, mask);

                    // Detect ArUco markers in the ROI
                    MatOfInt ids = new MatOfInt();
                    List<Mat> corners = new ArrayList<>();
                    Aruco.detectMarkers(roi, dictionary, corners, ids);

                    // If ArUco marker is found, select this contour
                    if (!ids.empty()) {
                        MatOfPoint2f approx = new MatOfPoint2f();
                        double epsilon = 0.015 * Imgproc.arcLength(contour2f, true);
                        Imgproc.approxPolyDP(contour2f, approx, epsilon, true);

                        org.opencv.core.Point[] approxPoints = approx.toArray();
                        if (approxPoints.length == 4) {
                            biggestWithAruco = approx;
                            break; // Stop at the first contour with an ArUco marker
                        }
                    }
                }
            }
            return biggestWithAruco != null ? biggestWithAruco : findBiggestContour(contours); // Fallback
        }

        private MatOfPoint2f findBiggestContour(List<MatOfPoint> contours) {
            double maxArea = 0;
            MatOfPoint2f biggest = null;

            for (MatOfPoint contour : contours) {
                double area = Imgproc.contourArea(contour);
                if (area > 1000 && area < 45000) {
                    MatOfPoint2f contour2f = new MatOfPoint2f();
                    contour.convertTo(contour2f, CvType.CV_32FC2);

                    MatOfPoint2f approx = new MatOfPoint2f();
                    double epsilon = 0.015 * Imgproc.arcLength(contour2f, true);
                    Imgproc.approxPolyDP(contour2f, approx, epsilon, true);

                    org.opencv.core.Point[] approxPoints = approx.toArray();
                    if (approxPoints.length == 4 && area > maxArea) {
                        biggest = approx;
                        maxArea = area;
                    }
                }
            }

            Log.i(TAG, "[DEBUG] Biggest contour area: " + maxArea);
            return biggest;
        }

        private void saveContourDetectionImage(Mat inputImage, List<MatOfPoint> contours, MatOfPoint2f biggestWithAruco, int callpreimg) {
            // Create a copy of the input image for drawing
            Mat contourImage = new Mat();
            if (inputImage.channels() == 1) {
                Imgproc.cvtColor(inputImage, contourImage, Imgproc.COLOR_GRAY2BGR);
            } else {
                contourImage = inputImage.clone();
            }

            // Draw all contours in light blue
            for (int i = 0; i < contours.size(); i++) {
                double area = Imgproc.contourArea(contours.get(i));
                if (area > 1000) { // Only draw significant contours
                    Imgproc.drawContours(contourImage, contours, i,
                            new org.opencv.core.Scalar(255, 255, 0), 2); // Light blue
                }
            }

            // Draw the biggest contour with ArUco (if exists) in red with thicker line
            if (biggestWithAruco != null) {
                List<MatOfPoint> biggestContour = new ArrayList<>();
                MatOfPoint biggestAsMatOfPoint = new MatOfPoint();
                biggestWithAruco.convertTo(biggestAsMatOfPoint, CvType.CV_32S);
                biggestContour.add(biggestAsMatOfPoint);

                Imgproc.drawContours(contourImage, biggestContour, -1,
                        new org.opencv.core.Scalar(0, 0, 255), 4); // Red, thick line

                // Draw corner points as circles
                org.opencv.core.Point[] points = biggestWithAruco.toArray();
                for (int i = 0; i < points.length; i++) {
                    org.opencv.core.Point point = points[i];
                    Imgproc.circle(contourImage, point, 8,
                            new org.opencv.core.Scalar(0, 255, 0), -1); // Green filled circles

                    // Add corner labels
                    Imgproc.putText(contourImage, String.valueOf(i),
                            new org.opencv.core.Point(point.x + 10, point.y - 10),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 0.7,
                            new org.opencv.core.Scalar(255, 255, 255), 2);
                }
            }

            // Add text overlay with detection info
            String info = String.format("Contours: %d, Area: %.0f",
                    contours.size(), biggestWithAruco != null ? Imgproc.contourArea(new MatOfPoint(biggestWithAruco.toArray())) : 0);
            Imgproc.putText(contourImage, info,
                    new org.opencv.core.Point(10, 30),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.8,
                    new org.opencv.core.Scalar(255, 255, 255), 2);

            // Save the contour detection image
            api.saveMatImage(contourImage, callpreimg + "Contour_detection.png");
            Log.i(TAG, "Saved contour detection visualization");

            // Clean up
            contourImage.release();
        }

        private org.opencv.core.Point[] orderPoints(org.opencv.core.Point[] pts) {
            // Sort by y-coordinate first
            java.util.Arrays.sort(pts, new java.util.Comparator<org.opencv.core.Point>() {
                @Override
                public int compare(org.opencv.core.Point a, org.opencv.core.Point b) {
                    return Double.compare(a.y, b.y);
                }
            });

            org.opencv.core.Point[] topPts = {pts[0], pts[1]};
            org.opencv.core.Point[] bottomPts = {pts[2], pts[3]};

            // Sort top and bottom by x-coordinate
            java.util.Arrays.sort(topPts, new java.util.Comparator<org.opencv.core.Point>() {
                @Override
                public int compare(org.opencv.core.Point a, org.opencv.core.Point b) {
                    return Double.compare(a.x, b.x);
                }
            });
            java.util.Arrays.sort(bottomPts, new java.util.Comparator<org.opencv.core.Point>() {
                @Override
                public int compare(org.opencv.core.Point a, org.opencv.core.Point b) {
                    return Double.compare(a.x, b.x);
                }
            });

            org.opencv.core.Point[] ordered = {
                    topPts[0],      // Top-left
                    topPts[1],      // Top-right
                    bottomPts[1],   // Bottom-right
                    bottomPts[0]    // Bottom-left
            };

            Log.i(TAG, "[DEBUG] Points ordered successfully");

            // Check minimum distance between points
            double minDist = 10.0;
            for (int i = 0; i < 4; i++) {
                for (int j = i + 1; j < 4; j++) {
                    double dist = Math.sqrt(Math.pow(ordered[i].x - ordered[j].x, 2) +
                            Math.pow(ordered[i].y - ordered[j].y, 2));
                    if (dist < minDist) {
                        Log.i(TAG, "[ERROR] Points " + i + " and " + j + " are too close: distance=" + dist);
                        return null;
                    }
                }
            }

            return ordered;
        }

        private Size calculateDimensions(org.opencv.core.Point[] pts) {
            double w1 = Math.sqrt(Math.pow(pts[0].x - pts[1].x, 2) + Math.pow(pts[0].y - pts[1].y, 2));
            double w2 = Math.sqrt(Math.pow(pts[2].x - pts[3].x, 2) + Math.pow(pts[2].y - pts[3].y, 2));
            double h1 = Math.sqrt(Math.pow(pts[0].x - pts[3].x, 2) + Math.pow(pts[0].y - pts[3].y, 2));
            double h2 = Math.sqrt(Math.pow(pts[1].x - pts[2].x, 2) + Math.pow(pts[1].y - pts[2].y, 2));

            int w = (int) Math.max(Math.max(w1, w2), 200);
            int h = (int) Math.max(Math.max(h1, h2), 200);

            Log.i(TAG, "[DEBUG] Calculated dimensions: width=" + w + " height=" + h);
            return new Size(w, h);
        }

        private boolean validateCorners(org.opencv.core.Point[] points) {
            if (points.length != 4) {
                Log.i(TAG, "[ERROR] Not exactly 4 corners: " + points.length);
                return false;
            }

            // Check for unique points
            for (int i = 0; i < points.length; i++) {
                for (int j = i + 1; j < points.length; j++) {
                    double dist = Math.sqrt(Math.pow(points[i].x - points[j].x, 2) +
                            Math.pow(points[i].y - points[j].y, 2));
                    if (dist < 5.0) {
                        Log.i(TAG, "[ERROR] Duplicate points detected");
                        return false;
                    }
                }
            }

            org.opencv.core.Point[] ordered = orderPoints(points.clone());
            if (ordered == null) {
                return false;
            }

            Size dims = calculateDimensions(ordered);
            if (Math.min(dims.width, dims.height) < 100) {
                Log.i(TAG, "[ERROR] Too small: " + dims.width + "x" + dims.height);
                return false;
            }

            if (Math.max(dims.width, dims.height) / Math.min(dims.width, dims.height) > 10) {
                Log.i(TAG, "[ERROR] Extreme aspect ratio");
                return false;
            }

            return true;
        }

        private Mat perspectiveTransform(Mat img, org.opencv.core.Point[] pts) {
            org.opencv.core.Point[] ordered = orderPoints(pts.clone());
            if (ordered == null) {
                Log.i(TAG, "[ERROR] Failed to order points");
                return img;
            }

            Size dims = calculateDimensions(ordered);

            Mat srcPoints = new Mat(4, 1, CvType.CV_32FC2);
            Mat dstPoints = new Mat(4, 1, CvType.CV_32FC2);

            srcPoints.put(0, 0, ordered[0].x, ordered[0].y);
            srcPoints.put(1, 0, ordered[1].x, ordered[1].y);
            srcPoints.put(2, 0, ordered[2].x, ordered[2].y);
            srcPoints.put(3, 0, ordered[3].x, ordered[3].y);

            dstPoints.put(0, 0, 0, 0);
            dstPoints.put(1, 0, dims.width, 0);
            dstPoints.put(2, 0, dims.width, dims.height);
            dstPoints.put(3, 0, 0, dims.height);

            Mat matrix = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);
            Mat warped = new Mat();
            Imgproc.warpPerspective(img, warped, matrix, dims);

            Log.i(TAG, "[DEBUG] Warped image size: " + warped.size());
            return warped;
        }

        private Mat correctImageOrientation(Mat image) {
            Mat gray = new Mat();
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);

            Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
            DetectorParameters parameters = DetectorParameters.create();

            List<Mat> corners = new ArrayList<>();
            Mat ids = new Mat();
            List<Mat> rejected = new ArrayList<>();

            Aruco.detectMarkers(gray, dictionary, corners, ids, parameters, rejected);

            if (!ids.empty() && corners.size() > 0) {
                Log.i(TAG, "Detected ArUco markers for rotation correction");

                Mat corner = corners.get(0);

                // Convert corner data properly
                Mat corner32F = new Mat();
                if (corner.type() != CvType.CV_32FC2) {
                    corner.convertTo(corner32F, CvType.CV_32FC2);
                } else {
                    corner32F = corner.clone();
                }

                // Use float array for CV_32FC2 data
                float[] floatData = new float[8];
                try {
                    corner32F.get(0, 0, floatData);

                    org.opencv.core.Point p1 = new org.opencv.core.Point(floatData[0], floatData[1]); // Top-left
                    org.opencv.core.Point p2 = new org.opencv.core.Point(floatData[2], floatData[3]); // Top-right

                    double angleRad = Math.atan2(p2.y - p1.y, p2.x - p1.x);
                    double angleDeg = Math.toDegrees(angleRad);
                    Log.i(TAG, "Calculated marker angle: " + angleDeg + " degrees");

                    double rotationAngle = Math.round(angleDeg / 90) * 90;
                    Log.i(TAG, "Applying rotation angle: " + rotationAngle + " degrees");

                    org.opencv.core.Point center = new org.opencv.core.Point(image.width() / 2.0, image.height() / 2.0);
                    Mat rotationMatrix = Imgproc.getRotationMatrix2D(center, rotationAngle, 1.0);

                    // Calculate new dimensions
                    double cos = Math.abs(rotationMatrix.get(0, 0)[0]);
                    double sin = Math.abs(rotationMatrix.get(0, 1)[0]);
                    int newW = (int) ((image.height() * sin) + (image.width() * cos));
                    int newH = (int) ((image.height() * cos) + (image.width() * sin));

                    // Adjust for translation
                    rotationMatrix.put(0, 2, rotationMatrix.get(0, 2)[0] + (newW / 2.0) - center.x);
                    rotationMatrix.put(1, 2, rotationMatrix.get(1, 2)[0] + (newH / 2.0) - center.y);

                    Mat rotated = new Mat();
                    Imgproc.warpAffine(image, rotated, rotationMatrix, new Size(newW, newH));

                    // Clean up
                    gray.release();
                    corner32F.release();
                    rotationMatrix.release();

                    Log.i(TAG, "Successfully rotated image to: " + rotated.size());
                    return rotated;

                } catch (Exception e) {
                    Log.i(TAG, "[ERROR] Failed to process corner data for rotation: " + e.getMessage());
                    // Clean up
                    gray.release();
                    corner32F.release();
                    return image; // Return original image if rotation fails
                }
            } else {
                Log.i(TAG, "No ArUco markers detected for rotation correction");
                gray.release();
                return image;
            }
        }

        private Mat cropRightCmRatio(Mat image) {
            int height = image.height();
            int width = image.width();

            // ---- Step 1: Crop right side (A4 width ratio: 15.7 / 21.0) ----
            double widthFraction = 15.7 / 21.0;
            int cropWidth = (int) (width * widthFraction);

            if (cropWidth <= 0) {
                Log.i(TAG, "Image is too narrow to crop");
                return image;
            }

            // ---- Step 2: Convert 6 cm from top and 3 cm from bottom to pixels ----
            // Based on A4 paper height (29.7 cm)
            int cropTop = (int) (height * (4.6 / 29.7));
            int cropBottom = (int) (height * (4.6 / 29.7));

            if (height <= (cropTop + cropBottom)) {
                Log.i(TAG, "Image is too short to crop 6 cm top and 3 cm bottom");
                return image;
            }

            // Final cropping rectangle
            Rect cropRect = new Rect(
                    0,                        // x: from left
                    cropTop,                 // y: start after top 6 cm
                    cropWidth,               // width: cropped to 15.7/21.0 of original
                    height - cropTop - cropBottom  // height: remove 6 cm top & 3 cm bottom
            );

            Mat cropped = new Mat(image, cropRect);
            Log.i(TAG, "Cropped image to size: " + cropped.size());
            api.saveMatImage(cropped, callpreimg + "crop_RTB.png");
            return cropped;
        }
    }

    // new vesion for crop image function
    public List<Mat> cropObjects(Mat image) {
        // Convert to grayscale if necessary
        Mat gray = new Mat();
        if (image.channels() == 3) {
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
        } else if (image.channels() == 4) {
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGRA2GRAY);
        } else {
            gray = image.clone();
        }

        // Apply denoising to reduce noise - UPDATED PARAMETERS
        try {
            Photo.fastNlMeansDenoising(gray, gray, 7.0f, 7, 21); // Changed from 10.0f to 7.0f
        } catch (Exception e) {
            Imgproc.bilateralFilter(gray, gray, 5, 25, 25); // Changed from (7, 50, 50) to (5, 25, 25)
        }

        // Apply Gaussian blur to further smooth noise - UPDATED KERNEL SIZE
        Mat blur = new Mat();
        Imgproc.GaussianBlur(gray, blur, new Size(5, 5), 0); // Changed from (7, 7) to (5, 5)

        // Apply CLAHE for better contrast - UPDATED PARAMETERS
        CLAHE clahe = Imgproc.createCLAHE();
        clahe.setClipLimit(2.0); // Changed from 1.0 to 2.0
        clahe.setTilesGridSize(new Size(4, 4)); // Changed from (8, 8) to (4, 4)
        Mat grayClahe = new Mat();
        clahe.apply(blur, grayClahe);

        // Adaptive thresholding with adjusted parameters - UPDATED PARAMETERS
        Mat thresh = new Mat();
        Imgproc.adaptiveThreshold(
                grayClahe, thresh, 255,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY_INV,
                7, 4 // Changed from (21, 10) to (11, 5)
        );

        // Morphological operations to remove noise and close gaps - UPDATED ORDER
        Mat kernel = Mat.ones(3, 3, CvType.CV_8U);
        Mat morph = new Mat();
        // Changed order: CLOSE first, then OPEN, and updated iterations
        Imgproc.morphologyEx(thresh, morph, Imgproc.MORPH_CLOSE, kernel, new org.opencv.core.Point(-1, -1), 2);
        Imgproc.morphologyEx(morph, thresh, Imgproc.MORPH_OPEN, kernel, new org.opencv.core.Point(-1, -1), 1);

        // Save intermediate threshold image for debugging
        api.saveMatImage(thresh, callpreimg + "thresh_debug.png");

        // Find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        List<Mat> croppedImages = new ArrayList<>();

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > 30) {
                Rect rect = Imgproc.boundingRect(contour);
                double aspectRatio = (double) rect.width / rect.height;
                double perimeter = Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);

                // UPDATED THRESHOLDS to match Python version
                if (aspectRatio > 0.2 && aspectRatio < 4.0 && perimeter > 20) { // Changed from (0.3, 3.0, 30) to (0.2, 4.0, 20)
                    int xNew = Math.max(0, rect.x - 5); // Add padding
                    int yNew = Math.max(0, rect.y - 5);
                    int width = Math.min(image.cols() - xNew, rect.width + 10); // Ensure within bounds
                    int height = Math.min(image.rows() - yNew, rect.height + 10);

                    if (width > 0 && height > 0) {
                        Mat cropRect = image.submat(yNew, yNew + height, xNew, xNew + width);
                        croppedImages.add(cropRect.clone());
                    }
                }
            }
        }
        return croppedImages;
    }

    private static Mat ManualA4Cropper(Mat undistort, Mat cameraM) {
        // Use zero distortion since image is already undistorted
        MatOfDouble zeroDist = new MatOfDouble(0, 0, 0, 0);

        // ArUco dictionary and detector
        Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        DetectorParameters parameters = DetectorParameters.create();
        List<Mat> corners = new ArrayList<>();
        Mat ids = new Mat();

        Aruco.detectMarkers(undistort, dictionary, corners, ids, parameters);

        if (ids.total() > 0) {
            // Estimate pose
            Mat rvec = new Mat();
            Mat tvec = new Mat();
            Aruco.estimatePoseSingleMarkers(corners, 5.0f, cameraM, zeroDist, rvec, tvec);

            // Define A4 paper corners in 3D
            MatOfPoint3f paperCorners3D = new MatOfPoint3f(
                    new Point3(-24.0, -14.0, 0),
                    new Point3(-2.6, -14.0, 0),
                    new Point3(-24.0, 5.0, 0),
                    new Point3(-2.6, 5.0, 0)
            );

            // Project to image
            MatOfPoint2f imagePoints = new MatOfPoint2f();
            Calib3d.projectPoints(paperCorners3D, rvec.row(0), tvec.row(0), cameraM, zeroDist, imagePoints);

            // Warp perspective
            MatOfPoint2f dstPoints = new MatOfPoint2f(
                    new org.opencv.core.Point(0, 0),
                    new org.opencv.core.Point(270, 0),
                    new org.opencv.core.Point(0, 200),
                    new org.opencv.core.Point(270, 200)
            );

            Mat transform = Imgproc.getPerspectiveTransform(imagePoints, dstPoints);
            Mat warped = new Mat();
            Imgproc.warpPerspective(undistort, warped, transform, new Size(270, 200));

            // Rotate 180 and flip horizontally
            Mat rotated = new Mat();
            Core.rotate(warped, rotated, Core.ROTATE_180);
            Mat flipped = new Mat();
            Core.flip(rotated, flipped, 1);

            return flipped;
        } else {
            return undistort;
        }
    }
}
