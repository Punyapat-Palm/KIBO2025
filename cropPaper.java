package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.util.Log;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import org.opencv.aruco.Aruco;
import org.opencv.aruco.DetectorParameters;
import org.opencv.aruco.Dictionary;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */

public class YourService extends KiboRpcService {
    private final String TAG = this.getClass().getSimpleName();
    @Override
    protected void runPlan1(){
        // The mission starts.
        api.startMission();
        Log.i(TAG, "start mission");
        // Move to a point.
        Point point = new Point(10.9d, -9.92284d, 5.195d);
        Quaternion quaternion = new Quaternion(0f, 0f, -0.707f, 0.707f);
        api.moveTo(point, quaternion, false);

        // Get a camera image.
        Mat image = api.getMatNavCam();

        //get camera matrix
        Mat cameraM = new Mat(3, 3, CvType.CV_64F);
        cameraM.put(0, 0, api.getNavCamIntrinsics()[0]);
        // Get lens distrotion parameters
        Mat cameraCoeff = new Mat(1, 5, CvType.CV_64F);
        cameraCoeff.put(0, 0, api.getNavCamIntrinsics()[1]);
        cameraCoeff.convertTo(cameraCoeff, CvType.CV_64F);

        //Undistort image
        Mat undistort = new Mat();
        Calib3d.undistort(image, undistort, cameraM, cameraCoeff);
        api.saveMatImage(undistort, "undistort.png");
        Log.i(TAG, "Undistort image size: " + undistort.size() + ", channels: " + undistort.channels());

        // Get A4 cropped image
        //Mat A4Crop = ArucoA4Cropper(undistort, cameraM);
        Log.i(TAG, "start cropping image");
        DocumentScanner scanner = new DocumentScanner();
        Mat A4Crop = scanner.processDocument(undistort);

        // Check if A4 crop was successful
        if (A4Crop != null && !A4Crop.empty()) {
            api.saveMatImage(A4Crop, "A4Crop.png");
        }
        /* ******************************************************************************** */
        /* Write your code to recognize the type and number of landmark items in each area! */
        /* If there is a treasure item, remember it.                                        */
        /* ******************************************************************************** */

        // When you recognize landmark items, let’s set the type and number.
        api.setAreaInfo(1, "item_name", 1);

        /* **************************************************** */
        /* Let's move to each area and recognize the items. */
        /* **************************************************** */

        // When you move to the front of the astronaut, report the rounding completion.
        point = new Point(11.143d, -6.7607d, 4.9654d);
        quaternion = new Quaternion(0f, 0f, 0.707f, 0.707f);
        api.moveTo(point, quaternion, false);
        api.reportRoundingCompletion();

        /* ********************************************************** */
        /* Write your code to recognize which target item the astronaut has. */
        /* ********************************************************** */

        // Let's notify the astronaut when you recognize it.
        api.notifyRecognitionItem();

        /* ******************************************************************************************************* */
        /* Write your code to move Astrobee to the location of the target item (what the astronaut is looking for) */
        /* ******************************************************************************************************* */

        // Take a snapshot of the target item.
        api.takeTargetItemSnapshot();
    }

    @Override
    protected void runPlan2(){
        // write your plan 2 here.
    }

    @Override
    protected void runPlan3(){
        // write your plan 3 here.
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

            boolean arucoDetected = detectAruco(gray, imgCopy);

            Mat edged = preprocessEdges(gray);
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(edged, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

            if (contours.isEmpty()) {
                Log.i(TAG, "[ERROR] No contours found");
                return inputImage;
            }

            MatOfPoint2f biggest = findBiggestContour(contours);
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
                Mat cropped = cropRightCmRatio(rotated);

                Log.i(TAG, "[SUCCESS] Document processing completed");
                return cropped;
            } catch (Exception e) {
                Log.i(TAG, "[ERROR] Processing failed: " + e.getMessage());
                return inputImage;
            } finally {
                // Release temporary matrices
                gray.release();
                imgCopy.release();
            }
        }

        private boolean detectAruco(Mat gray, Mat imgDraw) {
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

        private MatOfPoint2f findBiggestContour(List<MatOfPoint> contours) {
            double maxArea = 0;
            MatOfPoint2f biggest = null;

            for (MatOfPoint contour : contours) {
                double area = Imgproc.contourArea(contour);
                if (area > 1000) {
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

            // A4 width ratio: crop to 15.7/21.0 of original width
            double cropFraction = 15.7 / 21.0;
            int cropWidth = (int) (width * cropFraction);

            if (cropWidth <= 0) {
                Log.i(TAG, "Image is too narrow to crop");
                return image;
            }

            Rect cropRect = new Rect(0, 0, cropWidth, height);
            Mat cropped = new Mat(image, cropRect);

            Log.i(TAG, "Cropped image to: " + cropped.size());
            return cropped;
        }
    }

    private static List<Mat> cropObjects(Mat image) {
        // Convert to grayscale (check if already grayscale)
        Mat gray = new Mat();
        if (image.channels() == 3) {
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
        } else if (image.channels() == 4) {
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGRA2GRAY);
        } else {
            // Image is already grayscale
            gray = image.clone();
        }

        // Apply Gaussian blur to reduce noise
        Mat blur = new Mat();
        Imgproc.GaussianBlur(gray, blur, new Size(5, 5), 0);

        // Increase contrast
        Mat imageContrast = new Mat();
        double alpha = 1.5; // Contrast control (1.0-3.0)
        double beta = 0;    // Brightness control (changed from int to double)
        gray.convertTo(imageContrast, CvType.CV_8UC1, alpha, beta);

        // Adaptive thresholding
        Mat thresh = new Mat();
        Imgproc.adaptiveThreshold(imageContrast, thresh, 255,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY_INV, 15, 5);

        // Find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(thresh, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // List to store cropped images
        List<Mat> croppedImages = new ArrayList<>();

        // Crop each contour with square aspect ratio
        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint contour = contours.get(i);

            double area = Imgproc.contourArea(contour);
            if (area > 500) { // Only minimum area filter, no maximum
                Rect boundingRect = Imgproc.boundingRect(contour);
                int x = boundingRect.x;
                int y = boundingRect.y;
                int w = boundingRect.width;
                int h = boundingRect.height;

                // Determine the size of the square
                int sideLength = Math.max(w, h);
                int padding = 5;

                // Center the square around the original bounding box
                int centerX = x + w / 2;
                int centerY = y + h / 2;

                // Calculate new square bounds
                int xNew = Math.max(0, centerX - sideLength / 2 - padding);
                int yNew = Math.max(0, centerY - sideLength / 2 - padding);
                int side = sideLength + 2 * padding;

                // Ensure the square crop doesn't exceed image boundaries
                xNew = Math.min(xNew, image.cols() - side);
                yNew = Math.min(yNew, image.rows() - side);

                // Ensure valid dimensions
                if (xNew >= 0 && yNew >= 0 && xNew + side <= image.cols() && yNew + side <= image.rows() && side > 0) {
                    // Crop and store the square image
                    Rect cropRect = new Rect(xNew, yNew, side, side);
                    Mat croppedImage = new Mat(image, cropRect);
                    croppedImages.add(croppedImage.clone()); // Clone to avoid reference issues
                }
            }
        }

        // Clean up temporary matrices
        gray.release();
        blur.release();
        imageContrast.release();
        thresh.release();
        hierarchy.release();

        return croppedImages;
    }
}
