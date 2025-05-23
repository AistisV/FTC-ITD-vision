package org.firstinspires.ftc.teamcode.vision;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.firstinspires.ftc.robotcore.internal.camera.calibration.CameraCalibration;
import org.firstinspires.ftc.vision.VisionProcessor;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class SampleDetectionPipeline implements VisionProcessor {

    public SampleDetectionPipeline(Telemetry telemetry) {
        this.telemetry = telemetry;
    }

    public enum SelectedColor {
        RED,
        YELLOW,
        BLUE
    }

    static class AnalyzedSample {
        double angle;
        SelectedColor color;
        RotatedRect rotatedRect;
        Point centroid;
        Point3 worldCentroid;
        Point topPoint;
        Point segment1EndPoint;
        Point segment2EndPoint;
        List<Point> approxContourPoints;
    }

    // Image buffers
    Mat undistortedFrame = new Mat();
    Mat ycrcbMat = new Mat();
    Mat crMat = new Mat();
    Mat cbMat = new Mat();

    Mat thresholdMat = new Mat();
    Mat morphedThresholdMat = new Mat();
    Mat maskedColorMat = new Mat();
    Mat cannyMat = new Mat();
    Mat visualizationMat = new Mat();

    Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));

    // Camera parameters
    Mat cameraMatrix = new Mat(3, 3, CvType.CV_64FC1);
    Mat distCoeffs = new Mat(1, 5, CvType.CV_64FC1);
    public double cameraHeight = 26.7; // Height above the ground plane in CM
    public double cameraDownAngle = 42; // Angle in degrees the camera points downwards

    // Block height
    private final double blockHeight = 4; // In CM

    // Threshold values
    public Scalar redLower = new Scalar(0, 180, 80); // Can change second and third
    public Scalar redUpper = new Scalar(255, 255, 255);
    public Scalar yellowLower = new Scalar(0, 145, 0); // Can change second
    public Scalar yellowUpper = new Scalar(255, 255, 85); // Can change third
    public Scalar blueLower = new Scalar(0, 80, 150); // Can change third
    public Scalar blueUpper = new Scalar(255, 255, 255);

    public double minLengthForDirection = 1; // Adjust
    public double dotProductTresh = 0.8;
    public int searchWindow = 1;
    public int turnConfirmationTreshold = 1;
    public double approxPolyEpsilon = 0.02;

    public double MIN_AREA = 1000;
    public double MAX_AREA = 15000;
    public double MAX_ASPECT_RATIO = 6;

    private final int WIDTH = 640;
    private int HEIGHT = 480;

    private final int CALIBRATION_WIDTH = 2560;
    private final int CALIBRATION_HEIGHT = 1440;

    SelectedColor selectedColor = SelectedColor.RED;

    Telemetry telemetry = null;

    private void loadCameraCalibration(){
        double scaleX = (double) WIDTH / CALIBRATION_WIDTH;
        double scaleY = (double) HEIGHT / CALIBRATION_HEIGHT;

        double original_fx = 1514.7020418880806;
        double original_fy = 1514.4639462102391;
        double original_cx = 1291.6698643656734;
        double original_cy = 753.1148690399445;

        double new_fx = original_fx * scaleX;
        double new_fy = original_fy * scaleY;
        double new_cx = (original_cx / CALIBRATION_WIDTH) * WIDTH;   // Scale as a proportion
        double new_cy = (original_cy / CALIBRATION_HEIGHT) * HEIGHT; // Scale as a proportion

        double [] cameraMatrixData = new double[]{
                new_fx, 0.0, new_cx,
                0.0, new_fy, new_cy,
                0.0, 0.0, 1.0
        };
        cameraMatrix.put(0, 0, cameraMatrixData);

        double[] distCoeffsData = new double[]{
                0.18470664150127597, -0.42713561421364038, -0.0007563727080917796, 0.000007421943234453774, 0.2221525158164162
        };
        distCoeffs.put(0, 0, distCoeffsData);
    }

    private Point undistortPoint(Point distortedPoint) {
        MatOfPoint2f src = new MatOfPoint2f(new Point(distortedPoint.x, distortedPoint.y));
        MatOfPoint2f dst = new MatOfPoint2f();
        Calib3d.undistortPoints(src, dst, cameraMatrix, distCoeffs, new Mat());
        Point normalizedPoint = dst.toList().get(0);
        src.release();
        dst.release();

        double fx = cameraMatrix.get(0, 0)[0];
        double fy = cameraMatrix.get(1, 1)[0];
        double cx = cameraMatrix.get(0, 2)[0];
        double cy = cameraMatrix.get(1, 2)[0];

        double undistortedX = normalizedPoint.x * fx + cx;
        double undistortedY = normalizedPoint.y * fy + cy;

        return new Point(undistortedX, undistortedY);
    }

    @Override
    public void init(int width, int height, CameraCalibration calibration) {
        double aspectRatio = (double) width / height;
        HEIGHT = (int) (WIDTH / aspectRatio);

        loadCameraCalibration();
    }

    @Override
    public Object processFrame(Mat frame, long captureTimeNanos) {
//        Mat tempUndistortedFrame = new Mat();
        Imgproc.resize(frame, frame, new Size(WIDTH, HEIGHT));
//        visualizationMat = frame.clone();
//        Calib3d.undistort(frame, tempUndistortedFrame, cameraMatrix, distCoeffs);

        ArrayList<AnalyzedSample> detectedSamples = new ArrayList<>();
        findContours(frame, detectedSamples);
        calculateWorldCentroids(detectedSamples);

//        Core.multiply(visualizationMat, new Scalar(0.3, 0.3, 0.3), visualizationMat);
//        Core.addWeighted(visualizationMat, 0.7, maskedColorMat, 0.6, 0, visualizationMat);

        morphedThresholdMat.copyTo(frame);

        telemetry.update();

        return detectedSamples;
    }

    void findContours(Mat input, ArrayList<AnalyzedSample> outputList){
        // Convert the input image to YCrCb color space
//        Imgproc.GaussianBlur(input, ycrcbMat, new Size(5, 5), 0);
        Imgproc.cvtColor(input, ycrcbMat, Imgproc.COLOR_RGB2YCrCb);

        switch (selectedColor){
            case RED:
                Core.inRange(ycrcbMat, redLower, redUpper, thresholdMat);
                break;
            case BLUE:
                Core.inRange(ycrcbMat, blueLower, blueUpper, thresholdMat);
                break;
            case YELLOW:
                Core.inRange(ycrcbMat, yellowLower, yellowUpper, thresholdMat);
                break;
        }

        // Apply morphology for noise reduction
        Imgproc.morphologyEx(thresholdMat, morphedThresholdMat, Imgproc.MORPH_CLOSE, kernel, new Point(-1, -1), 2);
        Imgproc.morphologyEx(morphedThresholdMat, morphedThresholdMat, Imgproc.MORPH_OPEN, kernel, new Point(-1, -1), 1);


//        // Canny edge detection
//        maskedColorMat.release();
//        Core.bitwise_and(input, input, maskedColorMat, morphedThresholdMat);
//        Imgproc.Canny(maskedColorMat, cannyMat, 500, 500, 3, true);
//        Imgproc.dilate(cannyMat, cannyMat, kernel, new Point(-1, -1), 0);

        ArrayList<MatOfPoint> contoursList = new ArrayList<>();
        Imgproc.findContours(morphedThresholdMat, contoursList, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

//        telemetry.addData("Contours", contoursList.size());

        for (MatOfPoint contour : contoursList) {
            if (!filterContour(contour)) continue;

            analyzeContour(contour, selectedColor, outputList);
            contour.release();
        }

    }

    void analyzeContour(MatOfPoint contour, SelectedColor color, ArrayList<AnalyzedSample> outputList) {
        MatOfPoint2f originalContour2f = new MatOfPoint2f(contour.toArray());
        double arcLength = Imgproc.arcLength(originalContour2f, true);
        double epsilon = arcLength * approxPolyEpsilon;
        MatOfPoint2f approxContour2f = new MatOfPoint2f();
        Imgproc.approxPolyDP(originalContour2f, approxContour2f, epsilon, true);
        MatOfPoint approxContour = new MatOfPoint(approxContour2f.toArray());

        // Get the list of points for visualization and segment finding
        List<Point> approximatedPointsList = approxContour.toList();

        if(approximatedPointsList.size() < 3) return;

        // Calculate the centroid of the contour
        Moments moments = Imgproc.moments(contour);
        Point centroid = new Point(moments.m10 / moments.m00, moments.m01 / moments.m00);

        // Fit rotated rectangle
        RotatedRect distortedRotatedRect = Imgproc.minAreaRect(originalContour2f);
        Point[] rotatedRectPoints = new Point[4];
        distortedRotatedRect.points(rotatedRectPoints);

        Point initialTopPoint = approximatedPointsList.stream()
                .min(Comparator.comparingDouble(p -> p.y))
                .orElse(null);

        if(initialTopPoint == null) return;

        int initialTopIndex = approximatedPointsList.indexOf(initialTopPoint);
        if (initialTopIndex == -1) initialTopIndex = 0;

        // Search for the sharpest corner in a neighborhood around the initial highest point
        // This window defines how many points (on each side) around the initialTopIndex to check
        int searchWindowRadius = Math.min(searchWindow, approximatedPointsList.size() / 4); // Adjust this value, avoid too large for small contours
        // Capped at 1/4 of total points to prevent wrapping around
        double maxAngleChange = -1;
        int bestCornerIndex = initialTopIndex;

        for (int i = -searchWindowRadius; i <= searchWindowRadius; i++) {
            int currentIndex = (initialTopIndex + i + approximatedPointsList.size()) % approximatedPointsList.size();

            // To calculate angle change, we need points before and after the current point
            // Let's use a small step (e.g., 2 points away) to smooth out local noise
            int prevIndex = (currentIndex - 2 + approximatedPointsList.size()) % approximatedPointsList.size();
            int nextIndex = (currentIndex + 2) % approximatedPointsList.size();

            // Ensure we don't use the same point for calculation (e.g., if contour is very small)
            if (prevIndex == currentIndex || nextIndex == currentIndex || prevIndex == nextIndex) continue;

            Point p1 = approximatedPointsList.get(prevIndex); // Point before current
            Point p2 = approximatedPointsList.get(currentIndex); // Current point
            Point p3 = approximatedPointsList.get(nextIndex); // Point after current

            // Calculate vectors
            double dx1 = p2.x - p1.x;
            double dy1 = p2.y - p1.y;
            double dx2 = p3.x - p2.x;
            double dy2 = p3.y - p2.y;

            // Calculate angles of segments
            double angle1 = Math.atan2(dy1, dx1); // Angle of p1->p2 segment
            double angle2 = Math.atan2(dy2, dx2); // Angle of p2->p3 segment

            // Calculate the absolute change in angle (curvature)
            double angleChange = Math.abs(angle2 - angle1);
            // Normalize angleChange to be between 0 and PI (180 degrees)
            if (angleChange > Math.PI) {
                angleChange = 2 * Math.PI - angleChange;
            }

            // A sharper corner will have a larger angle change
            if (angleChange > maxAngleChange) {
                maxAngleChange = angleChange;
                bestCornerIndex = currentIndex;
            }
        }

        Point topPoint = approximatedPointsList.get(bestCornerIndex);

        ArrayList<Point> segment1 = getStraightSegment(approximatedPointsList, topPoint, true); // Clockwise
        ArrayList<Point> segment2 = getStraightSegment(approximatedPointsList, topPoint, false); // Counter-clockwise
        if (!(segment1.size() >= 2 && segment2.size() >= 2)) return;

        // Store the topPoint and the last point of each segment for drawing
        Point segment1EndPoint = segment1.get(segment1.size() - 1);
        Point segment2EndPoint = segment2.get(segment2.size() - 1);

        telemetry.addData("Top Point (Curvature-Refined)", topPoint.toString());
//        telemetry.addData("Max Angle Change", Math.toDegrees(maxAngleChange));
//        telemetry.addData("Segment1 End", segment1EndPoint.toString());
//        telemetry.addData("Segment2 End", segment2EndPoint.toString());
        telemetry.addData("Approximated Points Count", approximatedPointsList.size()); // Telemetry for approximated points


        // Calculate the world coordinates of each corner
        Point3[] worldCorners = new Point3[4];
        for (int i = 0; i < 4; i++) {
            worldCorners[i] = getWorldPosition(rotatedRectPoints[i]);
        }

        // Calculate the orientation from the world corners
        double angleWorld = calculateWorldAngle(worldCorners);
        telemetry.addData("Angle (World)", angleWorld);

        AnalyzedSample sample = new AnalyzedSample();
        sample.angle = angleWorld;
        sample.color = color;
        sample.rotatedRect = distortedRotatedRect;
        sample.centroid = centroid;
        sample.segment1EndPoint = segment1EndPoint; // Store segment end points
        sample.segment2EndPoint = segment2EndPoint; // Store segment end points
        sample.topPoint = topPoint;
        sample.approxContourPoints = approximatedPointsList;
        outputList.add(sample);

        originalContour2f.release();
        approxContour2f.release();
        approxContour.release();
    }

    private ArrayList<Point> getStraightSegment(List<Point> points, Point startPoint, boolean clockwise) {
        ArrayList<Point> segment = new ArrayList<>();
        int startIndex = points.indexOf(startPoint);
        if (startIndex == -1) return segment;

        segment.add(startPoint);
        Point prevPoint = startPoint;
        Point currentPoint;
        Vec2f overallDirection;
        int segmentLength = 0;

        int consecutiveTurnPoints = 0;

        for (int i = 1; i < points.size(); i++) {
            int index = (startIndex + (clockwise ? i : -i) + points.size()) % points.size();
            currentPoint = points.get(index);
            segment.add(currentPoint);
            segmentLength++;

            if (segmentLength > minLengthForDirection) {
                // Update overall direction
                overallDirection = new Vec2f((float) (currentPoint.x - startPoint.x), (float) (currentPoint.y - startPoint.y));
                Core.normalize(overallDirection, overallDirection);

                if (segmentLength > 2) {
                    Vec2f currentDirection = new Vec2f((float) (currentPoint.x - prevPoint.x), (float) (currentPoint.y - prevPoint.y));
                    Core.normalize(currentDirection, currentDirection);

                    double dotProduct = overallDirection.dot(currentDirection);

                    if (dotProduct < dotProductTresh) {
                        consecutiveTurnPoints++; // Increment if a turn is detected
                    } else {
                        consecutiveTurnPoints = 0; // Reset if the direction is good again
                    }

                    if (consecutiveTurnPoints >= turnConfirmationTreshold) {
                        // We remove the current point and all previously added "turn" points
                        // that led to the confirmation.
                        for (int k = 0; k < consecutiveTurnPoints; k++) {
                            if (segment.size() > 1) { // Ensure we don't remove the startPoint
                                segment.remove(segment.size() - 1);
                            } else {
                                break; // Should not happen if startPoint is the only element
                            }
                        }
                        break; // Stop adding points to this segment
                    }

                }
            }
            prevPoint = currentPoint;
        }
        return segment;
    }

    private static class Vec2f extends Mat {
        public Vec2f(float x, float y) {
            super(2, 1, CvType.CV_32FC1);
            put(0, 0, x);
            put(1, 0, y);
        }
        public float dot(Vec2f other) {
            return (float)(get(0, 0)[0] * other.get(0, 0)[0] + get(1, 0)[0] * other.get(1, 0)[0]);
        }
    }

    private double calculateWorldAngle(Point3[] corners) {
        // Calculate the squared length of two adjacent sides (to avoid sqrt)
        double lenSq01 = Math.pow(corners[1].x - corners[0].x, 2) + Math.pow(corners[1].y - corners[0].y, 2);
        double lenSq12 = Math.pow(corners[2].x - corners[1].x, 2) + Math.pow(corners[2].y - corners[1].y, 2);

        Point3 p1, p2;
        if (lenSq01 >= lenSq12) {
            p1 = corners[0];
            p2 = corners[1];
        } else {
            p1 = corners[1];
            p2 = corners[2];
        }

        // Calculate the angle of the longer side
        double deltaX = p2.x - p1.x;
        double deltaY = p2.y - p1.y;
        double angleRadians = Math.atan2(deltaY, deltaX);
        double angleDegrees = Math.toDegrees(angleRadians);

        // Normalize to 0-180
        if (angleDegrees < 0) {
            angleDegrees += 360;
        }
        if (angleDegrees > 180) {
            angleDegrees -= 180;
        }

        return angleDegrees;
    }

    boolean filterContour(MatOfPoint contour){
        double area = Imgproc.contourArea(contour);

        if (area < MIN_AREA || area > MAX_AREA) {
            return false;
        }

        org.opencv.core.Rect boundingRect = Imgproc.boundingRect(contour);
        double aspectRatio = (double) boundingRect.width / boundingRect.height;
        double aspectRatioInv = (double) boundingRect.height / boundingRect.width;

        return (aspectRatio <= MAX_ASPECT_RATIO && aspectRatioInv <= MAX_ASPECT_RATIO);
    }

    private void calculateWorldCentroids(ArrayList<AnalyzedSample> detectedSamples) {
        for (AnalyzedSample sample : detectedSamples) {
            sample.worldCentroid = getWorldPosition(sample.centroid);
            telemetry.addData("World Centroid X", sample.worldCentroid.x);
            telemetry.addData("World Centroid Y", sample.worldCentroid.y);
        }
    }

    Point3 getWorldPosition(Point imagePoint) {
        // Undistort and normalize
        Point undistortedPoint = undistortPoint(imagePoint);
        double x_n = (undistortedPoint.x - cameraMatrix.get(0, 2)[0]) / cameraMatrix.get(0, 0)[0];
        double y_n = (undistortedPoint.y - cameraMatrix.get(1, 2)[0]) / cameraMatrix.get(1, 1)[0];
        Point3 rayDirectionCamera = new Point3(x_n, y_n, 1);

        // World ray direction
        double angleRadians = Math.toRadians(cameraDownAngle);
        Mat cameraRotationMatrix = new Mat(3, 3, CvType.CV_64FC1);
        double[] rotationData = {1, 0, 0, 0, Math.cos(angleRadians), -Math.sin(angleRadians), 0, Math.sin(angleRadians), Math.cos(angleRadians)};
        cameraRotationMatrix.put(0, 0, rotationData);
        Point3 rayDirectionWorld = transformDirectionToWorld(rayDirectionCamera, cameraRotationMatrix);
        cameraRotationMatrix.release();

        // Camera origin
        Point3 rayOriginWorld = new Point3(0, 0, cameraHeight);

        // Calculate 't' for intersection with the ground plane (Z=0)
        double t = (0 - rayOriginWorld.z) / rayDirectionWorld.z;

        // Calculate the 3D position of the intersection point on the ground
        Point3 intersectionPointWorld = new Point3(
                rayOriginWorld.x + rayDirectionWorld.x * t,
                rayOriginWorld.y + rayDirectionWorld.y * t,
                0
        );

        return intersectionPointWorld;
    }

    Point3 transformDirectionToWorld(Point3 rayDirectionCamera, Mat cameraRotation) {
        Mat rayDirectionCameraMat = new Mat(3, 1, CvType.CV_64FC1);
        rayDirectionCameraMat.put(0, 0, rayDirectionCamera.x);
        rayDirectionCameraMat.put(1, 0, rayDirectionCamera.y);
        rayDirectionCameraMat.put(2, 0, rayDirectionCamera.z);

        Mat rayDirectionWorldMat = new Mat(3, 1, CvType.CV_64FC1);
        Core.gemm(cameraRotation, rayDirectionCameraMat, 1.0, new Mat(), 0.0, rayDirectionWorldMat, 0);

        return new Point3(
                rayDirectionWorldMat.get(0, 0)[0],
                rayDirectionWorldMat.get(1, 0)[0],
                rayDirectionWorldMat.get(2, 0)[0]
        );
    }

    @SuppressWarnings("unchecked")
    @Override
    public void onDrawFrame(Canvas canvas, int onscreenWidth, int onscreenHeight, float scaleBmpPxToCanvasPx, float scaleCanvasDensity, Object userContext) {
        if (!(userContext instanceof ArrayList)) {
            return;
        }

        ArrayList<AnalyzedSample> samplesToDraw = (ArrayList<AnalyzedSample>) userContext;

        Paint distortedPaint = new Paint();
        distortedPaint.setColor(Color.RED);
        distortedPaint.setStyle(Paint.Style.STROKE);
        distortedPaint.setStrokeWidth(scaleCanvasDensity * 2);

        Paint centerPaint = new Paint();
        centerPaint.setColor(Color.YELLOW);
        centerPaint.setStyle(Paint.Style.FILL);
        centerPaint.setStrokeWidth(scaleCanvasDensity * 2);

        Paint linePaint = new Paint();
        linePaint.setColor(Color.GREEN);
        linePaint.setStrokeWidth(scaleCanvasDensity * 5);

        Paint approxPointPaint = new Paint();
        approxPointPaint.setColor(Color.MAGENTA);
        approxPointPaint.setStyle(Paint.Style.FILL);
        approxPointPaint.setStrokeWidth(scaleCanvasDensity * 1);

        for (AnalyzedSample sample : samplesToDraw) {
            RotatedRect distortedRect = sample.rotatedRect;
            Point[] distortedPoints = new Point[4];
            distortedRect.points(distortedPoints);

            for (int i = 0; i < 4; ++i) {
                canvas.drawLine(
                        (float) (distortedPoints[i].x * scaleBmpPxToCanvasPx),
                        (float) (distortedPoints[i].y * scaleBmpPxToCanvasPx),
                        (float) (distortedPoints[(i + 1) % 4].x * scaleBmpPxToCanvasPx),
                        (float) (distortedPoints[(i + 1) % 4].y * scaleBmpPxToCanvasPx),
                        distortedPaint
                );
            }

            canvas.drawCircle(
                    (float) (sample.centroid.x * scaleBmpPxToCanvasPx),
                    (float) (sample.centroid.y * scaleBmpPxToCanvasPx),
                    5 * scaleCanvasDensity,
                    centerPaint
            );

            // Draw the lines using the segment endpoints
            if (sample.topPoint != null && sample.segment1EndPoint != null && sample.segment2EndPoint != null) {

                // Line 1: from topPoint to segment1EndPoint
                canvas.drawLine(
                        (float) (sample.topPoint.x * scaleBmpPxToCanvasPx),
                        (float) (sample.topPoint.y * scaleBmpPxToCanvasPx),
                        (float) (sample.segment1EndPoint.x * scaleBmpPxToCanvasPx),
                        (float) (sample.segment1EndPoint.y * scaleBmpPxToCanvasPx),
                        linePaint
                );

                // Line 2: from topPoint to segment2EndPoint
                canvas.drawLine(
                        (float) (sample.topPoint.x * scaleBmpPxToCanvasPx),
                        (float) (sample.topPoint.y * scaleBmpPxToCanvasPx),
                        (float) (sample.segment2EndPoint.x * scaleBmpPxToCanvasPx),
                        (float) (sample.segment2EndPoint.y * scaleBmpPxToCanvasPx),
                        linePaint
                );
            }


            for (Point p : sample.approxContourPoints) {
                canvas.drawCircle(
                        (float) (p.x * scaleBmpPxToCanvasPx),
                        (float) (p.y * scaleBmpPxToCanvasPx),
                        3 * scaleCanvasDensity,
                        approxPointPaint
                );
            }


        }
    }

}
