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
    public Scalar redLower = new Scalar(0, 170, 90); // Can change second and third
    public Scalar redUpper = new Scalar(255, 255, 255);
    public Scalar yellowLower = new Scalar(0, 145, 0); // Can change second
    public Scalar yellowUpper = new Scalar(255, 255, 85); // Can change third
    public Scalar blueLower = new Scalar(0, 80, 150); // Can change third
    public Scalar blueUpper = new Scalar(255, 255, 255);

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
        visualizationMat = frame.clone();
//        Calib3d.undistort(frame, tempUndistortedFrame, cameraMatrix, distCoeffs);

        ArrayList<AnalyzedSample> detectedSamples = new ArrayList<>();
        findContours(frame, detectedSamples);
        calculateWorldCentroids(detectedSamples);

        Core.multiply(visualizationMat, new Scalar(0.3, 0.3, 0.3), visualizationMat);
        Core.addWeighted(visualizationMat, 0.7, maskedColorMat, 0.6, 0, visualizationMat);

        visualizationMat.copyTo(frame);

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

        // Canny edge detection
        maskedColorMat.release();
        Core.bitwise_and(input, input, maskedColorMat, morphedThresholdMat);
        Imgproc.Canny(maskedColorMat, cannyMat, 100, 250);
        Imgproc.dilate(cannyMat, cannyMat, kernel, new Point(-1, -1), 1);

        ArrayList<MatOfPoint> contoursList = new ArrayList<>();
        Imgproc.findContours(cannyMat, contoursList, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

//        telemetry.addData("Contours", contoursList.size());

        for (MatOfPoint contour : contoursList) {
            if (!filterContour(contour)) continue;
            analyzeContour(contour, selectedColor, outputList);
            contour.release();
        }

    }

    void analyzeContour(MatOfPoint contour, SelectedColor color, ArrayList<AnalyzedSample> outputList){
        // Calculate the centroid of the contour
        Moments moments = Imgproc.moments(contour);
        Point centroid = new Point(moments.m10 / moments.m00, moments.m01 / moments.m00);

        // Fit rotated rectangle
        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
        RotatedRect distortedRotatedRect = Imgproc.minAreaRect(contour2f);

        // Get the corner points of the "distorted" rotated rectangle
        Point[] distortedPoints = new Point[4];
        distortedRotatedRect.points(distortedPoints);

        // Calculate the world coordinates of each corner
        Point3[] worldCorners = new Point3[4];
        for (int i = 0; i < 4; i++) {
            worldCorners[i] = getWorldPosition(distortedPoints[i]);
        }

        // Calculate the orientation from the world corners
        double angleWorld = calculateWorldAngle(worldCorners);
        telemetry.addData("Angle (World)", angleWorld);

        AnalyzedSample sample = new AnalyzedSample();
        sample.angle = angleWorld;
        sample.color = color;
        sample.rotatedRect = distortedRotatedRect;
        sample.centroid = centroid;
        outputList.add(sample);

        contour2f.release();
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
        }
    }

}
