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
        Point distortedCentroid;
    }

    // Image buffers
    Mat undistortedFrame = new Mat();
    Mat ycrcbMat = new Mat();
    Mat crMat = new Mat();
    Mat cbMat = new Mat();

    Mat thresholdMat = new Mat();
    Mat morphedThreshold = new Mat();
    Mat maskedColor = new Mat();
    Mat cannyMat = new Mat();

    Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));

    Mat cameraMatrix = new Mat(3, 3, CvType.CV_64FC1);
    Mat distCoeffs = new Mat(1, 5, CvType.CV_64FC1);

    // Threshold values
//    public int YELLOW_MASK_THRESHOLD = 75;
//    public int BLUE_MASK_THRESHOLD = 150;
//    public int RED_MASK_THRESHOLD = 195;
    public Scalar redLower = new Scalar(0, 170, 90); // Can change second and third
    public Scalar redUpper = new Scalar(255, 255, 255);
    public Scalar yellowLower = new Scalar(0, 145, 0); // Can change second
    public Scalar yellowUpper = new Scalar(255, 255, 85); // Can change third
    public Scalar blueLower = new Scalar(0, 80, 150); // Can change third
    public Scalar blueUpper = new Scalar(255, 255, 255); // Can change first

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

        // Perform the projection to get pixel coordinates
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
//        Calib3d.undistort(frame, tempUndistortedFrame, cameraMatrix, distCoeffs);

        ArrayList<AnalyzedSample> detectedSamples = new ArrayList<>();
        findContours(frame, detectedSamples);

//        Imgproc.cvtColor(ycrcbMat, ycrcbMat, Imgproc.COLOR_YCrCb2RGB);
//        ycrcbMat.copyTo(frame);
        maskedColor.copyTo(frame);
//        tempUndistortedFrame.release();

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
        Imgproc.morphologyEx(thresholdMat, morphedThreshold, Imgproc.MORPH_CLOSE, kernel, new Point(-1, -1), 2);
        Imgproc.morphologyEx(morphedThreshold, morphedThreshold, Imgproc.MORPH_OPEN, kernel, new Point(-1, -1), 1);

        // Canny edge detection
        maskedColor.release();
        Core.bitwise_and(input, input, maskedColor, morphedThreshold);
        Imgproc.Canny(maskedColor, cannyMat, 100, 250);
        Imgproc.dilate(cannyMat, cannyMat, kernel, new Point(-1, -1), 1);
//        Imgproc.morphologyEx(cannyMat, cannyMat, Imgproc.MORPH_OPEN, kernel, new Point(-1, -1), 0);

        ArrayList<MatOfPoint> contoursList = new ArrayList<>();
        Imgproc.findContours(cannyMat, contoursList, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        telemetry.addData("Contours", contoursList.size());

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

        // Undistort the centroid
        Point undistortedCentroid = undistortPoint(centroid);

        // Fit rotated rectangle
        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
        RotatedRect rotatedRect = Imgproc.minAreaRect(contour2f);

        double angle = rotatedRect.angle;
        // Adjust the angle based on rectangle dimensions and normalize to -90 to 90
        if (rotatedRect.size.width < rotatedRect.size.height) {
            angle += 90; // Rotate by 90 degrees if the rectangle more tall than wide
        }
        // Normalize angle to range -90 to 90
        if (angle > 90) {
            angle -= 180;
        }
        telemetry.addData("Angle", angle);
        telemetry.addData("Undistorted Centroid X", undistortedCentroid.x);
        telemetry.addData("Undistorted Centroid Y", undistortedCentroid.y);

        AnalyzedSample sample = new AnalyzedSample();
        sample.angle = angle;
        sample.color = color;
        sample.rotatedRect = rotatedRect;
        sample.centroid = undistortedCentroid; // Store the undistorted centroid
        sample.distortedCentroid = centroid;
        outputList.add(sample);

        contour2f.release();
//        approxCurve.release();
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

    @SuppressWarnings("unchecked")
    @Override
    public void onDrawFrame(Canvas canvas, int onscreenWidth, int onscreenHeight, float scaleBmpPxToCanvasPx, float scaleCanvasDensity, Object userContext) {
        if (!(userContext instanceof ArrayList)) {
            return;
        }

        ArrayList<AnalyzedSample> samplesToDraw = (ArrayList<AnalyzedSample>) userContext;

        Paint paint = new Paint();
        paint.setColor(Color.GREEN);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(scaleCanvasDensity * 4);

        Paint centerPaint = new Paint();
        centerPaint.setColor(Color.YELLOW);
        centerPaint.setStyle(Paint.Style.FILL);
        centerPaint.setStrokeWidth(scaleCanvasDensity * 2);

        Paint undistortedCenterPaint = new Paint();
        undistortedCenterPaint.setColor(Color.BLUE);
        undistortedCenterPaint.setStyle(Paint.Style.FILL);
        undistortedCenterPaint.setStrokeWidth(scaleCanvasDensity * 2);

        for (AnalyzedSample sample : samplesToDraw) {
            RotatedRect rotatedRect = sample.rotatedRect;
            Point[] points = new Point[4];
            rotatedRect.points(points);

            for (int i = 0; i < 4; ++i) {
                canvas.drawLine(
                        (float) (points[i].x * scaleBmpPxToCanvasPx),
                        (float) (points[i].y * scaleBmpPxToCanvasPx),
                        (float) (points[(i + 1) % 4].x * scaleBmpPxToCanvasPx),
                        (float) (points[(i + 1) % 4].y * scaleBmpPxToCanvasPx),
                        paint
                );
            }
            // Draw a small circle at the center
            canvas.drawCircle(
                    (float) (sample.centroid.x * scaleBmpPxToCanvasPx),
                    (float) (sample.centroid.y * scaleBmpPxToCanvasPx),
                    5 * scaleCanvasDensity, // Adjust radius as needed
                    centerPaint
            );
//            // Undistort the rectangle's center for visualization
//            Point undistortedRectCenter = undistortPoint(rotatedRect.center);
//
//            // Draw the undistorted center
//            if (undistortedRectCenter != null) {
//                canvas.drawCircle(
//                        (float) (undistortedRectCenter.x * scaleBmpPxToCanvasPx),
//                        (float) (undistortedRectCenter.y * scaleBmpPxToCanvasPx),
//                        7 * scaleCanvasDensity,
//                        undistortedCenterPaint
//                );
//            }
            // Draw a small circle at the undistorted centroid
            if (sample.centroid != null) {
                canvas.drawCircle(
                        (float) (sample.distortedCentroid.x * scaleBmpPxToCanvasPx),
                        (float) (sample.distortedCentroid.y * scaleBmpPxToCanvasPx),
                        7 * scaleCanvasDensity,
                        undistortedCenterPaint
                );
            }
        }

    }

}
