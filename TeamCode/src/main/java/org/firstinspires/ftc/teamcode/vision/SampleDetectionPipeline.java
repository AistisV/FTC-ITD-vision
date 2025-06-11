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
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.util.ArrayList;
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

    private enum Hypothesis {
        HYPOTHESIS_A, // worldTopPoint corresponds to model_corner_A ((0,0,0) local)
        HYPOTHESIS_B  // worldTopPoint corresponds to model_corner_B ((BLOCK_LENGTH, 0, 0) local)
    }

    static class AnalyzedSample {
        double angle;
        SelectedColor color;
        RotatedRect rotatedRect;
        Point3 worldCentroid;
        List<Point> approxContourPoints;
        Point topPoint2D;
        Point3 worldTopPoint;
        List<Point> projectedOptimalPoints;
        Point3 optimalWorldOrigin;
    }

    // Image buffers
    Mat ycrcbMat = new Mat();
    Mat thresholdMat = new Mat();
    Mat morphedThresholdMat = new Mat();
    Mat maskedColorMat = new Mat();
    Mat visualizationMat = new Mat();
    Mat distTransformMat = new Mat();

    Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));

    // Camera parameters
    Mat cameraMatrix = new Mat(3, 3, CvType.CV_64FC1);
    Mat distCoeffs = new Mat(1, 5, CvType.CV_64FC1);
    public double cameraHeight = 26.7; // Height above the ground plane in CM
    public double cameraDownAngle = 42; // Angle in degrees the camera points downwards (90 degrees is straight down)

    // Block height
    private final double BLOCK_HEIGHT = 3.7; // In CM
    private final double BLOCK_WIDTH = 3.7; // In CM
    private final double BLOCK_LENGTH = 8.7; // In CM

    public double approxPolyEpsilon = 0.02;

    // Threshold values
    public Scalar redLower = new Scalar(0, 180, 80); // Can change second and third
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


    private List<Point3> getTopFaceModelPoints() {
        List<Point3> points = new ArrayList<>();
        // Define the 4 vertices of the top face in a local 3D coordinate system.
        // Assuming the origin (0,0,0) of this local system is at one of the top corners.
        // The Z-coordinate is 0 here as it's relative to the block's own top face.
        points.add(new Point3(0, 0, 0)); // Top-Front-Left corner (P_model_A)
        points.add(new Point3(BLOCK_LENGTH, 0, 0)); // Top-Front-Right corner (P_model_B)
        points.add(new Point3(BLOCK_LENGTH, BLOCK_WIDTH, 0)); // Top-Back-Right corner
        points.add(new Point3(0, BLOCK_WIDTH, 0)); // Top-Back-Left corner
        return points;
    }

    /**
     * Calculates the camera's pose (rotation and translation) in the world coordinate system.
     * This defines the camera's position and orientation relative to the world origin.
     *
     * @param rvec_cam_to_world Output Mat for the camera's rotation vector relative to the world.
     * @param tvec_cam_to_world Output Mat for the camera's translation vector relative to the world.
     */
    private void getStaticCameraPoseInWorld(Mat rvec_cam_to_world, Mat tvec_cam_to_world) {
        // Camera position in world coordinates (from world origin to camera origin)
        tvec_cam_to_world.put(0, 0, 0); // X-world
        tvec_cam_to_world.put(1, 0, 0); // Y-world
        tvec_cam_to_world.put(2, 0, cameraHeight); // Z-world (height above ground)

        // Camera rotation in world coordinates (rotation from world frame to camera frame)
        // Camera points down along world's Y-axis (or rotated around world's X-axis)
        // A rotation around the world's X-axis by cameraDownAngle degrees.
        // Note: OpenCV's Rodrigues requires radians.
        double angleRad = Math.toRadians(cameraDownAngle);

        // Rotation matrix for X-axis rotation:
        // [ 1       0              0      ]
        // [ 0   cos(angle)  -sin(angle) ]
        // [ 0   sin(angle)   cos(angle) ]
        Mat R_x = new Mat(3, 3, CvType.CV_64FC1);
        R_x.put(0, 0, 1); R_x.put(0, 1, 0);               R_x.put(0, 2, 0);
        R_x.put(1, 0, 0); R_x.put(1, 1, Math.cos(angleRad)); R_x.put(1, 2, -Math.sin(angleRad));
        R_x.put(2, 0, 0); R_x.put(2, 1, Math.sin(angleRad)); R_x.put(2, 2, Math.cos(angleRad));

        // Convert rotation matrix to Rodrigues vector
        Calib3d.Rodrigues(R_x, rvec_cam_to_world);

        R_x.release(); // Important to release this Mat
    }

    /**
     * Projects the 3D model points of the block onto the 2D image plane,
     * given the block's world pose (rotation and translation in world coordinates).
     *
     * @param objectWorldRvec The rotation vector of the object in world coordinates.
     * @param objectWorldTvec The translation vector of the object's origin in world coordinates.
     * @return A List of OpenCV Point objects representing the projected 2D points.
     */
    public List<Point> projectModelPoints(Mat objectWorldRvec, Mat objectWorldTvec) {
        // --- 1. Get Camera's Pose in WORLD Coordinates (Extrinsics) ---
        Mat rvec_cam_to_world = new Mat(3, 1, CvType.CV_64FC1);
        Mat tvec_cam_to_world = new Mat(3, 1, CvType.CV_64FC1);
        getStaticCameraPoseInWorld(rvec_cam_to_world, tvec_cam_to_world);

        // --- 2. Calculate Object's Pose Relative to CAMERA ---
        // This is what projectPoints expects.
        // P_camera = R_camera_world * P_world + T_camera_world
        // where R_camera_world = R_world_camera.t() and T_camera_world = -R_camera_world * T_world_camera

        Mat R_cam_to_world = new Mat(3, 3, CvType.CV_64FC1);
        Calib3d.Rodrigues(rvec_cam_to_world, R_cam_to_world); // Convert rvec to rotation matrix

        Mat R_world_to_cam = R_cam_to_world.t(); // Inverse rotation (transpose)

        Mat t_cam_in_world = tvec_cam_to_world; // Camera's translation in world

        Mat tvec_world_to_cam = new Mat(3, 1, CvType.CV_64FC1);
        Core.gemm(R_world_to_cam, t_cam_in_world, -1.0, new Mat(), 0.0, tvec_world_to_cam, 0);

        // Now, apply this transformation to the object's world pose to get its pose relative to the camera
        // R_obj_cam = R_world_to_cam * R_obj_world
        // t_obj_cam = R_world_to_cam * t_obj_world + t_world_to_cam

        Mat R_obj_world = new Mat(3, 3, CvType.CV_64FC1);
        Calib3d.Rodrigues(objectWorldRvec, R_obj_world); // Use the input object's world rvec

        Mat rvec_obj_cam = new Mat(3, 1, CvType.CV_64FC1);
        Mat R_obj_cam_mat = new Mat(3, 3, CvType.CV_64FC1);
        Core.gemm(R_world_to_cam, R_obj_world, 1.0, new Mat(), 0.0, R_obj_cam_mat, 0);
        Calib3d.Rodrigues(R_obj_cam_mat, rvec_obj_cam);

        Mat tvec_obj_cam = new Mat(3, 1, CvType.CV_64FC1);
        Mat temp_t = new Mat(3,1,CvType.CV_64FC1);
        Core.gemm(R_world_to_cam, objectWorldTvec, 1.0, new Mat(), 0.0, temp_t, 0); // R_world_to_cam * t_obj_world
        Core.add(temp_t, tvec_world_to_cam, tvec_obj_cam); // + t_world_to_cam


        // --- Telemetry (Optional, for debugging transformed poses) ---
        telemetry.addData("Obj World tvec (X,Y,Z)", String.format("%.2f, %.2f, %.2f", objectWorldTvec.get(0,0)[0], objectWorldTvec.get(1,0)[0], objectWorldTvec.get(2,0)[0]));
        telemetry.addData("Cam World tvec (X,Y,Z)", String.format("%.2f, %.2f, %.2f", tvec_cam_to_world.get(0,0)[0], tvec_cam_to_world.get(1,0)[0], tvec_cam_to_world.get(2,0)[0]));
        telemetry.addData("Obj Cam tvec (X,Y,Z)", String.format("%.2f, %.2f, %.2f", tvec_obj_cam.get(0,0)[0], tvec_obj_cam.get(1,0)[0], tvec_obj_cam.get(2,0)[0]));


        List<Point3> modelPoints = getTopFaceModelPoints(); // Uses the block's model

        if (modelPoints.isEmpty()) {
            telemetry.addData("Error", "Model points list is empty!");
            // Release all Mats before returning
            rvec_cam_to_world.release(); tvec_cam_to_world.release();
            R_cam_to_world.release(); R_world_to_cam.release();
            tvec_world_to_cam.release(); R_obj_world.release();
            rvec_obj_cam.release(); R_obj_cam_mat.release();
            tvec_obj_cam.release(); temp_t.release();
            return new ArrayList<>();
        }

        MatOfPoint3f modelPointsMat = new MatOfPoint3f(modelPoints.toArray(new Point3[0]));
        MatOfPoint2f projectedPoints = new MatOfPoint2f();
        MatOfDouble distCoeffsMatOfDouble = new MatOfDouble(distCoeffs);

        // Perform the projection using the calculated object pose relative to the camera
        try {
            Calib3d.projectPoints(modelPointsMat, rvec_obj_cam, tvec_obj_cam, cameraMatrix, distCoeffsMatOfDouble, projectedPoints);
        } catch (Exception e) {
            telemetry.addData("Calib3d.projectPoints Error", e.getMessage());
            // Release all Mats before returning
            rvec_cam_to_world.release(); tvec_cam_to_world.release();
            R_cam_to_world.release(); R_world_to_cam.release();
            tvec_world_to_cam.release(); R_obj_world.release();
            rvec_obj_cam.release(); R_obj_cam_mat.release();
            tvec_obj_cam.release(); temp_t.release();
            modelPointsMat.release(); distCoeffsMatOfDouble.release();
            return new ArrayList<>();
        }

        // Telemetry *after* projection
        telemetry.addData("Projected Pts Status", "Projected " + projectedPoints.rows() + " points.");
        if (projectedPoints.rows() > 0) {
            telemetry.addData("First Projected Pt", String.format("%.2f, %.2f", projectedPoints.get(0, 0)[0], projectedPoints.get(0, 0)[1]));
            telemetry.addData("Last Projected Pt", String.format("%.2f, %.2f", projectedPoints.get(projectedPoints.rows()-1, 0)[0], projectedPoints.get(projectedPoints.rows()-1, 0)[1]));
        } else {
            telemetry.addData("Projected Pts", "None generated.");
        }

        List<Point> projectedPointsList = projectedPoints.toList();

        // Release all resources
        rvec_cam_to_world.release(); tvec_cam_to_world.release();
        R_cam_to_world.release(); R_world_to_cam.release();
        tvec_world_to_cam.release(); R_obj_world.release();
        rvec_obj_cam.release(); R_obj_cam_mat.release();
        tvec_obj_cam.release(); temp_t.release();
        modelPointsMat.release(); distCoeffsMatOfDouble.release();

        return projectedPointsList;
    }

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
        Imgproc.resize(frame, frame, new Size(WIDTH, HEIGHT));

        ArrayList<AnalyzedSample> detectedSamples = new ArrayList<>();
        findContours(frame, detectedSamples);

        // --- START DEBUGGING VISUALIZATION ---
        // This will overwrite the main camera feed with a view of the distance transform.
        // Bright areas are "good" (high cost value), dark areas are "bad".
        if (!distTransformMat.empty()) {
            Mat dtVis = new Mat();
            // Normalize the distance transform to a 0-255 range for visualization
            Core.normalize(distTransformMat, dtVis, 0, 255, Core.NORM_MINMAX);
            dtVis.convertTo(dtVis, CvType.CV_8UC1);

            // Convert the grayscale visualization to color so it can be drawn on the frame
            Imgproc.cvtColor(dtVis, dtVis, Imgproc.COLOR_GRAY2RGB);

            // Copy the visualization to the output frame.
            dtVis.copyTo(frame);
            dtVis.release();
        }
        // --- END DEBUGGING VISUALIZATION ---

//        morphedThresholdMat.copyTo(frame);

        if (!detectedSamples.isEmpty()) {
            for (AnalyzedSample sample : detectedSamples) {
                estimateOptimalPoseForSample(sample);
            }
        }

        telemetry.update();
        return detectedSamples;
    }

    void findContours(Mat input, ArrayList<AnalyzedSample> outputList){
        // Convert the input image to YCrCb color space
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

        Imgproc.distanceTransform(morphedThresholdMat, distTransformMat, Imgproc.DIST_L2, Imgproc.DIST_MASK_5);
        ArrayList<MatOfPoint> contoursList = new ArrayList<>();
        Imgproc.findContours(morphedThresholdMat, contoursList, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

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

        // Find the highest point
        Point topPoint2D = null;
        double minY = Double.MAX_VALUE;
        for (Point p : approximatedPointsList) {
            if (p.y < minY) {
                minY = p.y;
                topPoint2D = p;
            }
        }
        if (topPoint2D == null) return;
        Point3 worldTopPoint = getWorldPosition(topPoint2D); // Convert to 3D world coordinates

        // Calculate the worldCentroid of the contour
        Moments moments = Imgproc.moments(contour);
        Point centroid2D = new Point(moments.m10 / moments.m00, moments.m01 / moments.m00);
        Point3 worldCentroid = getWorldPosition(centroid2D);

        // Fit rotated rectangle
        RotatedRect distortedRotatedRect = Imgproc.minAreaRect(originalContour2f);

        telemetry.addData("Approximated Points Count", approximatedPointsList.size()); // Telemetry for approximated points

        AnalyzedSample sample = new AnalyzedSample();
        sample.angle = 0; // Will be set by optimizer
        sample.color = color;
        sample.rotatedRect = distortedRotatedRect;
        sample.worldCentroid = worldCentroid;
        sample.approxContourPoints = approximatedPointsList;

        sample.topPoint2D = topPoint2D;
        sample.worldTopPoint = worldTopPoint;
        sample.optimalWorldOrigin = null; // Will be set by optimizer
        sample.projectedOptimalPoints = new ArrayList<>(); // Will be set by optimizer
        outputList.add(sample);

        originalContour2f.release();
        approxContour2f.release();
        approxContour.release();
    }

//    private double calculateWorldAngle(Point3[] corners) {
//        // Calculate the squared length of two adjacent sides (to avoid sqrt)
//        double lenSq01 = Math.pow(corners[1].x - corners[0].x, 2) + Math.pow(corners[1].y - corners[0].y, 2);
//        double lenSq12 = Math.pow(corners[2].x - corners[1].x, 2) + Math.pow(corners[2].y - corners[1].y, 2);
//
//        Point3 p1, p2;
//        if (lenSq01 >= lenSq12) {
//            p1 = corners[0];
//            p2 = corners[1];
//        } else {
//            p1 = corners[1];
//            p2 = corners[2];
//        }
//
//        // Calculate the angle of the longer side
//        double deltaX = p2.x - p1.x;
//        double deltaY = p2.y - p1.y;
//        double angleRadians = Math.atan2(deltaY, deltaX);
//        double angleDegrees = Math.toDegrees(angleRadians);
//
//        // Normalize to 0-180
//        if (angleDegrees < 0) {
//            angleDegrees += 360;
//        }
//        if (angleDegrees > 180) {
//            angleDegrees -= 180;
//        }
//
//        return angleDegrees;
//    }

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

        return new Point3(
                rayOriginWorld.x + rayDirectionWorld.x * t,
                rayOriginWorld.y + rayDirectionWorld.y * t,
                0
        );
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

    /**
     * Projects the 3D model points onto the 2D image plane using the sample's
     * calculated optimal world origin and yaw angle, and stores the result
     * in the sample's `projectedOptimalPoints` field.
     *
     * @param sample The AnalyzedSample object containing the optimal world origin and yaw.
     */
    private void projectOptimalModelPointsForSample(AnalyzedSample sample) {
        Mat rvec_optimal_object_world = new Mat(3, 1, CvType.CV_64FC1);
        Mat tvec_optimal_object_world = new Mat(3, 1, CvType.CV_64FC1);

        // Use the sample's stored world origin
        tvec_optimal_object_world.put(0, 0, sample.optimalWorldOrigin.x);
        tvec_optimal_object_world.put(1, 0, sample.optimalWorldOrigin.y);
        tvec_optimal_object_world.put(2, 0, 0); // Z-world is 0 for objects on the ground

        // Use the sample's stored optimal yaw angle
        double yawRadians = Math.toRadians(sample.angle);
        Mat R_z_optimal = new Mat(3, 3, CvType.CV_64FC1);
        R_z_optimal.put(0, 0, Math.cos(yawRadians)); R_z_optimal.put(0, 1, -Math.sin(yawRadians)); R_z_optimal.put(0, 2, 0);
        R_z_optimal.put(1, 0, Math.sin(yawRadians)); R_z_optimal.put(1, 1, Math.cos(yawRadians));  R_z_optimal.put(1, 2, 0);
        R_z_optimal.put(2, 0, 0);                   R_z_optimal.put(2, 1, 0);                   R_z_optimal.put(2, 2, 1);
        Calib3d.Rodrigues(R_z_optimal, rvec_optimal_object_world);
        R_z_optimal.release();

        List<Point> projectedPoints = projectModelPoints(rvec_optimal_object_world, tvec_optimal_object_world);

        rvec_optimal_object_world.release();
        tvec_optimal_object_world.release();

        sample.projectedOptimalPoints = projectedPoints; // Store the result directly in the sample
    }

    private double calculatePoseCost(Point3 worldTopPointObserved, double yawDegrees, Hypothesis hypothesis) {
        List<Point3> modelPoints = getTopFaceModelPoints();
        Point3 model_corner_B = modelPoints.get(1);

        Mat rvec_object_world = new Mat(3, 1, CvType.CV_64FC1);
        Mat tvec_object_world = new Mat(3, 1, CvType.CV_64FC1);

        // 1. Create the 3x3 rotation matrix (R_z) and its rvec equivalent
        double yawRadians = Math.toRadians(yawDegrees);
        Mat R_z = new Mat(3, 3, CvType.CV_64FC1);
        R_z.put(0, 0, Math.cos(yawRadians)); R_z.put(0, 1, -Math.sin(yawRadians)); R_z.put(0, 2, 0);
        R_z.put(1, 0, Math.sin(yawRadians)); R_z.put(1, 1, Math.cos(yawRadians));  R_z.put(1, 2, 0);
        R_z.put(2, 0, 0);                   R_z.put(2, 1, 0);                   R_z.put(2, 2, 1);
        Calib3d.Rodrigues(R_z, rvec_object_world);

        // 2. Calculate the model's world origin based on the hypothesis
        Point3 current_model_origin_world;
        if (hypothesis == Hypothesis.HYPOTHESIS_A) {
            current_model_origin_world = worldTopPointObserved;
        } else { // HYPOTHESIS_B
            Mat vec_origin_to_corner_B_local = new Mat(3, 1, CvType.CV_64FC1);
            vec_origin_to_corner_B_local.put(0, 0, model_corner_B.x);
            vec_origin_to_corner_B_local.put(1, 0, model_corner_B.y);
            vec_origin_to_corner_B_local.put(2, 0, model_corner_B.z);

            Mat rotated_vec_origin_to_corner_B_world = new Mat(3, 1, CvType.CV_64FC1);
            // CORRECTED: Use the 3x3 rotation matrix R_z for rotation
            Core.gemm(R_z, vec_origin_to_corner_B_local, 1.0, new Mat(), 0.0, rotated_vec_origin_to_corner_B_world, 0);

            current_model_origin_world = new Point3(
                    worldTopPointObserved.x - rotated_vec_origin_to_corner_B_world.get(0, 0)[0],
                    worldTopPointObserved.y - rotated_vec_origin_to_corner_B_world.get(1, 0)[0],
                    0
            );

            vec_origin_to_corner_B_local.release();
            rotated_vec_origin_to_corner_B_world.release();
        }
        tvec_object_world.put(0, 0, current_model_origin_world.x);
        tvec_object_world.put(1, 0, current_model_origin_world.y);
        tvec_object_world.put(2, 0, 0);

        // 3. CORRECTED: Project points using the public helper method
        List<Point> projectedPoints = projectModelPoints(rvec_object_world, tvec_object_world);

        // 4. Clean up Mats created in this function
        R_z.release();
        rvec_object_world.release();
        tvec_object_world.release();

        // --- 5. Calculate the cost using the Distance Transform map ---
        double totalDtValue = 0;
        int points_in_bounds = 0;
        if (distTransformMat.empty() || distTransformMat.type() != CvType.CV_32FC1) {
            telemetry.addData("Error", "Distance Transform Mat is not ready for cost calculation!");
            return Double.MAX_VALUE;
        }

        int dt_cols = distTransformMat.cols();
        int dt_rows = distTransformMat.rows();

        for (Point p : projectedPoints) {
            int px = (int) Math.round(p.x);
            int py = (int) Math.round(p.y);
            if (px >= 0 && px < dt_cols && py >= 0 && py < dt_rows) {
                totalDtValue += distTransformMat.get(py, px)[0];
                points_in_bounds++;
            }
        }

        double cost;
        if (points_in_bounds > 0) {
            cost = -totalDtValue / points_in_bounds; // Average negative value
        } else {
            cost = Double.MAX_VALUE / 2; // High penalty if points are off-screen
        }

        return cost;
    }

    /**
     * Attempts to find the optimal yaw angle of the object by iteratively
     * minimizing the cost function, given the observed worldTopPoint and a hypothesis.
     * Performs a 180-degree grid search.
     *
     * @param worldTopPointObserved The 3D world coordinates of the observed top point.
     * @param hypothesis The hypothesis being evaluated (HYPOTHESIS_A or HYPOTHESIS_B).
     * @return The optimized Yaw angle (degrees) in the 0-360 range.
     */
    private double findOptimalYaw(Point3 worldTopPointObserved, Hypothesis hypothesis) {
        double bestYaw = 0;
        double minCost = calculatePoseCost(worldTopPointObserved, 0, hypothesis);

        // --- START DEBUGGING ---
        // Log the initial cost for the current hypothesis
        telemetry.addData(String.format("YawOpt %s Start", hypothesis.name()), String.format("Yaw: 0.0, Cost: %.4f", minCost));
        // --- END DEBUGGING ---

        double searchStepDegrees = 5.0; // Use a slightly larger step for debugging to reduce log spam

        for (double currentYaw = searchStepDegrees; currentYaw < 180; currentYaw += searchStepDegrees) {
            double cost = calculatePoseCost(worldTopPointObserved, currentYaw, hypothesis);

            // --- START DEBUGGING ---
            // Log the cost for every tested yaw angle
            telemetry.addData(String.format("YawOpt %s Test", hypothesis.name()), String.format("Yaw: %.1f, Cost: %.4f", currentYaw, cost));
            // --- END DEBUGGING ---

            if (cost < minCost) {
                minCost = cost;
                bestYaw = currentYaw;
            }
        }

        // --- START DEBUGGING ---
        // Log the final chosen yaw and cost
        telemetry.addData(String.format("YawOpt %s Final", hypothesis.name()), String.format("BestYaw: %.1f, MinCost: %.4f", bestYaw, minCost));
        // --- END DEBUGGING ---

        // The rest of the method is removed for brevity, but this now replaces the previous telemetry line.
        return bestYaw;
    }

    /**
     * Estimates the optimal 3D pose (origin and yaw) for a detected sample
     * by testing two hypotheses for the `worldTopPoint`'s correspondence.
     *
     * @param sample The AnalyzedSample object to update with the optimal pose.
     * @return The updated AnalyzedSample with optimal pose information.
     */
    private AnalyzedSample estimateOptimalPoseForSample(AnalyzedSample sample) {
        if (sample.worldTopPoint == null || sample.topPoint2D == null) {
            telemetry.addData("Error", "Sample worldTopPoint or topPoint2D is null for pose estimation.");
            return sample;
        }

        double bestOverallYaw = 0;
        Point3 final_optimalWorldOrigin = null; // This will store the ultimate (0,0,0) model origin
        double minOverallCost = Double.MAX_VALUE;

        // --- Hypothesis 1: worldTopPoint corresponds to model_corner_A ((0,0,0) local) ---
        // findOptimalYaw will call calculatePoseCost, which handles the origin derivation
        double H1_optimized_yaw = findOptimalYaw(sample.worldTopPoint, Hypothesis.HYPOTHESIS_A);
        // Calculate cost one last time with optimal yaw for proper comparison
        double H1_cost = calculatePoseCost(sample.worldTopPoint, H1_optimized_yaw, Hypothesis.HYPOTHESIS_A);
        telemetry.addData("H1 Cost/Yaw", String.format("%.2f / %.2f", H1_cost, H1_optimized_yaw));

        if (H1_cost < minOverallCost) {
            minOverallCost = H1_cost;
            bestOverallYaw = H1_optimized_yaw;
            // If H1 wins, the model's (0,0,0) origin is directly at worldTopPoint
            final_optimalWorldOrigin = sample.worldTopPoint;
        }

        // --- Hypothesis 2: worldTopPoint corresponds to model_corner_B ((BLOCK_LENGTH, 0, 0) local) ---
        // findOptimalYaw will call calculatePoseCost, which handles the origin derivation for H2
        double H2_optimized_yaw = findOptimalYaw(sample.worldTopPoint, Hypothesis.HYPOTHESIS_B);
        // Calculate cost one last time with optimal yaw for proper comparison
        double H2_cost = calculatePoseCost(sample.worldTopPoint, H2_optimized_yaw, Hypothesis.HYPOTHESIS_B);
        telemetry.addData("H2 Cost/Yaw", String.format("%.2f / %.2f", H2_cost, H2_optimized_yaw));

        if (H2_cost < minOverallCost) {
            minOverallCost = H2_cost;
            bestOverallYaw = H2_optimized_yaw;
            // If H2 wins, the model's (0,0,0) origin needs to be calculated based on the H2_optimized_yaw
            // This calculation is the same as performed inside calculatePoseCost for Hypothesis B.
            // In the "if (H2_cost < minOverallCost)" block
            List<Point3> modelPoints = getTopFaceModelPoints();
            Point3 model_corner_B = modelPoints.get(1);

            // CORRECTED: Create a 3x3 rotation matrix from the optimal yaw
            double yawRad_H2 = Math.toRadians(H2_optimized_yaw);
            Mat R_H2_optimal_mat = new Mat(3, 3, CvType.CV_64FC1);
            R_H2_optimal_mat.put(0, 0, Math.cos(yawRad_H2)); R_H2_optimal_mat.put(0, 1, -Math.sin(yawRad_H2)); R_H2_optimal_mat.put(0, 2, 0);
            R_H2_optimal_mat.put(1, 0, Math.sin(yawRad_H2)); R_H2_optimal_mat.put(1, 1, Math.cos(yawRad_H2));  R_H2_optimal_mat.put(1, 2, 0);
            R_H2_optimal_mat.put(2, 0, 0);                   R_H2_optimal_mat.put(2, 1, 0);                   R_H2_optimal_mat.put(2, 2, 1);

            Mat vec_origin_to_corner_B_local = new Mat(3, 1, CvType.CV_64FC1);
            vec_origin_to_corner_B_local.put(0, 0, model_corner_B.x);
            vec_origin_to_corner_B_local.put(1, 0, model_corner_B.y);
            vec_origin_to_corner_B_local.put(2, 0, model_corner_B.z);

            Mat rotated_vec_origin_to_corner_B_world = new Mat(3, 1, CvType.CV_64FC1);
            // CORRECTED: Use the 3x3 matrix for rotation
            Core.gemm(R_H2_optimal_mat, vec_origin_to_corner_B_local, 1.0, new Mat(), 0.0, rotated_vec_origin_to_corner_B_world, 0);

            final_optimalWorldOrigin = new Point3(
                    sample.worldTopPoint.x - rotated_vec_origin_to_corner_B_world.get(0, 0)[0],
                    sample.worldTopPoint.y - rotated_vec_origin_to_corner_B_world.get(1, 0)[0],
                    0 // Z remains 0
            );

            // Clean up all the Mats
            R_H2_optimal_mat.release();
            vec_origin_to_corner_B_local.release();
            rotated_vec_origin_to_corner_B_world.release();
        }

        // Update the sample with the best found pose
        if (final_optimalWorldOrigin != null) {
            sample.angle = bestOverallYaw;
            sample.optimalWorldOrigin = final_optimalWorldOrigin; // Store the optimal model origin

            telemetry.addData("Optimal Model Origin (X,Y)", String.format("%.2f, %.2f", final_optimalWorldOrigin.x, final_optimalWorldOrigin.y));
            telemetry.addData("Optimal Yaw", String.format("%.2f degrees", bestOverallYaw));
            telemetry.addData("Min Overall Cost", String.format("%.2f", minOverallCost));
        } else {
            telemetry.addData("Error", "Final optimal model origin is null after optimization.");
            sample.optimalWorldOrigin = null; // Ensure null if no pose found
        }

        // --- Final Projection: Project the model points using the determined optimal pose ---
        projectOptimalModelPointsForSample(sample);

        return sample;
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
//            RotatedRect distortedRect = sample.rotatedRect;
//            Point[] distortedPoints = new Point[4];
//            distortedRect.points(distortedPoints);
//
//            for (int i = 0; i < 4; ++i) {
//                canvas.drawLine(
//                        (float) (distortedPoints[i].x * scaleBmpPxToCanvasPx),
//                        (float) (distortedPoints[i].y * scaleBmpPxToCanvasPx),
//                        (float) (distortedPoints[(i + 1) % 4].x * scaleBmpPxToCanvasPx),
//                        (float) (distortedPoints[(i + 1) % 4].y * scaleBmpPxToCanvasPx),
//                        distortedPaint
//                );
//            }
//
//            canvas.drawCircle(
//                    (float) (sample.worldCentroid.x * scaleBmpPxToCanvasPx),
//                    (float) (sample.worldCentroid.y * scaleBmpPxToCanvasPx),
//                    5 * scaleCanvasDensity,
//                    centerPaint
//            );

            for (Point p : sample.approxContourPoints) {
                canvas.drawCircle(
                        (float) (p.x * scaleBmpPxToCanvasPx),
                        (float) (p.y * scaleBmpPxToCanvasPx),
                        3 * scaleCanvasDensity,
                        approxPointPaint
                );
            }

            // Draw the final, optimal pose as a cyan rectangle
            if (sample.projectedOptimalPoints != null && !sample.projectedOptimalPoints.isEmpty()) {
                Paint optimalPosePaint = new Paint();
                optimalPosePaint.setColor(Color.CYAN);
                optimalPosePaint.setStrokeWidth(scaleCanvasDensity * 4); // Make it thick
                optimalPosePaint.setStyle(Paint.Style.STROKE);

                for (int i = 0; i < sample.projectedOptimalPoints.size(); i++) {
                    Point p1 = sample.projectedOptimalPoints.get(i);
                    // Connect to the next point, wrapping around from the last to the first
                    Point p2 = sample.projectedOptimalPoints.get((i + 1) % sample.projectedOptimalPoints.size());

                    if (p1 == null || p2 == null) continue; // Safety check

                    canvas.drawLine(
                            (float) (p1.x * scaleBmpPxToCanvasPx), (float) (p1.y * scaleBmpPxToCanvasPx),
                            (float) (p2.x * scaleBmpPxToCanvasPx), (float) (p2.y * scaleBmpPxToCanvasPx),
                            optimalPosePaint
                    );
                }
            }

        }
    }

}
