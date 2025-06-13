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
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SampleDetectionPipeline implements VisionProcessor {

    public enum SelectedColor { RED, YELLOW, BLUE }
    SelectedColor selectedColor = SelectedColor.RED; // Default
//    SelectedColor selectedColor = SelectedColor.YELLOW; // Default
//    SelectedColor selectedColor = SelectedColor.BLUE; // Default


    private enum Hypothesis {
        HYPOTHESIS_A, // Anchor is corner (0, 0, 0)
        HYPOTHESIS_B, // Anchor is corner (L, 0, 0)
        HYPOTHESIS_C, // Anchor is corner (L, W, 0)
        HYPOTHESIS_D  // Anchor is corner (0, W, 0)
    }

    static class DetectedSample {
        Point3 position; // Origin corner
        Point3 worldCenter;
        double yaw;
        List<Point> projectedCorners;
        Point projectedCenter;
        SelectedColor color;
    }

    // Image buffers
    Mat ycrcbMat = new Mat();
    Mat thresholdMat = new Mat();
    Mat morphedThresholdMat = new Mat();
    Mat maskedColorMat = new Mat();
    Mat cannyMat = new Mat();

    Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));

    // world-to-camera transformation matrix
    private Mat R_world_to_cam_inst;
    private Mat tvec_world_to_cam_inst;

    // Camera parameters
    public double cameraHeight = 25; // Height above the ground plane in CM
    public double cameraDownAngle = 44; // Angle in degrees the camera points downwards (90 degrees is straight down)
    Mat cameraMatrix = new Mat(3, 3, CvType.CV_64FC1);
    Mat distCoeffs = new Mat(1, 5, CvType.CV_64FC1);
    private final int CALIBRATION_WIDTH = 2560;
    private final int CALIBRATION_HEIGHT = 1440;
    private final int FRAME_WIDTH = 640; // Preferred
    private int frameHeight = 480; // Preferred

    // Dimensions
    private final double BLOCK_WIDTH = 3.7; // In CM
    private final double BLOCK_LENGTH = 8.7; // In CM
    public double approxPolyEpsilon = 0.005; // Used for approximating shape

    // Thresholds
    public Scalar redLower = new Scalar(0, 175, 90); // Can change second and third
    public Scalar redUpper = new Scalar(255, 255, 255);
    public Scalar yellowLower = new Scalar(0, 145, 0); // Can change second
    public Scalar yellowUpper = new Scalar(255, 255, 85); // Can change third
    public Scalar blueLower = new Scalar(0, 80, 150); // Can change third
    public Scalar blueUpper = new Scalar(255, 255, 255);

    // Contour filtering
    public double MIN_AREA = 1000;
    public double MAX_AREA = 15000;
    public double MAX_ASPECT_RATIO = 6;

    Telemetry telemetry = null;

    public SampleDetectionPipeline(Telemetry telemetry) {
        this.telemetry = telemetry;
    }

    @Override
    public void init(int width, int height, CameraCalibration calibration) {
        double aspectRatio = (double) width / height;
        frameHeight = (int) (FRAME_WIDTH / aspectRatio);
        loadCameraCalibration();

        // Pre-calculate the constant world-to-camera transformation
        Mat rvec_cam_to_world = new Mat(3, 1, CvType.CV_64FC1);
        Mat tvec_cam_to_world = new Mat(3, 1, CvType.CV_64FC1);
        getStaticCameraPoseInWorld(rvec_cam_to_world, tvec_cam_to_world);

        Mat R_cam_to_world = new Mat(3, 3, CvType.CV_64FC1);
        Calib3d.Rodrigues(rvec_cam_to_world, R_cam_to_world);

        this.R_world_to_cam_inst = R_cam_to_world.t();
        this.tvec_world_to_cam_inst = new Mat(3, 1, CvType.CV_64FC1);
        Core.gemm(this.R_world_to_cam_inst, tvec_cam_to_world, -1.0, new Mat(), 0.0, this.tvec_world_to_cam_inst, 0);

        rvec_cam_to_world.release();
        tvec_cam_to_world.release();
        R_cam_to_world.release();
    }

    @Override
    public Object processFrame(Mat frame, long captureTimeNanos) {
        Imgproc.resize(frame, frame, new Size(FRAME_WIDTH, frameHeight));

        ArrayList<MatOfPoint> contours = findContours(frame);
        ArrayList<DetectedSample> detectedSamples = new ArrayList<>();

        for (MatOfPoint contour : contours) {
            Point topPoint = findHighestPoint(contour);
            if (topPoint == null) continue;

            DetectedSample sample = estimatePose(topPoint, selectedColor);
            if (sample == null) continue;

            detectedSamples.add(sample);
            telemetry.addData("Block Detected",
                    "X: %.1f cm, Y: %.1f cm, Yaw: %.1f deg",
                    sample.worldCenter.x, sample.worldCenter.y, sample.yaw);

            contour.release();
        }

//        Core.multiply(frame, new Scalar(0.5, 0.5, 0.5), frame);
//        Core.addWeighted(frame, 0.7, maskedColorMat, 0.6, 0, frame);
//        cannyMat.copyTo(frame);

        telemetry.update();
        return detectedSamples;
    }

    private ArrayList<MatOfPoint> findContours(Mat input){
        // Convert the input image to YCrCb color space
        Imgproc.cvtColor(input, ycrcbMat, Imgproc.COLOR_RGB2YCrCb);

        switch (selectedColor) {
            case RED: Core.inRange(ycrcbMat, redLower, redUpper, thresholdMat); break;
            case BLUE: Core.inRange(ycrcbMat, blueLower, blueUpper, thresholdMat); break;
            case YELLOW: Core.inRange(ycrcbMat, yellowLower, yellowUpper, thresholdMat); break;
        }

        // Apply morphology for noise reduction
        Imgproc.morphologyEx(thresholdMat, morphedThresholdMat, Imgproc.MORPH_CLOSE, kernel, new Point(-1, -1), 2);
        Imgproc.morphologyEx(morphedThresholdMat, morphedThresholdMat, Imgproc.MORPH_OPEN, kernel, new Point(-1, -1), 1);

        // Edge detection
        maskedColorMat.release();
        Core.bitwise_and(input, input, maskedColorMat, morphedThresholdMat);
        Imgproc.Canny(maskedColorMat, cannyMat, 50, 150, 3, true);
        Mat canyKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.dilate(cannyMat, cannyMat, canyKernel, new Point(-1, -1), 1);

        // Find contours
        ArrayList<MatOfPoint> contoursList = new ArrayList<>();
        Imgproc.findContours(morphedThresholdMat, contoursList, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // Filter contours
        contoursList.removeIf(contour -> {
            double area = Imgproc.contourArea(contour);
            if (area < MIN_AREA || area > MAX_AREA) {
                return true;
            }

            org.opencv.core.Rect boundingBox = Imgproc.boundingRect(contour);
            double width = boundingBox.width;
            double height = boundingBox.height;
            if (width == 0 || height == 0) return true; // Avoid division by zero

            double aspectRatio = width / height;
            return aspectRatio > MAX_ASPECT_RATIO || (1.0 / aspectRatio) > MAX_ASPECT_RATIO;
        });

        return contoursList;
    }

    private Point findHighestPoint(MatOfPoint contour) {
        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
        double arcLength = Imgproc.arcLength(contour2f, true);
        MatOfPoint2f approxContour2f = new MatOfPoint2f();
        Imgproc.approxPolyDP(contour2f, approxContour2f, arcLength * approxPolyEpsilon, true);

        List<Point> approxPoints = approxContour2f.toList();
        contour2f.release();
        approxContour2f.release();

        if (approxPoints.isEmpty()) return null;

        Point topPoint = approxPoints.get(0);
        for (Point p : approxPoints) {
            if (p.y < topPoint.y) {
                topPoint = p;
            }
        }
        return topPoint;
    }

    private DetectedSample estimatePose(Point topPoint2D, SelectedColor color) {
        Point3 worldAnchorPoint = getWorldPosition(topPoint2D);

        double bestOverallCost = Double.MAX_VALUE;
        Hypothesis bestHypothesis = null;
        double bestOverallYaw = 0;

        // Test all four possibilities for which corner the anchor point represents.
        for (Hypothesis hypothesis : Hypothesis.values()) {
            double yaw = findOptimalYaw(worldAnchorPoint, hypothesis);
            double cost = calculatePoseCost(worldAnchorPoint, yaw, hypothesis);

            if (cost < bestOverallCost) {
                bestOverallCost = cost;
                bestHypothesis = hypothesis;
                bestOverallYaw = yaw;
            }
        }

        if (bestHypothesis == null) return null;

        // --- Construct the Final Winning Pose ---
        DetectedSample block = new DetectedSample();
        block.yaw = bestOverallYaw;
        block.color = color;

        // Calculate the final origin based on the winning hypothesis and yaw
        List<Point3> modelPoints = getTopFaceModelPoints();
        Point3 winningCorner = modelPoints.get(bestHypothesis.ordinal());

        if (bestHypothesis == Hypothesis.HYPOTHESIS_A) {
            block.position = worldAnchorPoint;
        } else {
            Mat rvec = new Mat(3, 1, CvType.CV_64FC1);
            rvec.put(0,0,0); rvec.put(1,0,0); rvec.put(2,0, Math.toRadians(bestOverallYaw));
            Mat R_z = new Mat();
            Calib3d.Rodrigues(rvec, R_z);

            Mat vec_to_corner_local = new Mat(3, 1, CvType.CV_64FC1);
            vec_to_corner_local.put(0, 0, winningCorner.x);
            vec_to_corner_local.put(1, 0, winningCorner.y);
            vec_to_corner_local.put(2, 0, winningCorner.z);

            Mat rotated_vec = new Mat(3, 1, CvType.CV_64FC1);
            Core.gemm(R_z, vec_to_corner_local, 1.0, new Mat(), 0.0, rotated_vec, 0);

            block.position = new Point3(
                    worldAnchorPoint.x - rotated_vec.get(0, 0)[0],
                    worldAnchorPoint.y - rotated_vec.get(1, 0)[0],
                    0);

            rvec.release(); R_z.release(); vec_to_corner_local.release(); rotated_vec.release();
        }

        // Calculate the world coordinates of the block's center
        if (block.position != null) {
            Point3 localCenter = new Point3(BLOCK_LENGTH / 2.0, BLOCK_WIDTH / 2.0, 0);
            Mat rvec = createRvec(block.yaw);
            Mat R_z = new Mat();
            Calib3d.Rodrigues(rvec, R_z);
            Mat localCenterVec = new Mat(3, 1, CvType.CV_64FC1);
            localCenterVec.put(0, 0, localCenter.x, localCenter.y, localCenter.z);
            Mat rotatedCenterVec = new Mat();
            Core.gemm(R_z, localCenterVec, 1.0, new Mat(), 0.0, rotatedCenterVec, 0);
            block.worldCenter = new Point3(
                    block.position.x + rotatedCenterVec.get(0, 0)[0],
                    block.position.y + rotatedCenterVec.get(1, 0)[0],
                    0
            );
            rvec.release(); R_z.release(); localCenterVec.release(); rotatedCenterVec.release();
        }

        projectPoseFeatures(block);
        return block;
    }

    /**
     * Attempts to find the optimal yaw angle of the object by iteratively
     * minimizing the cost function, given the topPoint and a hypothesis.
     */
    private double findOptimalYaw(Point3 worldAnchorPoint, Hypothesis hypothesis) {
        double bestYaw = 0;
        double minCost = Double.MAX_VALUE;

        // Coarse Search
        double coarseBestYaw = 0;
        double coarseMinCost = calculatePoseCost(worldAnchorPoint, 0, hypothesis);

        for (double currentYaw = 10; currentYaw < 180; currentYaw += 10) {
            double cost = calculatePoseCost(worldAnchorPoint, currentYaw, hypothesis);
            if (cost < coarseMinCost) {
                coarseMinCost = cost;
                coarseBestYaw = currentYaw;
            }
        }

        // Fine Search
        double searchRadius = 15;
        double fineStep = 2;
        minCost = coarseMinCost;
        bestYaw = coarseBestYaw;

        for (double currentYaw = coarseBestYaw - searchRadius; currentYaw <= coarseBestYaw + searchRadius; currentYaw += fineStep) {
            double normalizedYaw = currentYaw;
            if (normalizedYaw < 0) normalizedYaw += 180;
            if (normalizedYaw >= 180) normalizedYaw -= 180;

            double cost = calculatePoseCost(worldAnchorPoint, normalizedYaw, hypothesis);
            if (cost < minCost) {
                minCost = cost;
                bestYaw = normalizedYaw;
            }
        }

        return bestYaw;
    }

    private double calculatePoseCost(Point3 worldAnchorPoint, double yawDegrees, Hypothesis hypothesis) {
        // Construct Pose from given hyphothesis
        List<Point3> modelPoints = getTopFaceModelPoints();
        Point3 origin;
        Mat rvec = new Mat(3, 1, CvType.CV_64FC1);
        rvec.put(0,0,0); rvec.put(1,0,0); rvec.put(2,0, Math.toRadians(yawDegrees));
        Mat R_z = new Mat();
        Calib3d.Rodrigues(rvec, R_z);

        if (hypothesis == Hypothesis.HYPOTHESIS_A) {
            origin = worldAnchorPoint;
        } else {
            Point3 modelCorner = modelPoints.get(hypothesis.ordinal());
            Mat vec_to_corner_local = new Mat(3, 1, CvType.CV_64FC1);
            vec_to_corner_local.put(0, 0, modelCorner.x);
            vec_to_corner_local.put(1, 0, modelCorner.y);
            vec_to_corner_local.put(2, 0, modelCorner.z);
            Mat rotated_vec = new Mat(3, 1, CvType.CV_64FC1);
            Core.gemm(R_z, vec_to_corner_local, 1.0, new Mat(), 0.0, rotated_vec, 0);
            origin = new Point3(
                    worldAnchorPoint.x - rotated_vec.get(0, 0)[0],
                    worldAnchorPoint.y - rotated_vec.get(1, 0)[0],
                    0);
            vec_to_corner_local.release();
            rotated_vec.release();
        }
        Mat tvec = new Mat(3,1,CvType.CV_64FC1);
        tvec.put(0,0,origin.x); tvec.put(1,0,origin.y); tvec.put(2,0,origin.z);

        // Project rectangle and compare against canny edge
        List<Point> projectedPoints = projectWorldPoints(rvec, tvec, getTopFaceModelPoints());
        if (projectedPoints.size() < 4) {
            rvec.release(); R_z.release(); tvec.release();
            return Double.MAX_VALUE;
        }

        Mat modelEdgeMask = Mat.zeros(morphedThresholdMat.size(), CvType.CV_8UC1);
        MatOfPoint projectedContour = new MatOfPoint(projectedPoints.toArray(new Point[0]));
        Imgproc.polylines(modelEdgeMask, Arrays.asList(projectedContour), true, new Scalar(255), 3);

        Mat edgeIntersection = new Mat();
        Core.bitwise_and(this.cannyMat, modelEdgeMask, edgeIntersection);

        double edgeIntersectionArea = Core.countNonZero(edgeIntersection);
        double modelEdgeArea = Core.countNonZero(modelEdgeMask);
        double edgeScore = (modelEdgeArea > 0) ? edgeIntersectionArea / modelEdgeArea : 0;

        double cost = -edgeScore;

        rvec.release(); R_z.release(); tvec.release();
        projectedContour.release(); modelEdgeMask.release(); edgeIntersection.release();

        return cost;
    }

    // Projects 3D world points to 2D image points.
    private List<Point> projectWorldPoints(Mat objectWorldRvec, Mat objectWorldTvec, List<Point3> pointsToProject) {
        // Transform Object's World Pose to be Camera-Relative
        Mat R_obj_world = new Mat(3, 3, CvType.CV_64FC1);
        Calib3d.Rodrigues(objectWorldRvec, R_obj_world);
        Mat R_obj_cam_mat = new Mat(3, 3, CvType.CV_64FC1);
        Core.gemm(this.R_world_to_cam_inst, R_obj_world, 1.0, new Mat(), 0.0, R_obj_cam_mat, 0);
        Mat rvec_obj_cam = new Mat(3, 1, CvType.CV_64FC1);
        Calib3d.Rodrigues(R_obj_cam_mat, rvec_obj_cam);

        Mat tvec_obj_cam = new Mat(3, 1, CvType.CV_64FC1);
        Mat temp_t = new Mat(3, 1, CvType.CV_64FC1);
        Core.gemm(this.R_world_to_cam_inst, objectWorldTvec, 1.0, new Mat(), 0.0, temp_t, 0);
        Core.add(temp_t, this.tvec_world_to_cam_inst, tvec_obj_cam);

        // Project points
        MatOfPoint3f pointsToProjectMat = new MatOfPoint3f(pointsToProject.toArray(new Point3[0]));
        MatOfPoint2f projectedPoints = new MatOfPoint2f();
        Calib3d.projectPoints(pointsToProjectMat, rvec_obj_cam, tvec_obj_cam, cameraMatrix, new MatOfDouble(distCoeffs), projectedPoints);

        List<Point> resultList = projectedPoints.toList();

        R_obj_world.release(); R_obj_cam_mat.release(); rvec_obj_cam.release();
        tvec_obj_cam.release(); temp_t.release();
        pointsToProjectMat.release(); projectedPoints.release();

        return resultList;
    }

    private void projectPoseFeatures(DetectedSample block) {
        if (block.position == null) return;
        Mat rvec = createRvec(block.yaw);
        Mat tvec = createTvec(block.position);

        // Project the 4 corners using the helper
        block.projectedCorners = projectWorldPoints(rvec, tvec, getTopFaceModelPoints());

        // Project the center point using the same helper
        Point3 localCenter = new Point3(BLOCK_LENGTH / 2.0, BLOCK_WIDTH / 2.0, 0);
        List<Point> projectedCenterList = projectWorldPoints(rvec, tvec, Arrays.asList(localCenter));
        if (projectedCenterList != null && !projectedCenterList.isEmpty()) {
            block.projectedCenter = projectedCenterList.get(0);
        }

        rvec.release();
        tvec.release();
    }

    // Helper to create a rotation vector from a yaw angle in degrees
    private Mat createRvec(double yawDegrees) {
        Mat rvec = new Mat(3, 1, CvType.CV_64FC1);
        rvec.put(0, 0, 0);
        rvec.put(1, 0, 0);
        rvec.put(2, 0, Math.toRadians(yawDegrees));
        return rvec;
    }

    // Helper to create a translation vector from a 3D world point
    private Mat createTvec(Point3 worldOrigin) {
        Mat tvec = new Mat(3, 1, CvType.CV_64FC1);
        tvec.put(0, 0, worldOrigin.x);
        tvec.put(1, 0, worldOrigin.y);
        tvec.put(2, 0, worldOrigin.z);
        return tvec;
    }

    @SuppressWarnings("unchecked")
    @Override
    public void onDrawFrame(Canvas canvas, int onscreenWidth, int onscreenHeight, float scaleBmpPxToCanvasPx, float scaleCanvasDensity, Object userContext) {
        if (!(userContext instanceof ArrayList)) return;
        ArrayList<DetectedSample> samplesToDraw = (ArrayList<DetectedSample>) userContext;

        Paint paint = new Paint();
        paint.setColor(Color.GREEN);
        paint.setStrokeWidth(scaleCanvasDensity * 4);
        paint.setStyle(Paint.Style.STROKE);

        Paint centerPaint = new Paint();
        centerPaint.setColor(Color.BLACK);
        centerPaint.setStyle(Paint.Style.FILL);

        for (DetectedSample sample : samplesToDraw) {
            if (sample.projectedCorners != null && !sample.projectedCorners.isEmpty()) {
                for (int i = 0; i < sample.projectedCorners.size(); i++) {
                    Point p1 = sample.projectedCorners.get(i);
                    Point p2 = sample.projectedCorners.get((i + 1) % sample.projectedCorners.size());
                    if (p1 == null || p2 == null) continue;
                    canvas.drawLine(
                            (float) (p1.x * scaleBmpPxToCanvasPx), (float) (p1.y * scaleBmpPxToCanvasPx),
                            (float) (p2.x * scaleBmpPxToCanvasPx), (float) (p2.y * scaleBmpPxToCanvasPx),
                            paint
                    );
                }

                if (sample.projectedCenter != null) {
                    canvas.drawCircle(
                            (float) (sample.projectedCenter.x * scaleBmpPxToCanvasPx),
                            (float) (sample.projectedCenter.y * scaleBmpPxToCanvasPx),
                            3 * scaleCanvasDensity, // 5 is the radius of the dot
                            centerPaint
                    );
                }
            }
        }
    }


    // ---- Utility functions ----

    private List<Point3> getTopFaceModelPoints() {
        List<Point3> points = new ArrayList<>();
        // Define the 4 vertices of the top face in a local 3D coordinate system.
        // Assuming the origin (0,0,0) of this local system is at one of the top corners.
        points.add(new Point3(0, 0, 0));
        points.add(new Point3(BLOCK_LENGTH, 0, 0));
        points.add(new Point3(BLOCK_LENGTH, BLOCK_WIDTH, 0));
        points.add(new Point3(0, BLOCK_WIDTH, 0));
        return points;
    }

    // Calculates the real-world 3D coordinates on the ground plane (Z=0)
    // that correspond to a given 2D pixel coordinate in the image.
    private Point3 getWorldPosition(Point imagePoint) {
        // Step 1: Correct for lens distortion at the given pixel.
        Point undistortedPoint = undistortPoint(imagePoint);

        // Step 2: Convert the 2D pixel to a normalized 3D ray in the camera's coordinate system.
        // This is done using the camera intrinsic parameters (fx, fy, cx, cy).
        double x_n = (undistortedPoint.x - cameraMatrix.get(0, 2)[0]) / cameraMatrix.get(0, 0)[0];
        double y_n = (undistortedPoint.y - cameraMatrix.get(1, 2)[0]) / cameraMatrix.get(1, 1)[0];
        Point3 rayDirectionCamera = new Point3(x_n, y_n, 1);

        // Step 3: Rotate this camera-centric ray into the world coordinate system.
        Mat cameraRotationMatrix = new Mat(3, 3, CvType.CV_64FC1);
        double angleRad = Math.toRadians(cameraDownAngle);
        cameraRotationMatrix.put(0, 0, new double[]{1, 0, 0, 0, Math.cos(angleRad), -Math.sin(angleRad), 0, Math.sin(angleRad), Math.cos(angleRad)});
        Point3 rayDirectionWorld = transformDirectionToWorld(rayDirectionCamera, cameraRotationMatrix);
        cameraRotationMatrix.release();

        // Step 4: Define the ray's origin, which is the camera's position in the world.
        Point3 rayOriginWorld = new Point3(0, 0, cameraHeight);

        // Step 5: Calculate where this 3D ray intersects the ground plane (where Z=0).
        // This is a standard line-plane intersection calculation.
        double t = -rayOriginWorld.z / rayDirectionWorld.z;

        // The intersection point is the final 3D world coordinate.
        return new Point3(rayOriginWorld.x + rayDirectionWorld.x * t, rayOriginWorld.y + rayDirectionWorld.y * t, 0);
    }

    // Apply a 3x3 rotation matrix to a 3D vector.
    private Point3 transformDirectionToWorld(Point3 rayDirectionCamera, Mat cameraRotation) {
        Mat rayDirectionCameraMat = new Mat(3, 1, CvType.CV_64FC1);
        rayDirectionCameraMat.put(0, 0, rayDirectionCamera.x, rayDirectionCamera.y, rayDirectionCamera.z);

        Mat rayDirectionWorldMat = new Mat(3, 1, CvType.CV_64FC1);
        Core.gemm(cameraRotation, rayDirectionCameraMat, 1.0, new Mat(), 0.0, rayDirectionWorldMat, 0);

        Point3 result = new Point3(rayDirectionWorldMat.get(0, 0)[0], rayDirectionWorldMat.get(1, 0)[0], rayDirectionWorldMat.get(2, 0)[0]);
        rayDirectionCameraMat.release();
        rayDirectionWorldMat.release();
        return result;
    }

    // Correct a 2D point for the camera's lens distortion
    private Point undistortPoint(Point distortedPoint) {
        MatOfPoint2f src = new MatOfPoint2f(distortedPoint);
        MatOfPoint2f dst = new MatOfPoint2f();

        // This OpenCV function computes the ideal, undistorted location of the point.
        Calib3d.undistortPoints(src, dst, cameraMatrix, new MatOfDouble(distCoeffs));

        Point normalizedPoint = dst.toList().get(0);
        src.release(); dst.release();

        // The result is in normalized coordinates, so we must convert it back to pixel coordinates.
        double fx = cameraMatrix.get(0, 0)[0];
        double fy = cameraMatrix.get(1, 1)[0];
        double cx = cameraMatrix.get(0, 2)[0];
        double cy = cameraMatrix.get(1, 2)[0];
        return new Point(normalizedPoint.x * fx + cx, normalizedPoint.y * fy + cy);
    }

    // Loads and scales the camera's intrinsic calibration parameters.
    // Done once at initialization.
    private void loadCameraCalibration(){
        // Scale parameters to match lower resolution than used in calibration.
        double scaleX = (double) FRAME_WIDTH / CALIBRATION_WIDTH;
        double scaleY = (double) frameHeight / CALIBRATION_HEIGHT;

        // Original calibrated values
        double original_fx = 1514.7020418880806;
        double original_fy = 1514.4639462102391;
        double original_cx = 1291.6698643656734;
        double original_cy = 753.1148690399445;

        // Scale focal lengths and principal points
        double new_fx = original_fx * scaleX;
        double new_fy = original_fy * scaleY;
        double new_cx = original_cx * scaleX;
        double new_cy = original_cy * scaleY;

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

    //Defines the camera's static position and orientation in the world.
    private void getStaticCameraPoseInWorld(Mat rvec_cam_to_world, Mat tvec_cam_to_world) {
        // The camera is at (0, 0, height) in world coordinates.
        tvec_cam_to_world.put(0, 0, 0, 0, cameraHeight);

        // The camera is tilted down, which is a rotation around the world's X-axis.
        double angleRad = Math.toRadians(cameraDownAngle);
        Mat R_x = new Mat(3, 3, CvType.CV_64FC1);
        R_x.put(0, 0,
                1, 0, 0,
                0, Math.cos(angleRad), -Math.sin(angleRad),
                0, Math.sin(angleRad), Math.cos(angleRad)
        );

        // Convert the 3x3 rotation matrix to a 3x1 Rodrigues vector for use in other functions.
        Calib3d.Rodrigues(R_x, rvec_cam_to_world);
        R_x.release();
    }
}
