import cv2
import numpy as np
import glob

class StereoCamera:
    def __init__(self, left_cam_params, right_cam_params, chessboard_size):
        self.camera_matrix1, self.dist_coeffs1 = self.load_calibration_parameters(left_cam_params)
        self.camera_matrix2, self.dist_coeffs2 = self.load_calibration_parameters(right_cam_params)
        self.chessboard_size = chessboard_size

    def load_calibration_parameters(self, file_name):
        file = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
        camera_matrix = file.getNode("camera_matrix").mat()
        dist_coeffs = file.getNode("distortion_coefficients").mat()
        file.release()
        return camera_matrix, dist_coeffs

    def load_stereo_images(self, left_img_dir, right_img_dir):
        left_images = [cv2.imread(file) for file in glob.glob(left_img_dir + '/*.jpg')]
        right_images = [cv2.imread(file) for file in glob.glob(right_img_dir + '/*.jpg')]
        return left_images, right_images

    def find_matching_chessboard_corners(self, images_left, images_right):
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)

        objpoints = []  # 3d points in real world space
        imgpoints_left = []  # 2d points in image plane for the left camera
        imgpoints_right = []  # 2d points in image plane for the right camera

        for img_l, img_r in zip(images_left, images_right):
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            ret_l, corners_l = cv2.findChessboardCorners(gray_l, self.chessboard_size, None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, self.chessboard_size, None)

            if ret_l and ret_r:
                objpoints.append(objp)
                imgpoints_left.append(corners_l)
                imgpoints_right.append(corners_r)
        
        return objpoints, imgpoints_left, imgpoints_right

    def stereo_calibration(self, objpoints, imgpoints_left, imgpoints_right, image_size):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right, self.camera_matrix1, self.dist_coeffs1, 
            self.camera_matrix2, self.dist_coeffs2, image_size, criteria=criteria, flags=0
        )
        return R, T

    def stereo_rectify(self, R, T, image_size):
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.camera_matrix1, self.dist_coeffs1,
            self.camera_matrix2, self.dist_coeffs2,
            image_size, R, T, None, None, 
            cv2.CALIB_ZERO_DISPARITY, 0
        )
        print(R1, R2, P1,P2,Q)
        return R1, R2, P1, P2, Q

    def compute_disparity_map(self, left_image, right_image):
        # Ensure images are in grayscale format
        if len(left_image.shape) == 3:
            left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        if len(right_image.shape) == 3:
            right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        stereo = cv2.StereoBM_create(numDisparities=16 * 2, blockSize=15)  # Adjust as needed
        disparity = stereo.compute(left_image, right_image)

        # Filter out noise and invalid values
        disparity = np.where(disparity < 0, 0, disparity)

        return disparity

    def display_depth_map(self, map1_x, map1_y, map2_x, map2_y):
        cap_left = cv2.VideoCapture(0)
        cap_right = cv2.VideoCapture(2)

        while True:
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()

            if not ret_left or not ret_right:
                print("Problem with one of the cameras")
                break


            rectified_left = cv2.remap(frame_left, map1_x, map1_y, cv2.INTER_LINEAR)
            rectified_right = cv2.remap(frame_right, map2_x, map2_y, cv2.INTER_LINEAR)

            disparity = self.compute_disparity_map(rectified_left, rectified_right)
            disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            cv2.imshow('Disparity Map', disparity_visual)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()


# Usage
stereo_cam = StereoCamera('left_camera.yml', 'right_camera.yml', (9, 6))
left_images, right_images = stereo_cam.load_stereo_images('left_images', 'right_images')
image_size = left_images[0].shape[1], left_images[0].shape[0]

objpoints, imgpoints_left, imgpoints_right = stereo_cam.find_matching_chessboard_corners(left_images, right_images)
R, T = stereo_cam.stereo_calibration(objpoints, imgpoints_left, imgpoints_right, image_size)
R1, R2, P1, P2, Q = stereo_cam.stereo_rectify(R, T, image_size)
map1_x, map1_y = cv2.initUndistortRectifyMap(stereo_cam.camera_matrix1, stereo_cam.dist_coeffs1, R1, P1, image_size, cv2.CV_32FC1)
map2_x, map2_y = cv2.initUndistortRectifyMap(stereo_cam.camera_matrix2, stereo_cam.dist_coeffs2, R2, P2, image_size, cv2.CV_32FC1)
stereo_cam.display_depth_map(map1_x, map1_y, map2_x, map2_y)
