import cv2
import numpy as np
import os
import sys

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

#save calibration parameter
def save_calibration_parameters(filename, mtx, dist):
    file = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
    file.write("camera_matrix", mtx)
    file.write("distortion_coefficients", dist)
    file.release()
        
# Main function
def main():
    if len(sys.argv) != 2:
        print("Usage: python monocular_calibration.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    images = load_images_from_folder(folder_path)

    # Calibration parameters
    chessboard_size = (9, 6)  # Size of chessboard (number of inner corners)
    square_size = 1.0  # Size of a square in your defined unit (e.g., inch, meter)

    # Termination criteria for corner sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size

    # Arrays to store object points and image points
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('Calibration Image', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save the camera matrix and distortion coefficients
    save_calibration_parameters('calibration_parameters.yml', mtx, dist)

    print("Calibration parameters saved to 'calibration_parameters.yml'")

if __name__ == "__main__":
    main()
