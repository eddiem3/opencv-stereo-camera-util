# OpenCV Camera Calibration Helper Scripts 

The program will take unlimited photos from two cameras with a one-second delay until the escape key is pressed. Photos are displayed using cv2.imshow(). Photos are stored in left_images and right_images directories respectively.
```
python take_calibration_photos.py
```

### Calibrate Camera
This script calibrates a camera based on the chessboard pattern and saves the parameters to camera_calibration.yml
```
python calibrate.py /path/to/chessboard/images/
```
