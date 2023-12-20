# Simple Helper Scripts to Aid in Camera Calibration with OpenCV
## Take Calibration Photos
The program will take an unlimited number of photos from two cameras with a one second delay until the escape key is pressed. Photos are displayed using cv2.imshow(). Photos are stored in left_images and right_images directory respectively.
```
python take_calibration_photos.py
```

### Calibrate Camera
This script calibrates a camera based on the chessboard pattern and saves the parametes to camera_calibration.yml
```
python calibrate.py /path/to/chessboard/images/
```
