import cv2
import time
import os

# Function to ensure a directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Directories for saving images
left_images_dir = 'left_images'
right_images_dir = 'right_images'

# Ensure directories exist
ensure_dir(left_images_dir)
ensure_dir(right_images_dir)

# Initialize the cameras
cap1 = cv2.VideoCapture(0)  # First camera
cap2 = cv2.VideoCapture(2)  # Second camera

i = 0  # Counter for the number of photos taken

while True:
    # Capture a frame from each camera
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1 and ret2:
        # Concatenate images horizontally
        combined_image = cv2.hconcat([frame1, frame2])

        # Display the concatenated image
        cv2.imshow('Images Side by Side', combined_image)

        # Save the frames as images in the respective directories
        cv2.imwrite(f'{left_images_dir}/camera1_photo_{i+1}.jpg', frame1)
        cv2.imwrite(f'{right_images_dir}/camera2_photo_{i+1}.jpg', frame2)
        print(f"Captured pair {i+1}")

        i += 1

    # Wait for 1 second or key press
    key = cv2.waitKey(1000)
    if key == 27:  # Escape key
        break

# Release the capture objects
cap1.release()
cap2.release()
cv2.destroyAllWindows()
