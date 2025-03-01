import cv2
import numpy as np

# Initialize video capture object
cap = cv2.VideoCapture("sample.mp4")  # Use 0 for webcam

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# List to store the center points of the tracked object
points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Create a copy of the original frame
    display_frame = frame.copy()

    # Apply the background subtractor
    fgmask = fgbg.apply(frame)

    # Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_area = 0
    largest_contour = None

    # Find the largest contour (assuming it's the robot)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500 and area > largest_area:  # Adjust threshold as needed
            largest_area = area
            largest_contour = cnt

    if largest_contour is not None:
        # Get bounding rectangle for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate the center point of the rectangle
        center = (x + w // 2, y + h // 2)

        # Draw a blue dot at the current position
        cv2.circle(
            display_frame, center, 10, (255, 0, 0), -1
        )  # Blue dot, larger radius

    # Display the original frame with the blue dot
    cv2.imshow("Robot Tracking", display_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
