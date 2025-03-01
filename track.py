import cv2
import numpy as np

# Initialize video capture object
cap = cv2.VideoCapture('sample.mp4')  # Use 0 for webcam

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# List to store the center points of the tracked object
points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the background subtractor
    fgmask = fgbg.apply(frame)

    # Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(cnt)

        # If the area is above a certain threshold, draw a rectangle around it
        if area > 500:  # Adjust this threshold as needed
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate the center point of the rectangle
            center = (x + w // 2, y + h // 2)
            points.append(center)

            # Draw a circle at the center point
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # Draw the path of the tracked object
    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], (255, 0, 0), 2)

    # Display the frames
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgmask)

    # Break the loop on 'q' key press
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
