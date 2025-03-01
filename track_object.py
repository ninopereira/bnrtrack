import cv2
import numpy as np

# Global variables for mouse callback
drawing = False
ix, iy = -1, -1
selection_complete = False
bbox = None
points = []  # Store trajectory points


def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, bbox, selection_complete, frame, original_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Restore the original frame
            frame = original_frame.copy()
            # Draw rectangle with bright color and thicker line
            cv2.rectangle(frame, (ix, iy), (x, y), (255, 0, 0), 3)
            # Add text to show dimensions with black background for better visibility
            w, h = abs(x - ix), abs(y - iy)
            text = f"{w}x{h}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Add black background to text for better visibility
            (text_width, text_height), _ = cv2.getTextSize(text, font, 0.6, 2)
            cv2.rectangle(
                frame,
                (x + 10, y + 5),
                (x + 10 + text_width, y + 20 + text_height),
                (0, 0, 0),
                -1,
            )
            # Draw text in white
            cv2.putText(frame, text, (x + 10, y + 20), font, 0.6, (255, 255, 255), 2)
            # Show the frame with rectangle while drawing
            cv2.imshow("Object Selection", frame)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Draw final rectangle on the frame
        cv2.rectangle(frame, (ix, iy), (x, y), (255, 0, 0), 3)
        bbox = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))
        selection_complete = True


# Initialize video capture
cap = cv2.VideoCapture("sample.mp4")

# Read first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    exit()

# Make a copy of the original frame
original_frame = frame.copy()

# Read second frame
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    exit()

# Create window and set mouse callback
cv2.namedWindow("Object Selection")
cv2.setMouseCallback("Object Selection", draw_rectangle)

print("Draw a rectangle around the object to track, then press ENTER")

# Wait for user to draw rectangle
while True:
    cv2.imshow("Object Selection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 13 and selection_complete:  # ENTER key
        break
    elif key == 27:  # ESC key
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Setup initial tracking window
x, y, w, h = bbox
track_window = (x, y, w, h)

# Set up the ROI for tracking
roi = frame[y : y + h, x : x + w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(
    hsv_roi, np.array((0.0, 60.0, 32.0)), np.array((180.0, 255.0, 255.0))
)
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Print ROI histogram to debug
print("ROI Histogram:", roi_hist)

# Setup the termination criteria
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# Calculate initial center point
center = (int(x + w / 2), int(y + h / 2))
points.append(center)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Calculate back projection
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Apply CamShift to get the new location
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    # Print tracking window to debug
    print("Tracking Window:", track_window)

    # Draw tracked object and update points
    pts = cv2.boxPoints(ret)
    pts = np.int32(pts)
    center = (int((pts[0][0] + pts[2][0]) / 2), int((pts[0][1] + pts[2][1]) / 2))
    points.append(center)

    # Draw current position (blue dot)
    cv2.circle(frame, center, 5, (255, 0, 0), -1)

    # Draw trajectory (line)
    if len(points) > 1:
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], (255, 0, 0), 5)

    # Display frame
    cv2.imshow("Tracking", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
