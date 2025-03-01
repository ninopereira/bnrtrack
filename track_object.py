import cv2
import numpy as np
import argparse
import os

# Global variables for mouse callback
drawing = False
ix, iy = -1, -1
selection_complete = False
bbox = None
points = []  # Store trajectory points
original_frame = None  # Declare original_frame as a global variable
frame = None  # Declare frame as a global variable


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


def main(input_file):
    global original_frame, frame  # Declare original_frame and frame as global in the main function

    # Initialize video capture
    cap = cv2.VideoCapture(input_file)

    # Define the codec and create VideoWriter object
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_file = os.path.splitext(input_file)[0] + "_out.mp4"
    out = cv2.VideoWriter(
        output_file, fourcc, fps * 2, (int(cap.get(3)), int(cap.get(4)))
    )

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

    # Increase the ROI by a factor (e.g., 1.5)
    roi_factor = 1
    x = int(x - (roi_factor - 1) * w / 2)
    y = int(y - (roi_factor - 1) * h / 2)
    w = int(w * roi_factor)
    h = int(h * roi_factor)

    # Ensure the ROI is within the frame boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(frame.shape[1] - x, w)
    h = min(frame.shape[0] - y, h)

    track_window = (x, y, w, h)

    # Initialize the CSRT tracker
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, track_window)

    # Calculate initial center point
    center = (int(x + w / 2), int(y + h / 2))
    points.append(center)

    frame_skip = 1  # Number of frames to skip

    while True:
        for _ in range(frame_skip):
            ret, frame = cap.read()
            if not ret:
                break

        if not ret:
            break

        # Update the tracker
        success, track_window = tracker.update(frame)

        if success:
            # Draw tracked object and update points
            x, y, w, h = [int(v) for v in track_window]
            center = (int(x + w / 2), int(y + h / 2))
            points.append(center)

            # Draw current position (blue dot)
            cv2.circle(frame, center, 5, (255, 0, 0), -1)

            # Draw trajectory (line)
            if len(points) > 1:
                for i in range(1, len(points)):
                    cv2.line(frame, points[i - 1], points[i], (255, 0, 0), 25)

        # Write the frame to the output video
        out.write(frame)

        # Display frame
        cv2.imshow("Tracking", frame)

        # Reduce the delay to increase playback speed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track an object in a video.")
    parser.add_argument("input_file", help="Path to the input video file")
    args = parser.parse_args()
    main(args.input_file)
