import cv2 as cv
import numpy as np

# Open the video file
cap = cv.VideoCapture('video.mp4')

# Minimum width and height for the detected rectangles
min_width_react = 80
min_height_react = 80

# Line position for counting objects
count_line_position = 550

# Initialize the background subtractor
algo = cv.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

detect=[]
offset=6 #Allowable error between pixel
counter =0


while True:
    ret, frame1 = cap.read()
    
    if not ret:
        break

    # Convert the frame to grayscale
    grey = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    # Apply Gaussian blur to the grayscale image
    blur = cv.GaussianBlur(grey, (3, 3), 5)
    
    # Apply the background subtractor
    img_sub = algo.apply(blur)
    
    # Dilate the image to fill in gaps
    dilat = cv.dilate(img_sub, np.ones((5,5)))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    dilatada = cv.morphologyEx(dilat, cv.MORPH_CLOSE, kernel)
    dilatada = cv.morphologyEx(dilatada, cv.MORPH_CLOSE, kernel)
    
    
    
    # Find contours in the dilated image
    contours, _ = cv.findContours(dilatada, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Draw the counting line
    cv.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)
    
    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv.boundingRect(contour)
        # Only consider contours that meet the size criteria
        if w >= min_width_react and h >= min_height_react:
            cv.rectangle(frame1, (x, y), (x + w, y + h), (0, 0,255), 2)

            center=center_handle(x,y,w,h)
            detect.append(center)
            cv.circle(frame1,center,4,(0,0,255),-1)

            for(x,y) in detect:
                if y<(count_line_position+offset) and y>(count_line_position-offset):
                    counter+=1
                    print("Vehicle Counter :"+str(counter))
                cv.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
                detect.remove((x,y))
                
                
            

        cv.putText(frame1,"Vehicle Counter :"+str(counter),(450,70),cv.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)

    #showing the dilate image
    # cv.imshow('Dilate image',dilatada)

    # Show the original video frame with rectangles and the counting line
    cv.imshow('Video Original', frame1)
    
    # Break the loop when the Enter key is pressed
    if cv.waitKey(1) == 13:
        break

# Release the video capture and close all windows
cap.release()
cv.destroyAllWindows()
