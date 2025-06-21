import cv2


def detectVehicleCoords(image)->tuple:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Apply the Otsu thresholding to get the optimal threshold value
    threshold, ret = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Apply the Canny edge detection with the threshold value from Otsu
    edges = cv2.Canny(image, threshold/2, threshold)

    # Create a kernels 
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

    # Apply morphological closing
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel1)

    # Apply morphological opening
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel2)

    # Find all contours in the image
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        # Find the rectangle that surrounds the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.imshow("Rect Image", image)
        #print("Coordinates of rectangle:", (x, y, w, h))
    else:
        print ("no contours")
        return (0,0,0,0)
    return x,y, x+w, y+h