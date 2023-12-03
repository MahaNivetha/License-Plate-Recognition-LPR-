#!/usr/bin/env python
# coding: utf-8

# In[16]:


#!/usr/bin/env python

# Import the necessary libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils 
import easyocr

# Read the image
img = cv2.imread('/Users/mahanivethakannappan/Downloads/test.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply bilateral filtering to remove noise
bfilter = cv2.bilateralFilter(gray, 11, 11, 17)

# Apply edge detection
edged = cv2.Canny(bfilter, 30, 200)

# Find contours
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None

# Find the contour with 4 corners
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

# Extract the region of interest (ROI)
(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2 + 3, y1:y2 + 3]

# Perform OCR using EasyOCR
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)

# Display the OCR result and accuracy
if result:
    print("OCR Result:", result[0][1])
    print("Accuracy:", result[0][-1])

    # Display the annotated image
    text = result[0][1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText(img, text=text, org=(location[0][0][0], location[1][0][1] + 60), fontFace=font, fontScale=1,
                      color=(0, 255, 0), thickness=5)
    res = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.show()
else:
    print("No OCR result found.")

# Display the original image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


# In[ ]:





# In[ ]:




