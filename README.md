## License Plate Recognition (LPR) using EasyOCR and OpenCV

This Python script demonstrates License Plate Recognition (LPR) using EasyOCR and OpenCV. It reads an image containing a vehicle license plate, extracts the plate region, and performs Optical Character Recognition (OCR) to recognize the characters on the license plate.

### Prerequisites

Make sure you have the required libraries installed. You can install them using the following:

```bash
pip install opencv-python imutils easyocr matplotlib numpy
```

### Usage

1. Set the `img_path` variable to the path of the image you want to process:

```python
img_path = '/Users/mahanivethakannappan/Downloads/test.jpg'
```

2. Run the script:

```bash
python license_plate_recognition.py
```

### Steps

1. **Read and Preprocess Image:**
   - Read the input image.
   - Convert the image to grayscale.
   - Apply bilateral filtering to remove noise.
   - Apply edge detection using the Canny algorithm.

2. **Find Contours:**
   - Find contours in the edge-detected image.
   - Sort the contours based on area in descending order.

3. **Locate License Plate:**
   - Find the contour with four corners, which represents the license plate.
   - Create a mask and extract the region of interest (ROI) containing the license plate.

4. **Perform OCR using EasyOCR:**
   - Use EasyOCR to perform Optical Character Recognition (OCR) on the license plate region.

5. **Display Results:**
   - Print the OCR result and accuracy.
   - Annotate the original image with the recognized text and display the result.

### Output

The script will display the original image along with the annotated image showing the recognized license plate text and a bounding box around the license plate.

### Note

- The accuracy of OCR may vary based on the quality of the input image and the license plate's condition.
