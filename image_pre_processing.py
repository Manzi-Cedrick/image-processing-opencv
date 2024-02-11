import cv2
import pytesseract
from PIL import Image

def pre_on_process(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 7)
    threshBitwiseImage = cv2.bitwise_not(thresh)
    cv2.imwrite('pre_on_process.jpg', threshBitwiseImage)
    return threshBitwiseImage

def show_pre_process(bitwise_image):
    cv2.imshow('threshBitwiseImage', bitwise_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def post_on_process(bitwise_image):
    # Determine the numbers in the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(bitwise_image, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if 0.8 < aspect_ratio < 1.2 and cv2.contourArea(cnt) > 100: 
            digits.append(cnt)

    detected_numbers = []
    for cnt in digits:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = thresh[y:y + h, x:x + w]
        text = pytesseract.image_to_string(roi, config='--psm 7 --oem 3 digits')
        detected_numbers.append(text.strip())

    final_numbers = []
    for num in detected_numbers:
        if len(num) == 1 and num.isdigit():
            final_numbers.append(num)

    return final_numbers    

# Test the functions
bitwise_image = pre_on_process('images/water-meter-reading.jpg')
show_pre_process(bitwise_image)
final_numbers = post_on_process(bitwise_image)
print(final_numbers)