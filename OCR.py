import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

Cropped_img_loc = '7.png'
cv2.imshow("Cropped image",cv2.imread(Cropped_img_loc))
cv2.waitKey(0)

Cropped_img_loc = '7.png'
text = pytesseract.image_to_string(Cropped_img_loc, lang='eng')
print("The Vehicle Number is: ",text)
