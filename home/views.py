from django.shortcuts import render, redirect, HttpResponse
from home import templates
from .form import NumberPlateForm
from .form import VideoUploadForm
from django.conf import settings
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import pytesseract
import os


def proccessing(image_path):
    img = cv2.imread(image_path)
    print(img)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    bfilter = cv2.bilateralFilter(gray, 11, 11, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    #plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    location = None
    for contour in contours:
    # cv2.approxPolyDP returns a resampled contour, so this will still return a set of (x, y) points
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask = mask)
    # plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))

    pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR/tesseract.exe'
    cropped_image = gray[x1:x2+3, y1:y2+3]
    # edged2 = cv2.Canny(cropped_image, 30, 200)
    config = ('-l eng --oem 3 --psm 10')
    text0 = pytesseract.image_to_string(cropped_image, config =config)
    # plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    print(text0)
    text =""
    for i in text0:
        if (65<=ord(i)<=90)or(48<=ord(i)<=57):
            text += i
    # print(text)

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # location = (approx[0][0][0]-180, approx[1][0][1]+30) # You can adjust the coordinates based on your preference
    # font_scale = 1
    # font_color = (0, 255, 0)  # Green color
    # thickness = 2
    # res = cv2.putText(img, text, location, font, font_scale, font_color, thickness)
    res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (255,0, 0), 3)
    # plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    return text


# Create your views here.

def index(request):
    return render(request, 'index.html')

def image(request):
    if request.method == 'POST':
        form = NumberPlateForm(request.POST, request.FILES)
        if form.is_valid():
            number_plate = form.save()
            number_plate.plate_text = proccessing(number_plate.image.path)
            number_plate.save()
            return render(request, 'base.html', {'result': number_plate.plate_text})
            
    else:
        form = NumberPlateForm()

    return render(request, 'base.html', {'form': form})



def extract_text(image, region):
    roi = image[region[1]:region[3], region[0]:region[2]]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray, config='--psm 8 --oem 3')
    return text.strip()


def extract_text_number_plate(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    bfilter = cv2.bilateralFilter(gray, 11, 11, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    # plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    location = None
    for contour in contours:
    # cv2.approxPolyDP returns a resampled contour, so this will still return a set of (x, y) points
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask = mask)
    # plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))

    cropped_image = gray[x1:x2+3, y1:y2+3]
    # edged2 = cv2.Canny(cropped_image, 30, 200)
    config = ('-l eng --oem 3 --psm 10')
    text0 = pytesseract.image_to_string(cropped_image, config =config)
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    text =""
    for i in text0:
        if (65<=ord(i)<=90)or(48<=ord(i)<=57) or (ord(i)==32) or(ord(i)==45):
            text+=i
    return text


def get_number_plate(video_path):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    cap = cv2.VideoCapture(video_path)
    img=[]
    lst=[]
    l=True

    plate_text=""
    number=""

    # Loop through frames in the video
    while cap.isOpened() and l==True:
        ret, frame = cap.read()

        if not ret:
            break

        # Resize the frame for faster processing
        frame = imutils.resize(frame, width=600)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and help OCR
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Canny edge detector to find edges
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours in the edges
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours=sorted(contours, key = cv2.contourArea, reverse = True)[:20] 

        # Loop through the contours
        for contour in contours:
            # Filter contours based on area
            if cv2.contourArea(contour) > 200 :
                # Get the bounding box coordinates
                x, y, w, h = cv2.boundingRect(contour)

                # Extract text from the license plate region
                plate_text = extract_text(frame, (x, y, x + w, y + h))
                p=True

                for i in plate_text:
                    if (65<=ord(i)<=90)or(48<=ord(i)<=57) or (ord(i)==32) or(ord(i)==45):
                        continue
                    else:
                        p=False
                        break

                # Display the frame with the license plate text
                if p==True and len(plate_text)>5:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    lst.append(plate_text)
                    img = frame
                    l=False
                    break 

        # Display the processed frame
        # cv2.imshow('Result', frame)
        

        # Break the loop if 'q' key is pressed
        # if (cv2.waitKey(1) & 0xFF == ord('q')) or (l == False):
        #     img = frame
        #     break

    try:
        number = extract_text_number_plate(img)

    except:
        print("An Error Occured In the last frame image.")

    if(len(number)<5):
        number= lst[0]
    else:
        number = number
        
    cap.release()
    cv2.destroyAllWindows()
    return number




def video(request):
    number_plate_text = None

    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save()
            number_plate_text = get_number_plate(video.file.path)
    else:
        form = VideoUploadForm()

    return render(request, 'base2.html', {'form': form, 'number_plate_text': number_plate_text})





