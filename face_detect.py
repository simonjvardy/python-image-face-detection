import cv2

# load the haar-cascade classifier data
face_cascade = cv2.CascadeClassifier("assets/xml/haarcascade_frontalface_default.xml")

# load the image to be assessed
image = cv2.imread("assets/img/photo.jpg")

# OpenCV is more accurate finding faces in greyscale images
# Convert the source image to greyscale for furhter processing
grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect the coordinates of a detected face as a rectangle
# top-left(x,y), width value, height value
faces = face_cascade.detectMultiScale(
    grey_image,
    scaleFactor=1.05,
    minNeighbors=5
)

# draw the rectangle on the image as a green box line width 3
for x, y, w, h in faces:
    image = cv2.rectangle(
        image,
        (x,y),
        (x+w, y+h),
        (0,255,0),
        3
    )

print(type(faces))
print(faces)

# resize the output image
resized = cv2.resize(
    image,
    (int(image.shape[1]/3),int(image.shape[0]/3))
)

cv2.imshow("Output", resized)

# open the image for a defined period or on key click, then close
cv2.waitKey(0)
cv2.destroyAllWindows()