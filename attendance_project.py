import cv2
import numpy as np
import face_recognition
import os

# Step 1: Get all image names from the folder where the images are stored
path = 'images'
images = []
studentNames = []
myList = os.listdir(path)
# print(myList) # print all the filenames

# Step 2: Create list of images
for student in myList:
   currentStudent = cv2.imread(f'{path}/{student}')
   images.append(currentStudent)
   studentNames.append(os.path.splitext(student)[0])
# print(studentNames)
# Step 3: Find encondings for a list of images
def findEncodings(images): # 'images' is a list of images
   encodeList = [] # List to store all the encodings
   for img in images:
       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       encode = face_recognition.face_encodings(img)[0]
       encodeList.append(encode)
   return encodeList


encodeListStudents = findEncodings(images)
print(len(encodeListStudents))

# Step 4: Get image from webcam
print("Capturing video")
capture = cv2.VideoCapture(0)
attendance =[]

while True:
   success, img = capture.read()
   imageSmall = cv2.resize(img, (0,0), None, 0.25, 0.25)
   imageSmall = cv2.cvtColor(imageSmall, cv2.COLOR_BGR2RGB)
   facesOnCam = face_recognition.face_locations(imageSmall)
   encodeFacesOnCam = face_recognition.face_encodings(imageSmall, facesOnCam)
   # Release web cam
   capture.release()
   cv2.destroyAllWindows()

   for encodeFace, faceloc in zip(encodeFacesOnCam, facesOnCam):
       matches = face_recognition.compare_faces(encodeListStudents, encodeFace)
       faceDis = face_recognition.face_distance(encodeListStudents, encodeFace)
       # print(faceDis)
       matchIndex = np.argmin(faceDis)
       if matches[matchIndex]:
           # print(studentNames[matchIndex])
           attendance.append(studentNames[matchIndex])
   break


print("Students in class")
print(studentNames)

print("Students present")
print(attendance)