import cv2
import numpy as np
import face_recognition

# Step 1: load images and covert to RGB
# Load image for recognition
imgElon = face_recognition.load_image_file("images/Elon musk.jpeg")
# Convert image to RGB
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

# Test image
# Load image
imgTest = face_recognition.load_image_file("test_images/test_image.jpeg")
# Convert image to RGB
imgElonTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)


# Step 2: Finding the faces in the image and their encodings
faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

# Test image
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)


# Step 3: Compare encodings(128 values) output is True or False
results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)

print(results, faceDis)
cv2.putText(imgElonTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# Display image
cv2.imshow("Elon musk", imgElon)
cv2.imshow("Elon test", imgElonTest)
cv2.waitKey(0)


