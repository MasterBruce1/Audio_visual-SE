import cv2
import dlib
import numpy as np
name = 0
grid_size = (15, 5)  # 15 frames high, 5 frames wide
frame_count = 75  # Total number of frames you want to process
# Initialize face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/LAVSE-master/LAVSE-master/lip extraction/shape_predictor_68_face_landmarks.dat')
cap = cv2.VideoCapture('D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/dataset/val_data/Pv01/file/lbay1a.mpg')
# display the fps and related properties
# fps = cap.get(cv2.CAP_PROP_FPS)
# totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# print("video fps:", fps)
# print("video total number of frames:", totalNoFrames)

#read a spercific frame
# frame_id = 1
# cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
grid_width =  44 * grid_size[1]
grid_height = 24 * grid_size[0]
final_image = np.zeros((grid_height, grid_width), dtype=np.uint8)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    

    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)
        
        # Extract lip coordinates
        lip_coords = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(48, 68)])  # Lip landmarks
        
        # Find bounding box
        x, y, w, h = cv2.boundingRect(lip_coords)
        #print(x,y,w,h)
        # Draw the bounding box around the lips
        adjusted_x = x - 6
        adjusted_y = y - 6 
        adjusted_w = 46 
        adjusted_h = 26
        #cv2.rectangle(frame, (adjusted_x, adjusted_y), (adjusted_x + w, adjusted_y + h), (0, 255, 0), 2)
        cv2.rectangle(frame, (adjusted_x, adjusted_y), (adjusted_x + adjusted_w, adjusted_y + adjusted_h), (0, 255, 0), 1)
        cropped_image = frame[(adjusted_y + 2):adjusted_y+adjusted_h, (adjusted_x + 2):adjusted_x+adjusted_w]
        graycropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        row = name // grid_size[1]
        col = name % grid_size[1]
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
        final_image[row * 24:(row + 1) * 24, col * 44:(col + 1) * 44] = graycropped
        name  = name + 1
    #cv2.imshow('Lips Bounding Box', graycropped)
    #print("Shape of Image:", graycropped.shape)
    #print("Shape of Image:", gray.shape)
cap.release()
cv2.imwrite('D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/dataset/val_data/Pv01/file/lbay1a{0}.jpg'.format(name), final_image)

cv2.destroyAllWindows()