import face_recognition
import cv2
import numpy as np
import joblib
import time
import openpyxl
import xlrd
import time
import datetime


# clf = joblib.load('encodings.joblib')
naam = joblib.load("name.joblib")
enc = joblib.load("pics.joblib")
clf = joblib.load('encodings.joblib')
names = joblib.load('name.joblib')
centroid = joblib.load('averages.joblib')


video_capture = cv2.VideoCapture(0)


face_locations = []
face_encodings = []
face_names_set1 = set()
face_names = []
process_this_frame = True
print(type(face_names_set1))
timeinit = time.time()
times = set()
indices = set()

a=0
while (time.time()-timeinit<=10.0):
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        l = 0
        
        
        '''face_names = []
        if(face_encodings !=[]):
            face_names = clf.predict(face_encodings)
            #print(clf.predict_proba(face_encodings))
            #print(np.shape(clf.predict_proba(face_encodings))
            for x in face_names:
                face_names_set1.add(x)
        '''
        
        face_names = []
        
        if(face_encodings !=[]):
            face_names = clf.predict(face_encodings)
            for (name,enc) in zip(face_names,face_encodings):
                j = names.index(name)
                
                if(np.linalg.norm(centroid[j]-enc)>0.40):
                    face_names[list(face_names).index(name)] = 'unknown'
                
            for x in face_names:
                if(x!='unknown'):face_names_set1.add(x)
                try:
                    indices.add(names.index(x))
                except:
                    b =1
    
                if(np.size(face_names_set1)>a):
                    times.add(datetime.datetime.now())
                    a = np.size(face_names_set1)
       
        
    
    process_this_frame = not process_this_frame

  
    
print(face_names_set1)

video_capture.release()



  

  
try:
    rows = joblib.load("column.joblib")
except:
    rows = 1

#for i in range(0,1000):
    
column = 30
xfile = openpyxl.load_workbook('Attendance.xlsx')
sheet = xfile['Sheet1']
print(indices)
print(face_names_set1)
print(times)
for (name,index) in zip(face_names_set1,indices):
    if(name!='unknown'):
        sheet.cell(row = rows+1,column = index+1, value = 'present' )
    
xfile.save('Attendance.xlsx')
rows +=1
joblib.dump(rows,"column.joblib")
