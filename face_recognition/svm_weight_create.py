import face_recognition
from sklearn import svm
import os
import joblib
import numpy
import itertools
import openpyxl

encodings = []
names = []
total = 0
avg = []
avg_dist = 0
store_dist = []
face_enc123 = []
train_dir = os.listdir('images/')

# Loop through each person in the training directory
for person in train_dir:
    pix = os.listdir("images/" + person)
    i=0
    # Loop through each training image for the current person
    for person_img in pix:
        # Get the face encodings for the face in each image file
        face = face_recognition.load_image_file("images/" + person + "/" + person_img)
        face_enc = face_recognition.face_encodings(face)[0]
        
        # Add face encoding for current image with corresponding label (name) to the training data
        print(person)
        encodings.append(face_enc)
        face_enc123.append(face_enc)
        names.append(person)
        total = numpy.add(total,face_enc)
        i = i+1
    avg.append(total/i)
    print(numpy.shape(face_enc123))
    res = itertools.combinations(face_enc123,2)
    d = 0
    for each in res:
        d = d+1
        avg_dist = avg_dist+numpy.linalg.norm(each[1]-each[0])
    avg_dist = avg_dist/d
    store_dist.append(avg_dist)
    avg_dist=0
    total=0
    face_enc123=[]
    
# Create and train the SVC classifier
clf = svm.SVC(gamma='scale',kernel='poly')
clf.fit(encodings,names)

store_encodings = []
name_store = []
name_store.append(names[0])
store_encodings.append(encodings[0])
y = names[0]
i = 1
for x in names:
        if(y!=x):
                store_encodings.append(encodings[i])
                name_store.append(x)
                y = x
                i = i+1
joblib.dump(store_encodings,"pics.joblib")
joblib.dump(name_store,"name.joblib")
joblib.dump(avg,"averages.joblib")
joblib.dump(clf,"encodings.joblib")
joblib.dump(store_dist,"dstance_avg.joblib")


xfile = openpyxl.load_workbook('Attendance.xlsx')
sheet = xfile['Sheet1']
for x in range(1,101):
    for y in range(1,101):
        sheet.cell(row=x, column=y)
rows = 1
for name in name_store:
    sheet.cell(row = 1,column = rows, value = name)
    rows +=1
xfile.save('Attendance.xlsx')
