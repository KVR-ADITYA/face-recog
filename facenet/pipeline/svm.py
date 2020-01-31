import generate_embedding
from sklearn import svm
import os
import joblib
import numpy
import PIL.Image

encodings = []
names = []

train_dir = os.listdir('images/')

# Loop through each person in the training directory
for person in train_dir:
    pix = os.listdir("images/" + person)
    
    # Loop through each training image for the current person
    for person_img in pix:
        # Get the face encodings for the face in each image file
        #images1 = imageio.imread(person_img)
        img = numpy.array(PIL.Image.open("images/" + person + "/" + person_img))
        face_enc=generate_embedding.extract_embeddings(img,model_path="./model")
        
        # Add face encoding for current image with corresponding label (name) to the training data
        print(person)
        print(numpy.shape(face_enc[0]))
        encodings.append(face_enc[0])
        names.append(person)
        
print(numpy.shape(encodings))
print(numpy.shape(names))

# Create and train the SVC classifier
clf = svm.SVC(gamma='scale')
clf.fit(encodings,names)

joblib.dump(clf,"encodings.joblib")
