# Image_recognition_Real_time



Steps:<br><br>
1)set the python path variable. In linux, export PYTHONPATH=[...]/Image_recognition_Real_time/face_recognition where [...] is the address of where the repository is cloned.<br>
2)Collect Images and store them in folders as follows...<br>

Structure:
```
      images/
            <person_1>/
                <person_1_face-1>.jpg
                <person_1_face-2>.jpg
                .
                .
                <person_1_face-n>.jpg
           <person_2>/
                <person_2_face-1>.jpg
                <person_2_face-2>.jpg
                .
                .
                <person_2_face-n>.jpg
            .
            .
            <person_n>/
                <person_n_face-1>.jpg
                <person_n_face-2>.jpg
                .
                .
                <person_n_face-n>.jpg</p>
```
keep the 'images' folder in the 'Image_recognition_Real_time' folder.<br>Images can also be created by running the create_face.py. It takes ten consecutive pictures and stores them in appropriate files without further preprocessing.<br>
    3)Run the svm_weight_create.py file. A new .joblib file named "encodings.joblib" should be created. Also, other joblib files like averages.joblib, name.joblib, distance_avg.joblib will be created. These will help in recognition.<br>This file creates face encodings of all the face images and trains an SVM SVC. In order to not recognise unknown faces, we used the other joblib files.<br>
    4)Run the svmcamera.py file, and make sure there is enough lighting. If your face is in the 'images' folder, you should be recognised. Recognition will not happen if the previous step is not followed.<br>
    5)Run the attended_sess.py to get a list of people that have appeared in front of the camera for eight seconds(customizable) sice starting the program.<br><br>


For more information, check out this [link.](https://drive.google.com/file/d/1n5lrjLDvdtw6V3ioV2i0cwltjd9YrO_t/view?usp=sharing)<br>

