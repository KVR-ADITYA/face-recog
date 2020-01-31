
import os
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import random
import imageio
import skimage
import cv2
import copy



def get_faces_from_frame(img,detect_multiple_faces=True,margin=44,image_size=160):

    face_roi=[]

    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto( log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
                
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    if img.ndim<2:
        # print('Unable to align "%s"' % image_path)
        raise Exception("Unable to Align")

    if img.ndim == 2:
        img = facenet.to_rgb(img)
    img = img[:,:,0:3]

    bounding_boxes,_=detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    
    if nrof_faces> 0 :
        det = bounding_boxes[:,0:4]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        
        if nrof_faces>1:
            if detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                img_center = img_size / 2
                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                det_arr.append(det[index,:])
        else:
            det_arr.append(np.squeeze(det))
        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            scaled = skimage.transform.resize(cropped, (image_size, image_size))
            prewhitened=facenet.prewhiten(scaled)
            # nrof_successfully_aligned += 1

            face_roi.append(prewhitened)

    else:
        raise Exception("Cannot Align")

    return np.stack(face_roi) # return face_roi


def main():
    img=imageio.imread("./people.png")

    faces=get_faces_from_frame(img)
    print(len(faces))
    print(type(faces[0]))
    c=0
    for face in faces:
        cv2.imshow("Face {}".format(c),face)
        c+=1

    cv2.waitKey(30000)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
