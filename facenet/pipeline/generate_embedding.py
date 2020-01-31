# The Program assumes that we get face ROI already extracted, and is passed into the appropriate functions

import extract_faces
import facenet
import imageio
import tensorflow as tf
import numpy as np


def extract_embeddings(images,model_path="./model"):
#    img=imageio.imread("./people.png")
#    image_roi=extract_faces.get_faces_from_frame(img,detect_multiple_faces=True)
#    images=image_roi


    with tf.Graph().as_default():
        
        with tf.Session() as sess:

            facenet.load_model(model_path)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)

            return(emb)

if __name__=="__main__":
	images=imageio.imread("./people.png")
	print(extract_embeddings(images))
