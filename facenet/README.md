# [FACENET](https://arxiv.org/pdf/1503.03832.pdf)
Generation of a 512 dimentional vector representative (face embeddings) for any face. Using these vectors we perform classification on any input face.

## Usage

### Face Extraction using MTCNN
The input frame containing faces must be passed ```get_faces_from_frame()``` in ```extract_faces.py```, which returns stacked numpy ndarray of all faces in the frame.

### Embedding Generation
The stack must then be passed to ```extract_embeddings()``` in ```generate_embedding.py```, which returns the required vector.


### Model Weights
Download (from [here](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit)) and places all the files( not the folder) in the ``` ./pipeline/model ``` directory to load and run the model.
The model has been trained on LFW dataset.



## Requirements
Tensorflow,
Numpy,
Scikit-Image,
Imageio,
OpenCV,
Scipy
