This program preprocess images for face recognition program.

You need load "shape_predictor_68_face_landmarks.dat" for dlib::shape_predictor and pass it as first argument to program.
(You can download it here: https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2).

Also you need organize directory with images from different persons as follows and pass it as second parameter:

/data_set/jason/image_0<br/>
/data_set/jason/image_1<br/>
/data_set/jason/image_2<br/>
...<br/>
/data_set/mike/image_0<br/>
/data_set/mike/image_1<br/>
/data_set/mike/image_2<br/>
...<br/>

Create directory where to save processed images and pass it as third parameter.

Read about fourth and fifth parameters here: http://dlib.net/dlib/image_transforms/interpolation_abstract.h.html
(100-300 is optimal for fourth and 0.5 is optimal for fifth).

Pass sixth parameter as 1 if you want to see processing stages.

Pass seventh parameter as 1 if you want draw rectangle around processed person faces (but processed images will be saved with this rectangle!).



