# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

import imutils
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)

def main(img):
	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(img)
	image = imutils.resize(image, width=800)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 2)
	i=0
	for rect in rects:

		(x, y, w, h) = rect_to_bb(rect)
		faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
		faceAligned = fa.align(image, gray, rect)
		
		cv2.imshow('q'+str(i),faceAligned)
		i += 1
	cv2.imshow('ds',image)	
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return faceAligned
	

	
    
if __name__ == '__main__':
	cv2.imshow('q',main('smiling_group3.jpg'))
	cv2.waitKey(0)
	cv2.destroyAllWindows()