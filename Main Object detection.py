# Write Python3 code here 
import os 
import cv2 
import numpy as np 
import tensorflow as tf 
import sys 
from PIL import Image
import codecs
import pytesseract
from googletrans import Translator
from PIL import ImageDraw,ImageOps,ImageFont

class Ocr_phase:
	def extract_text(self,img,boxes,scores):
		boxes=np.squeeze(boxes)
		scores=np.squeeze(scores)
		img_height, img_width, img_channel = img.shape
		absolute_coord = []
		THRESHOLD = 0.9 # adjust your threshold here
		N = len(boxes)
		for i in range(N):
			if scores[i] < THRESHOLD:
				continue
			
			box = boxes[i]
			ymin, xmin, ymax, xmax = box
			x_up = int(xmin*img_width)-10 #wider
			y_up = int(ymin*img_height) -10
			x_down = int(xmax*img_width) +10#right
			y_down = int(ymax*img_height)+5 
			absolute_coord.append((x_up,y_up,x_down,y_down))
		bounding_box_img = []
		for c in absolute_coord:
			bounding_box_img.append(img[c[1]:c[3], c[0]:c[2],:])
		for n in bounding_box_img:
			
			r,c = n.shape[:2]
			row, col = n.shape[:2]
			bottom = n[row-2:row, 0:col]
			mean = cv2.mean(bottom)[0]
			bordersize = 200
			img2 = cv2.copyMakeBorder(
				n,
				top=bordersize,
				bottom=bordersize,
				left=bordersize,
				right=bordersize,
				borderType=cv2.BORDER_CONSTANT,
				value=[mean, mean, mean]
			)



			img2 = cv2.resize(img2, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
			kernal_sharpening = np.array([[-1,-1,-1],
										[-1,9,-1],
										[-1,-1,-1]]) 

			sharpned = cv2.filter2D(img2,-1,kernal_sharpening) 


			img2 =cv2.medianBlur(sharpned,3)


			img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 



			r,c = img2.shape;


			kernal = np.ones((255,1))

			img2[:400,:] = 255
			img2[r-400:,:] = 255
			img2[:,c-400:] = 255
			img2[:,:400] = 255


			for i in range(400,r-400):
				for j in range(400,c-400):
					if img2[i,j] != 0:
						if img2[i,j] > 127:
							img2[i,j] = 255
						else:
							img2[i,j]=0
						
					


			StartCount = list();
			EndCount = list()

			check = True

			index = 0
			CheckIndex = True;


			for k in range(400,c-400):
				mul = img2[400:630,k] * kernal
				mul = np.array(mul);
				
				if mul.min() < 255 and check == True:
						
						StartCount.append(k-1)
						check = False
				elif mul.min() == 255 and check == False:
					EndCount.append(k);
					check = True
					CheckIndex = True
				
				
				if len(StartCount) == len(EndCount) and len(StartCount) != 0 and len(EndCount) != 0 and CheckIndex:
					
					if EndCount[index] - StartCount[index] < 26:
						img2[:,StartCount[index]:EndCount[index]] = 255
					
					CheckIndex = False
					index += 1
			words = pytesseract.image_to_string(img2, lang='jpn_vert')
			print (words,'\n')
			cv2.imshow('thresh', img2)
			cv2.waitKey()	
# This is needed since the notebook is stored in the object_detection folder. 
sys.path.append("..") 

# Import utilites 
from utils import label_map_util 
from utils import visualization_utils as vis_util 

# Name of the directory containing the object detection module we're using 
MODEL_NAME = 'inference_graph' # The path to the directory where frozen_inference_graph is stored. 
IMAGE_NAME = '2x.jpg' # The path to the image in which the object has to be detected. 

# Grab path to current working directory 
CWD_PATH = os.getcwd() 

# Path to frozen detection graph .pb file, which contains the model that is used 
# for object detection. 
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb') 

# Path to label map file 
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt') 

# Path to image 
PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME) 

# Number of classes the object detector can identify 
NUM_CLASSES = 1

# Load the label map. 
# Label maps map indices to category names, so that when our convolution 
# network predicts `5`, we know that this corresponds to `king`. 
# Here we use internal utility functions, but anything that returns a 
# dictionary mapping integers to appropriate string labels would be fine 
label_map = label_map_util.load_labelmap(PATH_TO_LABELS) 
categories = label_map_util.convert_label_map_to_categories( 
		label_map, max_num_classes = NUM_CLASSES, use_display_name = True) 
category_index = label_map_util.create_category_index(categories) 

# Load the Tensorflow model into memory. 

detection_graph = tf.Graph() 
with detection_graph.as_default(): 
	od_graph_def = tf.GraphDef() 
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid: 
		serialized_graph = fid.read() 
		od_graph_def.ParseFromString(serialized_graph) 
		tf.import_graph_def(od_graph_def, name ='') 

	sess = tf.Session(graph = detection_graph) 

# Define input and output tensors (i.e. data) for the object detection classifier 

# Input tensor is the image 
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0') 
# Output tensors are the detection boxes, scores, and classes 
# Each box represents a part of the image where a particular object was detected 
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0') 

# Each score represents level of confidence for each of the objects. 
# The score is shown on the result image, together with the class label. 
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0') 
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0') 

# Number of objects detected 
num_detections = detection_graph.get_tensor_by_name('num_detections:0') 

# Load image using OpenCV and 
# expand image dimensions to have shape: [1, None, None, 3] 
# i.e. a single-column array, where each item in the column has the pixel RGB value 
image = cv2.imread(PATH_TO_IMAGE) 
#

image_expanded = np.expand_dims(image, axis = 0) 										#############################
# Perform the actual detection by running the model with the image as input 
(boxes, scores, classes, num) = sess.run( 
	[detection_boxes, detection_scores, detection_classes, num_detections], 
	feed_dict ={image_tensor: image_expanded}) 
img = cv2.imread(PATH_TO_IMAGE,0)


#print(boxes)
#print(scores)
#print(classes)
x = Ocr_phase()
x.extract_text(image,boxes,scores)
#################################################################
# Draw the results of the detection (aka 'visualize the results') 

vis_util.visualize_boxes_and_labels_on_image_array( 
	image, 
	np.squeeze(boxes), 
	np.squeeze(classes).astype(np.int32), 
	np.squeeze(scores), 
	category_index, 
	use_normalized_coordinates = True, 
	line_thickness = 2, 
	min_score_thresh = 0.70) 

# All the results have been drawn on the image. Now display the image. 
cv2.imshow('Object detector', image) 

# Press any key to close the image 
cv2.waitKey(0) 

# Clean up 
cv2.destroyAllWindows() 

#for Learning
#python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config


#for graphs
#python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-11816 --output_directory inference_graph








