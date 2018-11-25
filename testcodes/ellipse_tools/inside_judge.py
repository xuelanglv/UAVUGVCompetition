import numpy as np
import cv2


from fast_rcnn.config import cfg

# xc = col_idx, yc = row_idx


def condinate_rotate(all_anchors):

	left_top = np.array((- all_anchors[:, 2] / 2, - all_anchors[:, 3] / 2)).T # left top
	left_bottom = np.array([- all_anchors[:, 2] / 2, all_anchors[:, 3] / 2]).T # left bottom
	right_top = np.array([all_anchors[:, 2] / 2, - all_anchors[:, 3] / 2]).T # right top
	right_bottom = np.array([all_anchors[:, 2] / 2, all_anchors[:, 3] / 2]).T # right bottom
	
	theta = all_anchors[:, 4]

	#positive angle when anti-clockwise rotation

	cos_theta = np.cos(np.pi / 180 * theta) # D
	sin_theta = np.sin(np.pi / 180 * theta) # D

	# [2, 2, n] n is the number of anchors
	rotation_matrix = [cos_theta, sin_theta, -sin_theta, cos_theta]


	# coodinate rotation
	
	return pts_dot(left_top, rotation_matrix) + np.array((all_anchors[:, 0], all_anchors[:, 1])).T, \
		pts_dot(left_bottom, rotation_matrix) + np.array((all_anchors[:, 0], all_anchors[:, 1])).T, \
		pts_dot(right_top, rotation_matrix) + np.array((all_anchors[:, 0], all_anchors[:, 1])).T, \
		pts_dot(right_bottom, rotation_matrix) + np.array((all_anchors[:, 0], all_anchors[:, 1])).T

def pts_dot(pts, rotat_matrix):
	
	return np.array([pts[:, 0] * rotat_matrix[0] + pts[:, 1] * rotat_matrix[2], pts[:, 0] * rotat_matrix[1] + pts[:, 1] * rotat_matrix[3]]).T
	

	
def ind_inside(pt1, pt2, pt3, pt4, img_width, img_height):

	
	#tic = time.time()
	#inside_ind = []
	
	padding_w = cfg.IMG_PADDING * img_width
	padding_h = cfg.IMG_PADDING * img_height
	iw = img_width+padding_w
	ih = img_height+padding_h
	
	#print type(pt1),pt1.shape
	pt = np.hstack((pt1,pt2,pt3,pt4))
	print pt.shape
	tmp = (pt[:,0:8:2]>-padding_w) & (pt[:,1:8:2]>-padding_h) & (pt[:,0:8:2]<iw) & (pt[:,1:8:2]<ih)
	ins = np.where(tmp[:,0]&tmp[:,1]&tmp[:,2]&tmp[:,3])[0].tolist()

	return ins

import time 

if __name__ == "__main__":
	tic = time.time()
	query_boxes = np.array([
			#[0, 0, 100, 100, 0], # outside
			#[20, 20, 100, 100, 45.0], # outside
			#[200, 200, 100, 100, 0], # outside
			#[195, 195, 100, 50, 45.0], # outside
			[100, 100, 100, 100, 0], # outside
			[200, 200, 200, 100, 45] # inside
			])
	

	pt1, pt2, pt3, pt4 = condinate_rotate(query_boxes)
	print pt1
	print pt2 
	print pt3
	print pt4
	t = ind_inside(pt1, pt2, pt3, pt4, 200, 200)

	print time.time() - tic
	print t

	img = np.zeros(shape = (400,400,3), dtype = np.uint8) + 255
	cv2.ellipse(img,((200,200),(200,100),0),(255,0,0),2)
	cv2.ellipse(img,((200,200),(200,100),45),(0,0,255),2)
	
	cv2.imshow("Test", img)
	cv2.waitKey()
	

