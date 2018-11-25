import numpy as np

def angle_distance_dir(theta1, theta2):
	#print np.where(np.logical_or(theta1 > 90 , theta1 < -90))
	idx1 = np.where(np.logical_or(theta1 > 90 , theta1 < -90))
	idx2 = np.where(np.logical_or(theta2 > 90 , theta2 < -90))
	if len(idx1) > 0:
		assert 'The theta1 must be in the valid range'
	if len(idx2) > 0:
		assert 'The theta2 must be in the valid range'
	
	theta1 = theta1/180 * np.pi
	theta2 = theta2/180 * np.pi
	
	dir_angle =  np.sin(2*theta1 - 2*theta2)
	dir_angle = dir_angle/np.abs(dir_angle)
	
	d_theta = 1.0 - np.cos(2*theta1 - 2*theta2)
	
	idx_clockwise = np.sin(2 * theta1 - 2 * theta2) < 0
	d_theta[idx_clockwise] *= -1.0
	
	return d_theta

def angle_distance_div(theta1, dtheta):
	idx1 = np.where(np.logical_or(theta1 > 90 , theta1 < -90))
	if len(idx1) > 0:
		assert 'The theta1 must be in the valid range'
		
	dtheta[np.where(dtheta > 2)] = 2.0
	dtheta[np.where(dtheta < -2)] = -2.0
	
	theta1 = theta1/180 * np.pi
	
	
	idx_clockwise = dtheta < 0
	dis_angle = np.arccos(1.0 - np.abs(dtheta))/2
	dis_angle[idx_clockwise] *= -1.0
	
	theta2 = (theta1 + dis_angle)/np.pi * 180
	
	
	idx = np.where(theta2 > 90)
	theta2[idx] = theta2[idx] - 180
	idx = np.where(theta2 < -90)
	theta2[idx] = theta2[idx] + 180	
	
	return theta2



# [ctr_x, ctr_y, height, width, angle] anti-clock-wise arc
def rbbox_transform(ex_rois, gt_rois):
	ex_widths = ex_rois[:, 3] 
	ex_heights = ex_rois[:, 2] 
	ex_ctr_x = ex_rois[:, 0]
	ex_ctr_y = ex_rois[:, 1]
	ex_angle = ex_rois[:, 4] 	

	gt_widths = gt_rois[:, 3]
	gt_heights = gt_rois[:, 2]
	gt_ctr_x = gt_rois[:, 0]
	gt_ctr_y = gt_rois[:, 1]
	gt_angle = gt_rois[:, 4]

	targets_dx = (gt_ctr_x - ex_ctr_x) * 1.0 / ex_widths
	targets_dy = (gt_ctr_y - ex_ctr_y) * 1.0 / ex_heights
	targets_dw = np.log(gt_widths * 1.0 / ex_widths)
	targets_dh = np.log(gt_heights * 1.0 / ex_heights)
	
	targets_dtheta = angle_distance_dir(gt_angle * 1.0, ex_angle * 1.0) 

	targets = np.vstack(
		 (targets_dx, targets_dy, targets_dw, targets_dh, targets_dtheta)
	).transpose()

	return targets


def rbbox_transform_inv(boxes, deltas):

	if boxes.shape[0] == 0:
		return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

		boxes = boxes.astype(deltas.dtype, copy=False)

	widths = boxes[:, 3]
	heights = boxes[:, 2]
	ctr_x = boxes[:, 0]
	ctr_y = boxes[:, 1]

	angle = boxes[:, 4]
	
	dx = deltas[:, 0::5]
	dy = deltas[:, 1::5]
	dw = deltas[:, 2::5]
	dh = deltas[:, 3::5]
	da = deltas[:, 4::5] 
	
	pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
	pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
	pred_w = np.exp(dw) * widths[:, np.newaxis]
	pred_h = np.exp(dh) * heights[:, np.newaxis]
	
	pred_angle = angle_distance_div(angle[:, np.newaxis] * 1.0, da * 1.0)
	
	pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
	# ctr_x1
	pred_boxes[:, 0::5] = pred_ctr_x
	# ctr_y1
	pred_boxes[:, 1::5] = pred_ctr_y
	# height
	pred_boxes[:, 2::5] = pred_h
	# width
	pred_boxes[:, 3::5] = pred_w
	# angle
	pred_boxes[:, 4::5] = pred_angle

	return pred_boxes


if __name__ == "__main__":
	ex_rois = np.array([[100, 100, 100, 100, 45], [33, 34, 76, 2, 90]])
	gt_rois = np.array([[101, 99, 50, 50, -60], [123, 545, 3, 5, -89]])
	targets = rbbox_transform(ex_rois, gt_rois)
	print 'gt delta: ', targets
	print 'gt target:', gt_rois
	print 'est target:', rbbox_transform_inv(ex_rois, targets)

