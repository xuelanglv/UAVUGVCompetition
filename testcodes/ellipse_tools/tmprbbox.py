import numpy as np 


def angle_diff(boxes, query_boxes):
	'''
	Parameters
	----------------
	boxes: (N, 5) --- x_ctr, y_ctr, height, width, angle
	query: (K, 5) --- x_ctr, y_ctr, height, width, angle
	----------------
	Returns
	---------------- 
	diff (N, K) angles
	'''
	N = boxes.shape[0]
	K = query_boxes.shape[0]

	#angle_diff = np.zeros((N, K), dtype = np.float32)

	angles_pro = boxes[:, 4].reshape(N,1)
	angles_gt = query_boxes[:, 4].reshape(1,K)

	ret = np.abs(angles_pro - angles_gt)
	
	change = np.where(ret>150)
	ret[change] = np.abs(180 - ret[change])	
			
	return ret

if __name__ == "__main__":

	query_boxes = np.array([
			[1151.86, 537.293, 1244.822, 1436.03, 1],
			[1151.86, 637.293, 1234.822, 1446.03, 1],
			[450.0, 450.0, 100.0,150.0 , 2]
		], dtype = np.float32)



	boxes = np.array([
			[60.0, 60.0, 100.0,  100.0, 0.0], # 4 pts
			[50.0, 50.0, 100.0, 100.0, 45.0], # 8 pts
			[80.0, 50.0, 100.0, 100.0, 0.0], # overlap 4 edges
			[50.0, 50.0, 200.0, 50.0, 45.0], # 6 edges
			[200.0, 200.0, 100.0, 100.0, 0], # no intersection
			[60.0, 60.0, 100.0,  100.0, 0.0], # 4 pts
			[50.0, 50.0, 100.0, 100.0, 45.0], # 8 pts
			], dtype = np.float32)
	#boxes = np.tile(boxes,(10000,1));

	#boxes = generate_anchors()

	print boxes

	
