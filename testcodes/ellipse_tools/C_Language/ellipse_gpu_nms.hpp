#ifndef _ELLIPSE_GPU_NMS_
#define _ELLIPSE_GPU_NMS_

void _ellipse_nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int device_id);



#endif //_ELLIPSE_GPU_NMS_
