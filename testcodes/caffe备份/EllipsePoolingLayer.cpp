#include "caffe/fast_rcnn_layers.hpp"

namespace caffe {

template<typename Dtype>
void EllipseONPoolingLayer<Dtype>::GetEllipseAreaON(int h, int w) {
	int* elp_fea = new int[h*w];
	memset(elp_fea, 0, h * w * sizeof(int));

	float cen_x = (w-1)/2, cen_y = (h-1)/2;

	float ix, iy, theta, tx, ty;
	for(int i = 0; i < w; i++)
	{
		for(int j = 0;j < h; j++)
		{
			ix = i - cen_x;
			iy = j - cen_y;
			theta = atan2(iy, ix);
			tx = w/2 * cos(theta) + cen_x;
			ty = h/2 * sin(theta) + cen_y;
			elp_fea[int(ty+0.5)*w + int(tx+0.5)] = 1;
		}
	}

	for(int i = 0 ; i< h; i++)
	{
		for(int j = 0; j< w; j++)
		{
			if(elp_fea[i * w + j] > 0)
			{
				valid_x.push_back(i);
				valid_y.push_back(j);
			}
		}
	}

	valid_num = valid_x.size();


	delete[] elp_fea;

}

template<typename Dtype>
void EllipseONPoolingLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	EllipseONPoolingParameter roi_pool_param =
			this->layer_param_.ellipse_on_pool_param();
	CHECK_GT(roi_pool_param.pooled_h(), 0) << "pooled_h must be > 0";
	CHECK_GT(roi_pool_param.pooled_w(), 0) << "pooled_w must be > 0";
	pooled_height_ = roi_pool_param.pooled_h();
	pooled_width_ = roi_pool_param.pooled_w();
	spatial_scale_ = roi_pool_param.spatial_scale();
	LOG(INFO) << "Spatial scale: " << spatial_scale_;

	GetEllipseAreaON(pooled_height_, pooled_width_);

}

template<typename Dtype>
void EllipseONPoolingLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();

	top[0]->Reshape(bottom[1]->num(), channels_, 1, valid_num);
	max_idx_.Reshape(bottom[1]->num(), channels_, 1,
			valid_num);

}


template<typename Dtype>
void EllipseONPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
	NOT_IMPLEMENTED;
}

template<typename Dtype>
void EllipseONPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
	NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(EllipseONPoolingLayer);
#endif

INSTANTIATE_CLASS(EllipseONPoolingLayer);
REGISTER_LAYER_CLASS(EllipseONPooling);

}
