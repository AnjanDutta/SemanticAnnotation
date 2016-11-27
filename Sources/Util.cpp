/*
 * Util.cpp
 *
 *  Created on: Sep 2, 2016
 *      Author: Anjan Dutta
 */

#include <tokenizer.hpp>
#include "../Headers/Util.h"

using namespace std;
using namespace cv;

namespace myutil
{
//Function to find angle with Sine rule
double FindOtherAngle(double biggerAngle, double largestDistance, double smallDistance)
{
	double otherAngle;
	otherAngle = smallDistance *sin(biggerAngle * 3.1415926535 /180);
	otherAngle = otherAngle/largestDistance;
	otherAngle = asin(otherAngle)*180.0 / PI;
	return otherAngle;
}

//Function to find angle opposite to largest side of triangle
double FindBiggerAngle(double largestDistance, double smallDistanceOne, double smallDistanceTwo)
{
	double biggerAngle;
	biggerAngle =  pow(smallDistanceOne,2) + pow(smallDistanceTwo, 2) - pow(largestDistance,2);
	biggerAngle = fabs(biggerAngle/(2*smallDistanceOne*smallDistanceTwo));
	biggerAngle = acos(biggerAngle)*180.0 / PI;
	return biggerAngle;
}

bool is_file_exist(const char *fileName)
{
	ifstream infile(fileName);
	return infile.good();
}

std::vector<std::vector<int> > GenDistCols(int n)
{
	std::vector<int> col(3);
	std::vector<std::vector<int> > col_list;
	int count = 0;

	int step = (int) floor(255/(ceil(cbrt(n)) - 1));

	for(int b = 0; b < 256; b += step)
	{
		for(int g = 0; g < 256; g += step)
		{
			for(int r = 0; r < 256; r += step)
				if(!(r == g && g == b && b == r))
				{
					count++;

					col[0] = b;
					col[1] = g;
					col[2] = r;

					//					cout<<b<<", "<<g<<", "<<r<<endl;

					col_list.push_back(col);

					if(count>=n)
						break;
				}
			if(count>=n)
				break;
		}
		if(count>=n)
			break;
	}
	return col_list;
}

static inline float normL2Sqr(const float* a, const float* b, int n)
{
	float s = 0.f;
	for( int i = 0; i < n; i++ )
	{
		float v = a[i] - b[i];
		s += v*v;
	}
	return s;
}

class KMeansPPDistanceComputer : public ParallelLoopBody
{
private:
	KMeansPPDistanceComputer& operator=(const KMeansPPDistanceComputer&); // to quiet MSVC

	float *tdist2;
	const float *data;
	const float *dist;
	const int dims;
	const size_t step;
	const size_t stepci;

public:
	KMeansPPDistanceComputer( float *_tdist2, const float *_data, const float *_dist, int _dims, size_t _step, size_t _stepci )
: tdist2(_tdist2), data(_data), dist(_dist), dims(_dims), step(_step), stepci(_stepci) {}

	void operator()( const cv::Range& range ) const
	{
		const int begin = range.start;
		const int end = range.end;

		for ( int i = begin; i<end; i++ )
		{
			tdist2[i] = std::min(normL2Sqr(data + step*i, data + stepci, dims), dist[i]);
		}
	}
};

class KMeansConstrainedDistanceComputer : public ParallelLoopBody
{
private:
	KMeansConstrainedDistanceComputer& operator=(const KMeansConstrainedDistanceComputer&); // to quiet MSVC

	double *distances;
	int *labels;
	const Mat& data;
	const Mat& centers;
	const vector< vector <int> > CL;
public:
	KMeansConstrainedDistanceComputer( double *_distances, int *_labels, const Mat& _data, const Mat& _centers, const vector< vector <int> > _CL )
	: distances(_distances), labels(_labels), data(_data), centers(_centers), CL(_CL)
{}
	void operator()( const Range& range ) const
	{
		const int begin = range.start;
		const int end = range.end;
		const int K = centers.rows;
		const int dims = centers.cols;
		vector< vector <int> > indices(K);

		for( int i = 0; i < end; ++i)
		{
			const float *sample = data.ptr<float>(i);
			int k_best = 0;
			double min_dist = DBL_MAX;

			double dist;

			for( int k = 0; k < K; k++ )
			{
				sort ( indices[k].begin(), indices[k].end() );
				const float* center = centers.ptr<float>(k);
				dist = normL2Sqr(sample, center, dims);

				vector<int> intersection( CL[i].size() + indices[k].size() );
				std::vector<int>::iterator it = set_intersection( CL[i].begin(), CL[i].end(), indices[k].begin(), indices[k].end(), intersection.begin() );
				intersection.resize(it - intersection.begin());

				if( min_dist > dist && intersection.empty() )
				{
					min_dist = dist;
					k_best = k;
				}

			}

			distances[i] = min_dist;
			labels[i] = k_best;
			indices[k_best].push_back(i);
		}
	}
};

class KMeansDistanceComputer : public ParallelLoopBody
{
private:
	KMeansDistanceComputer& operator=(const KMeansDistanceComputer&); // to quiet MSVC

	double *distances;
	int *labels;
	const Mat& data;
	const Mat& centers;
public:
	KMeansDistanceComputer( double *_distances, int *_labels, const Mat& _data, const Mat& _centers )
	: distances(_distances), labels(_labels), data(_data), centers(_centers)
{}
	void operator()( const Range& range ) const
	{
		const int begin = range.start;
		const int end = range.end;
		const int K = centers.rows;
		const int dims = centers.cols;

		for( int i = begin; i < end; ++i)
		{
			const float *sample = data.ptr<float>(i);
			int k_best = 0;
			double min_dist = DBL_MAX;

			for( int k = 0; k < K; k++ )
			{
				const float* center = centers.ptr<float>(k);
				const double dist = normL2Sqr(sample, center, dims);

				if( min_dist > dist )
				{
					min_dist = dist;
					k_best = k;
				}
			}

			distances[i] = min_dist;
			labels[i] = k_best;
		}
	}
};

static void generateRandomCenter(const std::vector<Vec2f>& box, float* center, RNG& rng)
{
	size_t j, dims = box.size();
	float margin = 1.f/dims;
	for( j = 0; j < dims; j++ )
		center[j] = ((float)rng*(1.f+margin*2.f)-margin)*(box[j][1] - box[j][0]) + box[j][0];
}

/*
	k-means center initialization using the following algorithm:
	Arthur & Vassilvitskii (2007) k-means++: The Advantages of Careful Seeding
 */
static void generateCentersPP(const Mat& _data, Mat& _out_centers,
		int K, RNG& rng, int trials)
{
	int i, j, k, dims = _data.cols, N = _data.rows;
	const float* data = _data.ptr<float>(0);
	size_t step = _data.step/sizeof(data[0]);
	std::vector<int> _centers(K);
	int* centers = &_centers[0];
	std::vector<float> _dist(N*3);
	float* dist = &_dist[0], *tdist = dist + N, *tdist2 = tdist + N;
	double sum0 = 0;

	centers[0] = (unsigned)rng % N;

	for( i = 0; i < N; i++ )
	{
		dist[i] = normL2Sqr(data + step*i, data + step*centers[0], dims);
		sum0 += dist[i];
	}

	for( k = 1; k < K; k++ )
	{
		double bestSum = DBL_MAX;
		int bestCenter = -1;

		for( j = 0; j < trials; j++ )
		{
			double p = (double)rng*sum0, s = 0;
			for( i = 0; i < N-1; i++ )
				if( (p -= dist[i]) <= 0 )
					break;
			int ci = i;

			parallel_for_(Range(0, N),
					KMeansPPDistanceComputer(tdist2, data, dist, dims, step, step*ci));
			for( i = 0; i < N; i++ )
			{
				s += tdist2[i];
			}

			if( s < bestSum )
			{
				bestSum = s;
				bestCenter = ci;
				std::swap(tdist, tdist2);
			}
		}
		centers[k] = bestCenter;
		sum0 = bestSum;
		std::swap(dist, tdist);
	}

	for( k = 0; k < K; k++ )
	{
		const float* src = data + step*centers[k];
		float* dst = _out_centers.ptr<float>(k);
		for( j = 0; j < dims; j++ )
			dst[j] = src[j];
	}
}

double cop_kmeans(InputArray _data, int K, InputOutputArray _bestLabels, TermCriteria criteria, int attempts, int flags, OutputArray _centers,
		const vector< vector <int> > CL )
{
	const int SPP_TRIALS = 3;
	Mat data0 = _data.getMat();
	bool isrow = data0.rows == 1;
	int N = isrow ? data0.cols : data0.rows;
	int dims = (isrow ? 1 : data0.cols)*data0.channels();
	int type = data0.depth();

	attempts = std::max(attempts, 1);
	CV_Assert( data0.dims <= 2 && type == CV_32F && K > 0 );
	CV_Assert( N >= K );

	Mat data(N, dims, CV_32F, data0.ptr(), isrow ? dims * sizeof(float) : static_cast<size_t>(data0.step));

	_bestLabels.create(N, 1, CV_32S, -1, true);

	Mat _labels, best_labels = _bestLabels.getMat();
	if( flags & CV_KMEANS_USE_INITIAL_LABELS )
	{
		CV_Assert( (best_labels.cols == 1 || best_labels.rows == 1) &&
				best_labels.cols*best_labels.rows == N &&
				best_labels.type() == CV_32S &&
				best_labels.isContinuous());
		best_labels.copyTo(_labels);
	}
	else
	{
		if( !((best_labels.cols == 1 || best_labels.rows == 1) &&
				best_labels.cols*best_labels.rows == N &&
				best_labels.type() == CV_32S &&
				best_labels.isContinuous()))
			best_labels.create(N, 1, CV_32S);
		_labels.create(best_labels.size(), best_labels.type());
	}
	int* labels = _labels.ptr<int>();

	Mat centers(K, dims, type), old_centers(K, dims, type), temp(1, dims, type);
	std::vector<int> counters(K);
	std::vector<Vec2f> _box(dims);
	Vec2f* box = &_box[0];
	double best_compactness = DBL_MAX, compactness = 0;
	RNG& rng = theRNG();
	int a, iter, i, j, k;

	if( criteria.type & TermCriteria::EPS )
		criteria.epsilon = std::max(criteria.epsilon, 0.0001);
	else
		criteria.epsilon = FLT_EPSILON;
	criteria.epsilon *= criteria.epsilon;

	if( criteria.type & TermCriteria::COUNT )
		criteria.maxCount = std::min(std::max(criteria.maxCount, 2), 100);
	else
		criteria.maxCount = 100;

	if( K == 1 )
	{
		attempts = 1;
		criteria.maxCount = 2;
	}

	const float* sample = data.ptr<float>(0);
	for( j = 0; j < dims; j++ )
		box[j] = Vec2f(sample[j], sample[j]);

	for( i = 1; i < N; i++ )
	{
		sample = data.ptr<float>(i);
		for( j = 0; j < dims; j++ )
		{
			float v = sample[j];
			box[j][0] = std::min(box[j][0], v);
			box[j][1] = std::max(box[j][1], v);
		}
	}

	for( a = 0; a < attempts; a++ )
	{
		double max_center_shift = DBL_MAX;
		for( iter = 0;; )
		{
			swap(centers, old_centers);

			if( iter == 0 && (a > 0 || !(flags & KMEANS_USE_INITIAL_LABELS)) )
			{
				if( flags & KMEANS_PP_CENTERS )
					generateCentersPP(data, centers, K, rng, SPP_TRIALS);
				else
				{
					for( k = 0; k < K; k++ )
						generateRandomCenter(_box, centers.ptr<float>(k), rng);
				}
			}
			else
			{
				if( iter == 0 && a == 0 && (flags & KMEANS_USE_INITIAL_LABELS) )
				{
					for( i = 0; i < N; i++ )
						CV_Assert( (unsigned)labels[i] < (unsigned)K );
				}

				// compute centers
				centers = Scalar(0);
				for( k = 0; k < K; k++ )
					counters[k] = 0;

				for( i = 0; i < N; i++ )
				{
					sample = data.ptr<float>(i);
					k = labels[i];
					float* center = centers.ptr<float>(k);
					j=0;
					#if CV_ENABLE_UNROLLED
					for(; j <= dims - 4; j += 4 )
					{
						float t0 = center[j] + sample[j];
						float t1 = center[j+1] + sample[j+1];

						center[j] = t0;
						center[j+1] = t1;

						t0 = center[j+2] + sample[j+2];
						t1 = center[j+3] + sample[j+3];

						center[j+2] = t0;
						center[j+3] = t1;
					}
					#endif
					for( ; j < dims; j++ )
						center[j] += sample[j];
					counters[k]++;
				}

				if( iter > 0 )
					max_center_shift = 0;

				for( k = 0; k < K; k++ )
				{
					if( counters[k] != 0 )
						continue;

					// if some cluster appeared to be empty then:
					//   1. find the biggest cluster
					//   2. find the farthest from the center point in the biggest cluster
					//   3. exclude the farthest point from the biggest cluster and form a new 1-point cluster.
					int max_k = 0;
					for( int k1 = 1; k1 < K; k1++ )
					{
						if( counters[max_k] < counters[k1] )
							max_k = k1;
					}

					double max_dist = 0;
					int farthest_i = -1;
					float* new_center = centers.ptr<float>(k);
					float* old_center = centers.ptr<float>(max_k);
					float* _old_center = temp.ptr<float>(); // normalized
					float scale = 1.f/counters[max_k];
					for( j = 0; j < dims; j++ )
						_old_center[j] = old_center[j]*scale;

					for( i = 0; i < N; i++ )
					{
						if( labels[i] != max_k )
							continue;
						sample = data.ptr<float>(i);
						double dist = normL2Sqr(sample, _old_center, dims);

						if( max_dist <= dist )
						{
							max_dist = dist;
							farthest_i = i;
						}
					}

					counters[max_k]--;
					counters[k]++;
					labels[farthest_i] = k;
					sample = data.ptr<float>(farthest_i);

					for( j = 0; j < dims; j++ )
					{
						old_center[j] -= sample[j];
						new_center[j] += sample[j];
					}
				}

				for( k = 0; k < K; k++ )
				{
					float* center = centers.ptr<float>(k);
					CV_Assert( counters[k] != 0 );

					float scale = 1.f/counters[k];
					for( j = 0; j < dims; j++ )
						center[j] *= scale;

					if( iter > 0 )
					{
						double dist = 0;
						const float* old_center = old_centers.ptr<float>(k);
						for( j = 0; j < dims; j++ )
						{
							double t = center[j] - old_center[j];
							dist += t*t;
						}
						max_center_shift = std::max(max_center_shift, dist);
					}
				}
			}

			if( ++iter == MAX(criteria.maxCount, 2) || max_center_shift <= criteria.epsilon )
				break;

			// assign labels
			Mat dists(1, N, CV_64F);
			double* dist = dists.ptr<double>(0);
			parallel_for_(Range(0, N),
					KMeansConstrainedDistanceComputer(dist, labels, data, centers, CL));

			compactness = 0;
			for( i = 0; i < N; i++ )
				compactness += dist[i];

			// cout<<"attempt: "<<a<<", iteration: "<<iter<<", max iteration: "<<criteria.maxCount<<", epsilon: "<<max_center_shift
					// <<", max epsilon: "<<criteria.epsilon<<", compactness: "<<compactness<<endl;

		}

		if( compactness < best_compactness )
		{
			best_compactness = compactness;
			if( _centers.needed() )
				centers.copyTo(_centers);
			_labels.copyTo(best_labels);
		}
	}

	return best_compactness;
}

vector<string> read_config_file( string filename )
{
    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    boost::char_separator<char> sep(" ");

    vector<string> tokens;
    std::ifstream file( filename );
    std::string temp;

    while( getline(file, temp) )
    {
        tokenizer toks( temp , sep );
        for( tokenizer::iterator beg = toks.begin(); beg != toks.end(); ++beg )
            tokens.push_back( *beg );
    }
    return tokens;
}

}