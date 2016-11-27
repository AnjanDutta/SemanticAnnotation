/*
 * Util.h
 *
 *  Created on: Sep 2, 2016
 *      Author: Anjan Dutta
 */

#ifndef HEADERS_UTIL_H_
#define HEADERS_UTIL_H_

#define PI 3.1415926535897932384;

#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <filesystem/path.hpp>
#include <filesystem/operations.hpp>

using namespace std;
using namespace cv;

typedef struct
{
	int xmin;
	int ymin;
	int w;
	int h;
	int type; // 0 heading 1 record 2 line 3 word
	std::string filename;
	std::string transcription;
	std::string semantic_label;
	double confidence;
	int idline;
	int idcell;
	int idregion;
} BoundingBox;

typedef struct point
{
	unsigned int x;
	unsigned int y;

	bool operator==(const point& rhs) const
    {
        return (x == rhs.x && y == rhs.y);
    }
	bool operator<(const point& rhs) const
	{
	    return (y < rhs.y || (y == rhs.y && x < rhs.x));
	}
	bool operator>(const point& rhs) const
	{
	    return (y > rhs.y || (y == rhs.y && x > rhs.x));
	}
	bool operator<=(const point& rhs) const
	{
	    return (y <= rhs.y || (y == rhs.y && x <= rhs.x));
	}
	bool operator>=(const point& rhs) const
	{
	    return (y >= rhs.y || (y == rhs.y && x >= rhs.x));
	}
    point operator+(const point& rhs)
    {
        return {x+rhs.x, y+rhs.y};
    }
    point operator/(const int& rhs)
    {
        return {x/rhs, y/rhs};
    }
	friend ostream& operator <<(ostream &o, const point &a)
	{
		o << a.x << " " << a.y << flush;
		return o;
	}
	friend istream& operator >>(istream &i, point &a)
	{
		i >> a.x >> a.y;
		return i;
	}
}point;

typedef struct myrectangle
{
	unsigned int x;
	unsigned int y;
	unsigned int w;
	unsigned int h;

	friend ostream& operator <<(ostream &o, const myrectangle &r)
	{
		o<<r.x<<" "<<r.y<<" "<<r.w<<" "<<r.h<<flush;
		return o;
	}
	friend istream& operator >>(istream &i, myrectangle &r)
	{
		i >> r.x >> r.y >> r.w >> r.h;
		return i;
	}
}myrectangle;

namespace myutil
{
	double FindOtherAngle(double biggerAngle, double largestDistance, double smallDistance);
	double FindBiggerAngle(double largestDistance, double smallDistanceOne, double smallDistanceTwo);
	bool is_file_exist(const char *fileName);
	std::vector<std::vector<int> > GenDistCols(int n);
	static inline float normL2Sqr(const float* a, const float* b, int n);
	unsigned int CalMedian(vector<unsigned int> values);
    vector<string> read_config_file( string filename );
  	double cop_kmeans(InputArray _data, int K, InputOutputArray _bestLabels, TermCriteria criteria, int attempts, int flags, OutputArray _centers,
			const vector< vector <int> > CL);
	template<typename T1, typename T2, typename T3>	void sort3(T1 &a1, T1 &b1, T1 &c1, T2 &a2, T2 &b2, T2 &c2, T3 &a3, T3 &b3, T3 &c3)
	{
		if(a1>b1)
		{
			std::swap(a1,b1);
			std::swap(a2,b2);
			std::swap(a3,b3);
		}
		if(a1>c1)
		{
			std::swap(a1,c1);
			std::swap(a2,c2);
			std::swap(a3,c3);
		}

		if(b1>c1)
		{
			std::swap(b1,c1);
			std::swap(b2,c2);
			std::swap(b3,c3);
		}
	}
    template <typename T>
    vector<size_t> sort_ascend(const vector<T> &v)
    {
        // initialize original indices locations
        vector<size_t> idx(v.size());
        iota(idx.begin(), idx.end(), 0);

        // sort indices based on comparing values in v
        sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

        return idx;
    }
    template <typename T>
    vector<size_t> sort_descend(const vector<T> &v)
    {
        // initialize original indices locations
        vector<size_t> idx(v.size());
        iota(idx.begin(), idx.end(), 0);

        // sort indices based on comparing values in v
        sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

        return idx;
    }
	template<typename _Tp, typename _AccTp> static inline _AccTp normL2Sqr(const _Tp* a, int n)
	{
	    _AccTp s = 0;
	    int i=0;
	#if CV_ENABLE_UNROLLED
	    for( ; i <= n - 4; i += 4 )
	    {
	        _AccTp v0 = a[i], v1 = a[i+1], v2 = a[i+2], v3 = a[i+3];
	        s += v0*v0 + v1*v1 + v2*v2 + v3*v3;
	    }
	#endif
	    for( ; i < n; i++ )
	    {
	        _AccTp v = a[i];
	        s += v*v;
	    }
	    return s;
	}
	template<typename _Tp, typename _AccTp> static inline _AccTp normL2Sqr(const _Tp* a, const _Tp* b, int n)
	{
	    _AccTp s = 0;
	    int i= 0;
	#if CV_ENABLE_UNROLLED
	    for(; i <= n - 4; i += 4 )
	    {
	        _AccTp v0 = _AccTp(a[i] - b[i]), v1 = _AccTp(a[i+1] - b[i+1]), v2 = _AccTp(a[i+2] - b[i+2]), v3 = _AccTp(a[i+3] - b[i+3]);
	        s += v0*v0 + v1*v1 + v2*v2 + v3*v3;
	    }
	#endif
	    for( ; i < n; i++ )
	    {
	        _AccTp v = _AccTp(a[i] - b[i]);
	        s += v*v;
	    }
	    return s;
	}
}

#endif /* HEADERS_UTIL_H_ */
