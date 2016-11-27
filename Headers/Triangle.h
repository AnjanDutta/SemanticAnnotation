/*
 * Triangles.h
 *
 *  Created on: Sep 2, 2016
 *      Author: Anjan Dutta
 */

#ifndef HEADERS_TRIANGLE_H_
#define HEADERS_TRIANGLE_H_

#include "Util.h"
#include "Word.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>
#include <dirent.h>

using namespace std;
using namespace cv;

class triangle
{
private:
	friend boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		string doc_name;
		unsigned int reg_num, cluster_id, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y;
        int cols1, rows1, elem_type1, cols2, rows2, elem_type2, cols3, rows3, elem_type3;
		double side1, side2, side3, angle1, angle2, angle3;
        float distance_from_cluster_center;
		size_t elem_size1, data_size1, elem_size2, data_size2, elem_size3, data_size3;

		if(Archive::is_saving::value)
		{
            doc_name = this->doc_name; reg_num = this->reg_num; cluster_id = this->cluster_id;
            distance_from_cluster_center = this->distance_from_cluster_center;
            p1_x = this->p1.x; p1_y = this->p1.y; p2_x = this->p2.x; p2_y = this->p2.y; p3_x = this->p3.x;
            p3_y = this->p3.y; cols1 = this->bow_desc1.cols; rows1 = this->bow_desc1.rows;
			elem_type1 = this->bow_desc1.type(); elem_size1 = this->bow_desc1.elemSize(); data_size1 = cols1*rows1*elem_size1;
			cols2 = this->bow_desc2.cols; rows2 = this->bow_desc2.rows; elem_type2 = this->bow_desc2.type();
			elem_size2 = this->bow_desc2.elemSize(); data_size2 = cols2*rows2*elem_size2; cols3 = this->bow_desc3.cols; rows3 = this->bow_desc3.rows;
			elem_type3 = this->bow_desc3.type(); elem_size3 = this->bow_desc3.elemSize(); data_size3 = cols3*rows3*elem_size3;

			ar & doc_name & reg_num & p1_x & p1_y & p2_x & p2_y & p3_x & p3_y & side1 & side2 & side3 & angle1 & angle2 & angle3 & cols1 & rows1 &
			elem_size1 & elem_type1 & cols2 & rows2 & elem_size2 & elem_type2 & cols3 & rows3 & elem_size3 & elem_type3 & cluster_id &
            distance_from_cluster_center & boost::serialization::make_array(bow_desc1.ptr(), data_size1) &
            boost::serialization::make_array(bow_desc2.ptr(), data_size2) & boost::serialization::make_array(bow_desc3.ptr(), data_size3);
		}

		if(Archive::is_loading::value)
		{
			ar & doc_name & reg_num & p1_x & p1_y & p2_x & p2_y & p3_x & p3_y & side1 & side2 & side3 & angle1 & angle2 & angle3 & cols1 & rows1 &
			elem_size1 & elem_type1 & cols2 & rows2 & elem_size2 & elem_type2 & cols3 & rows3 & elem_size3 & elem_type3 & cluster_id &
            distance_from_cluster_center;

			this->doc_name = doc_name; this->reg_num = reg_num; this->cluster_id = cluster_id; this->p1.x = p1_x; this->p1.y = p1_y;
			this->p2.x = p2_x; this->p2.y = p2_y; this->p3.x = p3_x; this->p3.y = p3_y; this->side1 = side1; this->side2 = side2;
			this->side3 = side3; this->angle1 = angle1; this->angle2 = angle2; this->angle3 = angle3; this->distance_from_cluster_center = distance_from_cluster_center;

			this->bow_desc1.create(rows1, cols1, elem_type1);
			this->bow_desc2.create(rows2, cols2, elem_type2);
			this->bow_desc3.create(rows3, cols3, elem_type3);

			data_size1 = cols1*rows1*elem_size1;
			data_size2 = cols2*rows2*elem_size2;
			data_size3 = cols3*rows3*elem_size3;

			ar & boost::serialization::make_array(this->bow_desc1.ptr(), data_size1) &
			boost::serialization::make_array(this->bow_desc2.ptr(), data_size2) &
			boost::serialization::make_array(this->bow_desc3.ptr(), data_size3);
		}
	}
	string doc_name;
	unsigned int reg_num;
	point p1;
	point p2;
	point p3;
	unsigned int line1;
	unsigned int line2;
	unsigned int line3;
	double side1;
	double side2;
	double side3;
	double angle1;
	double angle2;
	double angle3;
	Mat bow_desc1;
	Mat bow_desc2;
	Mat bow_desc3;
	unsigned int cluster_id;
    float distance_from_cluster_center;
public:
	void set_values(string doc_name, unsigned int reg_num, point p1, point p2, point p3, unsigned int line1,
                    unsigned int line2, unsigned int line3, Mat bow_desc1, Mat bow_desc2, Mat bow_desc3);
	void set_cluster_id( unsigned int cluster_id );
    void set_distance_from_cluster_center( float distance_from_cluster_center );
    float get_distance_from_cluster_center();
	string get_doc_name();
	unsigned int get_reg_num();
	point get_p1();
	point get_p2();
	point get_p3();
	unsigned int get_line1();
	unsigned int get_line2();
	unsigned int get_line3();
	double get_side1();
	double get_side2();
	double get_side3();
	double get_angle1();
	double get_angle2();
	double get_angle3();
	Mat get_bow1();
	Mat get_bow2();
	Mat get_bow3();
	unsigned int get_cluster_id();
	void print_values();
	void order_points();
	void cal_sides();
	void cal_angles();
	bool check_sides(void);
	bool check_angles(void);
	void draw_triangle( Mat image, Scalar col );
	friend void CreateTriangles( string file_words, string file_cluster_words, string file_triangles, string file_feats_tris, unsigned int column_id );
	friend void CreateCLConstraints( string file_triangle, string file_cl_constraints, unsigned int column_id );
	friend void ClusterTriangles( string file_triangles, string file_cl_constraints, string file_feats_tris, string file_labels_tris, unsigned int column_id );
    friend void PlotTriangles( string file_words, string file_triangles, string dir_docs, string dir_clustered_tris );
};

#endif /* HEADERS_TRIANGLE_H_ */
