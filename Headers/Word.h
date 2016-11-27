/*
 * Words.h
 *
 *  Created on: Sep 2, 2016
 *      Author: Anjan Dutta
 */

#ifndef HEADERS_WORD_H_
#define HEADERS_WORD_H_

#include "Util.h"
#include "ImageProcessing.hpp"
#include "VisualWords.hpp"
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <algorithm/string.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;

class WordInformation
{
public:
    // -[ Constructor, destructor and assignation operator ]-----------------------------------------------------------------------------------------
    /// Default constructor.
    WordInformation(void) : m_cluster_ids(-1) {}
    /// Constructor which initializes the word information object.
    WordInformation(const std::string &word_uris, const BoundingBox &coordinates) : m_word_uris(word_uris), m_coordinates(coordinates), m_cluster_ids(-1) {}

    // -[ Access functions ]-------------------------------------------------------------------------------------------------------------------------
    /// Returns the URI of the word.
    inline const std::string& getWordURI(void) const { return m_word_uris; }
    /// Sets the URI of the word.
    inline void setWordURI(const std::string &word_uris) { m_word_uris = word_uris; }
    /// Returns the bounding box of the word snippet.
    inline const BoundingBox& getBoundingBox(void) const { return m_coordinates; }
    /// Sets the bounding box information of the word snippet.
    inline void setBoundingBox(const BoundingBox &coordinates) { m_coordinates = coordinates; }
    /// Returns a constant reference to the BoVW signature of the word snippet.
    inline const std::vector<std::tuple<double, unsigned int> >& getSignature(void) const { return m_bovw_signature; }
    /// Sets the BoVW signature from a sparse-vector.
    inline void setSignature(const std::vector<std::tuple<double, unsigned int> > &bovw_signature) { m_bovw_signature = bovw_signature; }
    /// Sets the BoVW signature from a dense-vector.
    void setSignature(const std::vector<double> &bovw_signature);
    /// Returns the visual cluster identifier.
    inline unsigned int getVisualClusterID(void) const { return m_cluster_ids; }
    /// Sets the visual cluster identifier.
    inline void setVisualClusterID(unsigned int cluster_ids) { m_cluster_ids = cluster_ids; }
protected:
    /// URI of the word.
    std::string m_word_uris;
    /// Coordinates of the word in the image.
    BoundingBox m_coordinates;
    /// BoVW signature of the word.
    std::vector<std::tuple<double, unsigned int> > m_bovw_signature;
    /// Identifier of the cluster assigned to the word.
    unsigned int m_cluster_ids;
    // TODO: ADD CLUSTER REPRESENTATIVES
};

class word
{
private:
	friend boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		string doc_name;
		unsigned int record_id, region_id, line_id, column_id, cluster_id, bb_x, bb_y, bb_w, bb_h, cntr_x, cntr_y;
        int elem_type, cols, rows;
		size_t elem_size, data_size;

		ar & doc_name & region_id & line_id & column_id & cluster_id & bb_x & bb_y & bb_h & bb_w & cntr_x & cntr_y;

		if(Archive::is_saving::value)
		{
			doc_name = this->doc_name; record_id = this->record_id; region_id = this->region_id; line_id = this->line_id;
			column_id = this->column_id; cluster_id = this->cluster_id; bb_x = this->bb.x; bb_y = this->bb.y;
			bb_w = this->bb.w; bb_h = this->bb.h; cntr_x = this->centroid.x; cntr_y = this->centroid.y;
			cols = this->bow_desc.cols; rows = this->bow_desc.rows; elem_type = this->bow_desc.type();
			elem_size = this->bow_desc.elemSize(); data_size = cols*rows*elem_size;

			ar & doc_name & record_id & region_id & line_id & column_id & cluster_id & bb_x & bb_y & bb_w & bb_h &
					cntr_x & cntr_y & cols & rows & elem_size & elem_type &
					boost::serialization::make_array(bow_desc.ptr(), data_size);
		}

		if(Archive::is_loading::value)
		{
			ar & doc_name & record_id & region_id & line_id & column_id & cluster_id & bb_x & bb_y & bb_w & bb_h &
					cntr_x & cntr_y & cols & rows & elem_size & elem_type;
			this->doc_name = doc_name; this->record_id = record_id; this->region_id = region_id; this->line_id = line_id;
			this->column_id = column_id; this->cluster_id = cluster_id; this->bb.x = bb_x; this->bb.y = bb_y;
			this->bb.w = bb_w; this->bb.h = bb_h; this->centroid.x = cntr_x; this->centroid.y = cntr_y;
			this->bow_desc.create(rows, cols, elem_type);

			data_size = cols*rows*elem_size;
			ar & boost::serialization::make_array(this->bow_desc.ptr(), data_size);
		}
	}
	string doc_name;
    unsigned int record_id; // Qidenus uses record
	unsigned int region_id; // CVC uses regions / cells
	unsigned int line_id;
	unsigned int column_id;
    unsigned int cluster_id;
	myrectangle bb;
	point centroid;
	Mat bow_desc;
public:
	void set_values(std::string doc_name, unsigned int record_id, unsigned int region_id, unsigned int line_id,
					unsigned int column_id, unsigned int cluster_id, unsigned int x, unsigned int y, unsigned int w,
					unsigned int h);
    void set_cluster_id(unsigned int cluster_id);
    void set_line_id(unsigned int line_id);
    void set_column_id(unsigned int column_id);
	void set_bow(Mat bow_desc);
	void print_values();
	std::string get_doc_name();
	unsigned int get_record_id();
	unsigned int get_region_id();
	unsigned int get_line_id();
	unsigned int get_column_id();
    unsigned int get_cluster_id();
	myrectangle bbox();
	point get_centroid();
	Mat bow();
    friend vector<word> ReadFile1(string, unsigned int);
    friend vector<word> ReadFile2(string, unsigned int);
    friend vector<word> ReadXML(string, unsigned int);
    friend vector<unsigned int> CountFreqWordClusters(vector<word> words, unsigned int column_id);
	friend bool CheckCompbl(word, word, word);
	friend void CreateDictionary(string dir_docs, vector<word> words, float per_words_voc, unsigned int size_dict_words,
                                 string file_voc);
	friend void ComputeBOW( vector<word> words, string dir_docs, string file_voc, string file_words, unsigned int column_id );
	friend void ClusterWords( string dir_docs, string dir_clustered_words, string file_words, string file_cluster_words, unsigned int column_id );
};

#endif /* HEADERS_WORD_H_ */
