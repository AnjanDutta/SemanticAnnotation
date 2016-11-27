/*
 * Main.cpp
 *
 *  Created on: Aug 17, 2016
 *      Author: Anjan Dutta
 */

#include "../Headers/SemanticAnnotation.h"

using namespace std;
using namespace cv;
using namespace myutil;

// extern vector<word> ReadFile1(string, unsigned int);
extern vector<word> ReadFile2(string, unsigned int);
extern vector<word> ReadXML(string, unsigned int);
// extern void CreateDictionary(string dir_docs, vector<word> words, float per_words_voc, unsigned int size_dict_words, string file_voc);
extern void ComputeBOW( vector<word> words, string dir_docs, string file_voc, string file_words, unsigned int column_id );
extern void ClusterWords( string dir_docs, string dir_clustered_words, string file_words, string file_cluster_words, unsigned int column_id );
extern void CreateTriangles( string file_words, string file_cluster_words, string file_triangles, string file_feats_tris, unsigned int column_id );
extern void CreateCLConstraints( string file_triangle, string file_cl_constraints, unsigned int column_id );
extern void ClusterTriangles( string file_triangles, string file_cl_constraints, string file_feats_tris, string file_labels_tris, unsigned int column_id );
extern void PlotTriangles( string file_words, string file_triangles, string dir_docs, string dir_clustered_tris );

int main(int argc, char *argv[])
{
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////                          Parameters                           ////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    vector<string> params = read_config_file( argv[1] );
    string collection;
    unsigned int column_id = 0;
	string file_voc;

    for( int i = 0; i < params.size() ; i++ )
    {
        if( params[i] == "collection" )
            collection = params[i+1];
        if( params[i] == "column_id" )
            column_id = (unsigned int) atoi( params[i+1].c_str() );
		if( params[i] == "vocabulary" )
			file_voc = params[i+1];
    }

	bool force_compute_words = false; // flag to control force computation of BOW descs for words
    bool force_compute_cluster_words = false; // flag to control force computation of word's clusters
	bool force_compute_triangles = false; // flag to control force computation of triangles
	bool force_compute_constraints_triangles = true; // flag to control force computation of triangles
	bool force_compute_cluster_triangles = true; // flag to control force computation of triangle's clusters

    string usr = string(getenv("USER"));

    string dir_docs = "/home/" + usr + "/Workspace/Qidenus/Database/Documents/" + collection + "/";
    string dir_gts = "/home/" + usr + "/Workspace/Qidenus/Database/GroundTruths/" + collection + "/";
	string dir_sd = "/home/" + usr + "/Workspace/Qidenus/SavedData/" + collection + "/";
	string dir_res = "/home/" + usr + "/Workspace/Qidenus/Results/" + collection + "/";
	string dir_clustered_words = dir_res + "ClusteredWords/Column_" + to_string(column_id) + "/";
	string dir_clustered_tris = dir_res + "ClusteredTriangles/Column_" + to_string(column_id) + "/";

    string file_words = dir_sd + "Words_" + to_string(column_id) + ".bin";
	string file_cluster_words = dir_sd + "WordClusters_" + to_string(column_id) + ".xml";
	string file_triangles = dir_sd + "Triangles_" + to_string(column_id) + ".bin";
    string file_feats_tris = dir_sd + "FeatsTrians_" + to_string(column_id) + ".xml";
	string file_cl_constraints = dir_sd + "CLConstraints_" + to_string(column_id) + ".bin";
	string file_labels_tris = dir_sd + "LabelsTrians_" + to_string(column_id) + ".xml";

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//// Step: 1:: Read file and extract the words /////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	vector<word> words;

//	if(collection == "Wiengersthof_03-07")
//		words = ReadFile2("/home/adutta/Workspace/Qidenus/Database/list_words/allwords.csv", column_id);
//	else if(collection == "Absdorf_01-04")
//        words = ReadFile2("")
//		words = ReadXML(dir_gts + "Absdorf_01-04_1853_1882_9652_03-Taufe_00XX.xml", column_id);
//	else
//	{
//		cout << "Collection not found" << endl;
//		exit(1);
//	}

    words = ReadFile2( "/home/" + usr + "/Workspace/Qidenus/Database/list_words/" + collection + "/allwords.csv", column_id );

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//// Step:2:: Compute the BOW for word images //////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	if( !boost::filesystem::exists( file_words ) || force_compute_words )
		ComputeBOW( words, dir_docs, file_voc, file_words, column_id );
    words.clear();

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//// Step:3:: Cluster the words ////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	if( !boost::filesystem::exists( file_cluster_words ) || force_compute_cluster_words )
		ClusterWords( dir_docs, dir_clustered_words, file_words, file_cluster_words, column_id );

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//// Step:4:: Create triangles and their features //////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	if( !boost::filesystem::exists( file_triangles ) || !boost::filesystem::exists( file_feats_tris ) || force_compute_triangles )
		CreateTriangles( file_words, file_cluster_words, file_triangles, file_feats_tris, column_id );

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//// Step:5:: Create CL constraints ////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	if( !boost::filesystem::exists( file_cl_constraints ) || force_compute_constraints_triangles )
		CreateCLConstraints( file_triangles, file_cl_constraints, column_id );

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//// Step:6:: Cluster the triangles ////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	if( !boost::filesystem::exists( file_labels_tris ) || force_compute_cluster_triangles )
		ClusterTriangles( file_triangles, file_cl_constraints, file_feats_tris, file_labels_tris, column_id );

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//// Step:6:: Plot the triangles on respective documents for visualization /////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    PlotTriangles( file_words, file_triangles, dir_docs, dir_clustered_tris );

	return EXIT_SUCCESS;
}