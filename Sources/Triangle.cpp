/*
 * Triangles.cpp
 *
 *  Created on: Sep 2, 2016
 *      Author: Anjan Dutta
 */

#include "../Headers/Triangle.h"

using namespace std;
using namespace myutil;

void triangle::set_values(std::string doc_name, unsigned int reg_num, point p1, point p2, point p3, unsigned int line1,
                          unsigned int line2, unsigned int line3, Mat bow_desc1, Mat bow_desc2, Mat bow_desc3)
{
	this->doc_name = doc_name;
	this->reg_num = reg_num;
	this->p1 = p1;
	this->p2 = p2;
	this->p3 = p3;
	this->line1 = line1;
	this->line2 = line2;
	this->line3 = line3;
	this->bow_desc1 = bow_desc1;
	this->bow_desc2 = bow_desc2;
	this->bow_desc3 = bow_desc3;
}
void triangle::set_cluster_id(unsigned int cluster_id)
{
	this->cluster_id = cluster_id;
}
void triangle::set_distance_from_cluster_center( float distance_from_cluster_center )
{
    this->distance_from_cluster_center = distance_from_cluster_center;
}
void triangle::print_values()
{
	cout<<setprecision(2)<<fixed;
	std::cout<<"Name: "<<doc_name<<", Coors: "<<p1.x<<" "<<p1.y<<" "<<p2.x<<" "<<p2.y<<" "<<p3.x<<" "<<p3.y<<", Sides: "<<
			side1<<" "<<side2<<" "<<side3<<", Angles: "<<angle1<<" "<<angle2<<" "<<angle3<<std::endl;
}
void triangle::order_points()
{
	if( line1 > line2 || (line1 == line2 && p1.x > p2.x))
	{
		swap(p1, p2);
		swap(line1, line2);
		swap(bow_desc1, bow_desc2);
	}
	if( line1 > line3 || (line1 == line3 && p1.x > p3.x))
	{
		swap(p1, p3);
		swap(line1, line3);
		swap(bow_desc1, bow_desc3);
	}
	if( line2 > line3 || (line2 == line3 && p2.x > p3.x))
	{
		swap(p2, p3);
		swap(line2, line3);
		swap(bow_desc2, bow_desc3);
	}
}
void triangle::cal_sides()
{
	side1 = sqrt(pow(p2.x - p3.x,2) + pow(p2.y - p3.y,2));
	side2 = sqrt(pow(p1.x - p3.x,2) + pow(p1.y - p3.y,2));
	side3 = sqrt(pow(p2.x - p1.x,2) + pow(p2.y - p1.y,2));
}
void triangle::cal_angles()
{
	double tot;

	if(side1 > side2 && side1 > side3)
	{
		// side1 is longest
		angle1 = FindBiggerAngle(side1, side2, side3);
		angle2 = FindOtherAngle(angle1, side1, side2);
		angle3 = FindOtherAngle(angle1, side1, side3);

		tot = angle1 + angle2 + angle3;

		if(tot < 180)
			angle1 = 180 - (angle2 + angle3);
	}
	else if(side2 > side3 && side2 > side1)
	{
		// side2 is longest
		angle2 = FindBiggerAngle(side2, side1, side3);
		angle3 = FindOtherAngle(angle2, side2, side3);
		angle1 = FindOtherAngle(angle2, side2, side1);

		tot = angle1 + angle2 + angle3;

		if(tot < 180)
			angle2 = 180 - (angle3 + angle1);
	}
	else
	{
		// side3 is longest
		angle3 = FindBiggerAngle(side3, side1, side2);
		angle1 = FindOtherAngle(angle3, side3, side1);
		angle2 = FindOtherAngle(angle3, side3, side2);

		tot = angle1 + angle2 + angle3;

		if(tot < 180)
			angle3 = 180 - (angle1 + angle2);
	}
}
void triangle::draw_triangle(Mat image, Scalar col)
{
	line(image, Point(p1.x, p1.y), Point(p2.x, p2.y), col, 3, 8, 0);
	line(image, Point(p2.x, p2.y), Point(p3.x, p3.y), col, 3, 8, 0);
	line(image, Point(p3.x, p3.y), Point(p1.x, p1.y), col, 3, 8, 0);

	circle(image, Point(p1.x, p1.y), 10, Scalar(255, 0, 0), -1, 8, 0);
	circle(image, Point(p2.x, p2.y), 10, Scalar(0, 255, 0), -1, 8, 0);
	circle(image, Point(p3.x, p3.y), 10, Scalar(0, 0, 255), -1, 8, 0);
}
bool triangle::check_sides(void)
{
	if(side1 > 50 && side2 > 50 && side3 > 50 &&
				(side1 + side2) > side3 && (side2 + side3) > side1 && (side3 + side1) > side2)
		return true;
	else
		return false;
}
bool triangle::check_angles(void)
{
	// if any of the angles is less than zero then return false
	if(angle1 < 10 || angle2 < 10 || angle3 < 10)
		return false;
	else
		return true;
}
point triangle::get_p1()
{
	return p1;
}
point triangle::get_p2()
{
	return p2;
}
point triangle::get_p3()
{
	return p3;
}
unsigned int triangle::get_line1()
{
	return line1;
}
unsigned int triangle::get_line2()
{
	return line2;
}
unsigned int triangle::get_line3()
{
	return line3;
}
string triangle::get_doc_name()
{
	return doc_name;
}
unsigned int triangle::get_reg_num()
{
	return reg_num;
}
double triangle::get_side1()
{
	return side1;
}
double triangle::get_side2()
{
	return side2;
}
double triangle::get_side3()
{
	return side3;
}
double triangle::get_angle1()
{
	return angle1;
}
double triangle::get_angle2()
{
	return angle2;
}
double triangle::get_angle3()
{
	return angle3;
}
unsigned int triangle::get_cluster_id()
{
	return cluster_id;
}
Mat triangle::get_bow1()
{
	return bow_desc1;
}
Mat triangle::get_bow2()
{
	return bow_desc2;
}
Mat triangle::get_bow3()
{
	return bow_desc3;
}
float triangle::get_distance_from_cluster_center()
{
    return distance_from_cluster_center;
}
void CreateTriangles(string file_words, string file_cluster_words, string file_triangles, string file_feats_tris, unsigned int column_id)
{
    cout<<"Loading words for constructing triangles in column "<<column_id<<"..."<<flush;

	ifstream inFILE(file_words.c_str(), ios::in | ios::binary);
	if(!inFILE)
	{
		cerr << "Cannot open the file" << endl;
		exit(1);
	}
	boost::archive::binary_iarchive ia(inFILE);
	size_t num_words;
	ia >> num_words;

	vector<word> words;

	for(size_t i = 0; i < num_words; ++i)
	{
		word word_tmp;
	    ia >> word_tmp;
	    words.push_back(word_tmp);
	}

	inFILE.close();

    cout<<"Done."<<endl;

    vector<unsigned int> freq_word_clust;
    freq_word_clust = CountFreqWordClusters( words, column_id );

   	vector<size_t> indices = sort_descend( freq_word_clust );

    // Take some mostly populated clusters
	indices.resize(10);

	// Now create a cropped dictionary with 20 mostly occurring clusters of words

	Mat C, C_cropped;
	FileStorage fs( file_cluster_words, FileStorage::READ );
	fs["C_clusters"] >> C;
	fs.release();

	for(int i = 0; i<indices.size(); i++)
		C_cropped.push_back(C.row(indices[i]));

	vector<triangle> tris;

	vector<Mat> feats_tris;

	cout<<"Constructing triangles in column "<<column_id<<"..."<<flush;

	for(int i = 0; i < (int) words.size(); i++)
		for(int j = i+1; j < (int) words.size(); j++)
			for(int k = j+1; k < (int) words.size(); k++)
				if( find( indices.begin(), indices.end(), words[i].get_cluster_id()) != indices.end() && // Checking whether the ith word belongs to a frequent cluster
                    find( indices.begin(), indices.end(), words[j].get_cluster_id()) != indices.end() && // Checking whether the jth word belongs to a frequent cluster
					find( indices.begin(), indices.end(), words[k].get_cluster_id()) != indices.end() ) // Checking whether the kth word belongs to a frequent cluster
				{
					bool compbl = CheckCompbl(words[i],words[j],words[k]);

					if(compbl)
					{
                        Mat desci = Mat(1, C_cropped.rows, CV_32F);
                        Mat descj = Mat(1, C_cropped.rows, CV_32F);
                        Mat desck = Mat(1, C_cropped.rows, CV_32F);

                        for(int i1 = 0; i1 < C_cropped.rows; i1++)
                        {
                            desci.at<float>(i1) = (float) norm( words[i].bow(), C_cropped.row(i1), NORM_L2 );
                            descj.at<float>(i1) = (float) norm( words[j].bow(), C_cropped.row(i1), NORM_L2 );
                            desck.at<float>(i1) = (float) norm( words[k].bow(), C_cropped.row(i1), NORM_L2 );
                        }
						normalize(desci, desci, 1.0, 0.0, NORM_MINMAX);
						normalize(descj, descj, 1.0, 0.0, NORM_MINMAX);
						normalize(desck, desck, 1.0, 0.0, NORM_MINMAX);

//						cout<<desci<<endl;
//						cout<<descj<<endl;
//						cout<<desck<<endl;
//						getchar();

						triangle tri;

						tri.set_values(words[i].get_doc_name(), // document name
						words[i].get_region_id(), // region number
						words[i].get_centroid(), words[j].get_centroid(), words[k].get_centroid(), // centroids
						words[i].get_line_id(), words[j].get_line_id(), words[k].get_line_id(), // lines
						desci, descj, desck); // bows

						tri.order_points();

						tri.cal_sides();

						if(!tri.check_sides())
							continue;

						tri.cal_angles();

						if(!tri.check_angles())
							continue;

						vector<Mat> bows;

						bows.push_back(desci);
						bows.push_back(descj);
						bows.push_back(desck);

						Mat feats;

						// Concatenating the feats
						hconcat(bows, feats);

						// Normalizing the values between 0 and 1
						normalize(feats, feats);

						feats_tris.push_back(feats);

						tris.push_back(tri);
					}
				}

	// writing the triangles into a binary file
	ofstream outFILE;
	outFILE.open(file_triangles.c_str(), ios::out | ios::binary);
	boost::archive::binary_oarchive oa(outFILE);
	size_t num_tris = tris.size();
	oa << num_tris;
	for(size_t i = 0; i < tris.size(); i++)
		oa << tris[i];
	outFILE.close();

	//create Mat vertically concatenating feats_tris
	Mat F;

	vconcat(feats_tris, F);

	FileStorage fs1(file_feats_tris, FileStorage::WRITE);
	fs1 << "feats_tris" << F;
	fs1.release();

	cout<<"Done."<<endl;

	cout<<"Total "<<tris.size()<<" triangles generated in column "<<column_id<<"."<<endl;
}

void CreateCLConstraints( string file_triangles, string file_cl_constraints, unsigned int column_id )
{

	cout<<"Generating cannot link constraints among triangles in column "<<column_id<<"..."<<flush;

	// Load the triangle file to create constraints
	ifstream inFILE(file_triangles.c_str(), ios::in | ios::binary);
	if(!inFILE)
	{
		cerr << "Cannot open the file" << endl;
		exit(1);
	}

	boost::archive::binary_iarchive ia(inFILE);

	size_t num_tris;
	ia >> num_tris;
	vector<triangle> tris;

	for(size_t i = 0; i < num_tris; ++i)
	{
		triangle tri_tmp;
		ia >> tri_tmp;
		tris.push_back(tri_tmp);
	}

	inFILE.close();

	vector< vector <int> > CL(tris.size()); // Contains cannot link constraints

	int max_const_tris = 0, count_const_tris;

    for( int t1 = 0; t1 < (int) tris.size(); t1++)
	{
		count_const_tris = 1;
		for( int t2 = 0; t2 < (int) tris.size(); t2++)
			if( t1 == t2 )
				continue;
			else if( tris[t1].get_doc_name() == tris[t2].get_doc_name() && tris[t1].get_reg_num() == tris[t2].get_reg_num() )
			{
                CL[t1].push_back(t2);
				count_const_tris++;
				if(count_const_tris > max_const_tris)
					max_const_tris = count_const_tris;
			}
			else
			{
				count_const_tris = 1;
				continue;
			}
	}

	//	#pragma omp parallel num_threads(4)
	//	{
	//		#pragma omp parallel for shared(CL) reduction(+:max_tris)
	//		for( int t1 = 0; t1 < (int) tris.size(); t1++)
	//		{
	//
	//			int count_tris = 1;
	//			vector<int> CL_aux;
	//
	//			for( int t2 = 0; t2 < (int) tris.size(); t2++)
	//			{
	//
	//				if( tris[t1].get_doc_name() == tris[t2].get_doc_name() && tris[t1].get_reg_num() == tris[t2].get_reg_num() )
	//				{
	//
	//					CL_aux.push_back(t2);
	//
	//					count_tris++;
	//					if(count_tris > max_tris)
	//						max_tris = 0 + count_tris;
	//				}
	//				else
	//				{
	//					count_tris = 1;
	//					continue;
	//				}
	//			}
	//
	//			#pragma omp critical
	//			{
	//				CL[t1].insert(CL[t1].end(),CL_aux.begin(),CL_aux.end());
	//			}
	//		}
	//	}

	// Writing the generated constraints into a file

	ofstream outFILE;
	outFILE.open(file_cl_constraints.c_str(), ios::out | ios::binary);
	boost::archive::binary_oarchive oa(outFILE);
	size_t num_cl = CL.size();
	oa << num_cl;
	oa << max_const_tris;
	for(size_t i = 0; i < CL.size(); i++)
		oa << CL[i];
	outFILE.close();

	cout<<"Done."<<endl;
}

void ClusterTriangles( string file_triangles, string file_cl_constraints, string file_feats_tris, string file_labels_tris, unsigned int column_id )
{
	cout<<"Loading feature matrix and cannot link constraints of triangles in column "<<column_id<<"..."<<flush;

	ifstream inFILE(file_cl_constraints.c_str(), ios::in | ios::binary);
	if(!inFILE)
	{
		cerr << "Cannot open the file" << endl;
		exit(1);
	}
	boost::archive::binary_iarchive ia(inFILE);
	size_t num_cl;
	int max_const_tris;

	ia >> num_cl;
	ia >> max_const_tris;

	vector< vector <int> > CL(num_cl);

	for(size_t i = 0; i < CL.size(); ++i)
	{
		vector<int> CL_tmp;
		ia >> CL_tmp;
		CL.at(i) = CL_tmp;
	}
	inFILE.close();

	Mat F;
	FileStorage fs1(file_feats_tris, FileStorage::READ);
	fs1["feats_tris"] >> F;
	fs1.release();

	cout<<"Done"<<endl;

	int attempts = 1;

	Mat L, C;

	int num_clust_triangles = max_const_tris;

	cout<<"Clustering triangles in column "<<column_id<<" with "<<num_clust_triangles<<" classes..."<<flush;

	double compactness = cop_kmeans( F, num_clust_triangles, L, cv::TermCriteria(), attempts, KMEANS_RANDOM_CENTERS, C, CL );

	//	double compactness = kmeans( F, num_clust_triangles, L, cv::TermCriteria(), attempts, KMEANS_PP_CENTERS, C );

    ifstream inFILE_tris(file_triangles.c_str(), ios::in | ios::binary);
    boost::archive::binary_iarchive ia_tris(inFILE_tris);
    size_t num_tris;
    ia_tris >> num_tris;
    vector<triangle> tris;
    for(size_t i = 0; i < num_tris; ++i)
    {
        triangle tri_tmp;
        ia_tris >> tri_tmp;
        tris.push_back(tri_tmp);
    }
    inFILE_tris.close();

    for(int i = 0; i < L.rows; i++) {
        tris[i].set_cluster_id((unsigned int) L.at<int>(i));
        Mat desc;
        desc.push_back( tris[i].get_bow1() );
        desc.push_back( tris[i].get_bow2() );
        desc.push_back( tris[i].get_bow3() );

        hconcat( desc, desc );
        normalize(desc, desc);
        float dist = (float) norm( desc, C.row( (unsigned int) L.at<int>(i) ), NORM_L2 );
        tris[i].set_distance_from_cluster_center( dist );
    }

    ofstream outFILE_tris(file_triangles.c_str(), ios::out | ios::binary);
    boost::archive::binary_oarchive oa(outFILE_tris);
    num_tris = tris.size();
    oa << num_tris;
    for(size_t i = 0; i < tris.size(); i++)
        oa << tris[i];
    outFILE_tris.close();

	ofstream outFILE_labels_tris( file_labels_tris.c_str(), ios::out | ios::binary );
	outFILE_labels_tris.close();

	cout<<"Done."<<endl;
}

void PlotTriangles( string file_words, string file_triangles, string dir_docs, string dir_clustered_tris )
{
    // Get the image names in the corresponding document folder

    DIR *dpdf = opendir(dir_docs.c_str());
    struct dirent *epdf;
    vector<string> docs;
    if (dpdf != NULL){
        while (epdf = readdir(dpdf)){
            char* filename = epdf->d_name;
            size_t len = strlen(filename);
            if(len > 4 && strcmp(filename + len - 4, ".jpg") == 0){
                docs.push_back(string(filename).substr(0,len-4));
            }
        }
    }

    cout<<"Loading words and triangles for plotting..."<<flush;

    ifstream inFILE_words(file_words.c_str(), ios::in | ios::binary);
    if(!inFILE_words)
    {
        cerr << "Cannot open the file: " << file_words << endl;
        exit(1);
    }
    boost::archive::binary_iarchive ia_words(inFILE_words);
    size_t num_words;
    ia_words >> num_words;
    vector<word> words;
    for(size_t i = 0; i < num_words; ++i)
    {
        word word_tmp;
        ia_words >> word_tmp;
        words.push_back(word_tmp);
    }
    inFILE_words.close();

    ifstream inFILE_tris(file_triangles.c_str(), ios::in | ios::binary);
    if(!inFILE_tris)
    {
        cerr << "Cannot open the file: " << file_triangles << endl;
        exit(1);
    }
    boost::archive::binary_iarchive ia_tris(inFILE_tris);
    size_t num_tris;
    ia_tris >> num_tris;
    vector<triangle> tris;
    for(size_t i = 0; i < num_tris; ++i)
    {
        triangle tri_tmp;
        ia_tris >> tri_tmp;
        tris.push_back(tri_tmp);
    }
    inFILE_tris.close();

    cout<<"Done."<<endl;

    // First get the number of clusters of triangles
    unsigned int number_clusters_triangles;
    number_clusters_triangles = 0;
    for(int i = 0; i < (int) tris.size(); i++)
        if( tris[i].get_cluster_id() > number_clusters_triangles )
            number_clusters_triangles = tris[i].get_cluster_id();

    number_clusters_triangles++;

    vector<int> freq_tris_clust(number_clusters_triangles);

    for(int i = 0; i < (int) freq_tris_clust.size(); i++)
        for(int j = 0; j < (int) tris.size(); j++)
            if(tris[j].get_cluster_id() == i)
                freq_tris_clust[i]++;

    vector<size_t> indices_tri_clusters = sort_descend( freq_tris_clust );

    cout<<"Frequent triangle cluster indices: "<<flush;
    for(int i = 0; i < std::min( 10, (int) indices_tri_clusters.size() ); i++)
        cout<<indices_tri_clusters[i]<<", "<<flush;
    cout<<endl;

    cout<<"Creating the directory structures for triangles..."<<flush;

    if( !boost::filesystem::exists( dir_clustered_tris ))
    {
        string cmd_mkdir = "mkdir " + dir_clustered_tris;
        int ignore = std::system(cmd_mkdir.c_str());
    }
    else
    {
        string cmd_rm = "rm -r " + dir_clustered_tris + "*";
        int ignore = std::system(cmd_rm.c_str());
    }

    for(int icl = 0; icl < number_clusters_triangles; icl++)
    {
		int rank = std::find( indices_tri_clusters.begin(), indices_tri_clusters.end(), icl) - indices_tri_clusters.begin();
		char str_rank[10];
		sprintf( str_rank, "%04d", rank );
		string cmd_mkdir = "mkdir " + dir_clustered_tris + "Rank_" + str_rank + "_Cluster_" + to_string(icl);
        int ignore = std::system(cmd_mkdir.c_str());
    }

    cout<<"Done."<<endl;

    vector<int> compression_params;
    compression_params.push_back( CV_IMWRITE_JPEG_QUALITY );
    compression_params.push_back( 15 );

    for(int icl = 0; icl < number_clusters_triangles; icl++)
    {
		int rank = std::find( indices_tri_clusters.begin(), indices_tri_clusters.end(), icl) - indices_tri_clusters.begin();
		char str_rank[10];
		sprintf( str_rank, "%04d", rank );

        cout<<"Plotting triangles of " << icl << "th clusters..." << flush;

        for (int idoc = 0; idoc < docs.size(); idoc++)
        {
            string doc1 = docs[idoc];

            Mat im = imread( dir_docs + doc1 + ".jpg", CV_LOAD_IMAGE_COLOR );

            bool imwrite_flag = false;

            for (int itri = 0; itri < (int) tris.size(); itri++)
                if (tris[itri].get_doc_name() == doc1 && tris[itri].get_cluster_id() == icl )
                {
                    int cl = tris[itri].get_cluster_id();
                    imwrite_flag = true;

                    tris[itri].draw_triangle(im, Scalar(0, 0, 0));

                    point centroid_triangle = ( tris[itri].get_p1() + tris[itri].get_p2() + tris[itri].get_p3() )/3;
                    string str = to_string( tris[itri].get_distance_from_cluster_center() );
                    putText( im, str, Point( centroid_triangle.x, centroid_triangle.y ), FONT_HERSHEY_PLAIN, 3, cvScalar(0, 0, 0), 2 );
                }

            if( imwrite_flag )
            {
                for(int iw = 0; iw < (int) words.size(); iw++)
                    if( words[iw].get_doc_name() == doc1 )
                    { //&& find(indices_word_clusters.begin(), indices_word_clusters.end(),
                        // words[iw].get_cluster_id()) != indices_word_clusters.end() ) {
                        myrectangle rect = words[iw].bbox();
                        rectangle(im, Point( rect.x, rect.y ), Point( rect.x+rect.w, rect.y+rect.h ), Scalar( 0, 0, 0 ), 3, 8, 0);

                        string str = to_string(words[iw].get_cluster_id())+" "+to_string(words[iw].get_line_id());

                        putText(im, str, Point(rect.x, rect.y), FONT_HERSHEY_PLAIN, 3, cvScalar(0, 0, 255), 2 );
                    }

                imwrite( dir_clustered_tris + "Rank_" + str_rank + "_Cluster_" + to_string(icl) + "/" + doc1 + ".jpg", im, compression_params );
            }
        }
        cout<<"Done."<<endl;
    }
}

