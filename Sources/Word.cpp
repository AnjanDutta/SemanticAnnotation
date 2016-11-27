/*
 * Word.cpp
 *
 *  Created on: Sep 2, 2016
 *      Author: Anjan Dutta
 */

#include "../Headers/Word.h"

using namespace std;
using namespace cv;
using namespace myutil;

void WordInformation::setSignature(const std::vector<double> &bovw_signature)
{
    unsigned int non_zero;

    non_zero = 0;
    for (unsigned int i = 0; i < bovw_signature.size(); ++i)
        if (bovw_signature[i] != 0)
            ++non_zero;

    m_bovw_signature.resize(non_zero);
    non_zero = 0;
    for (unsigned int i = 0; i < bovw_signature.size(); ++i)
    {
        if (bovw_signature[i] != 0)
        {
            m_bovw_signature[non_zero] = std::make_pair(bovw_signature[i], i);
            ++non_zero;
        }
    }
}
void word::set_values(std::string doc_name, unsigned int record_id, unsigned int region_id, unsigned int line_id,
                      unsigned int column_id, unsigned int cluster_id, unsigned int x, unsigned int y, unsigned int w,
                      unsigned int h)
{
	this->doc_name = doc_name;
	this->record_id = record_id;
	this->region_id = region_id;
	this->line_id = line_id;
	this->column_id = column_id;
    this->cluster_id = cluster_id;
	this->bb.x = x;
	this->bb.y = y;
	this->bb.w = w;
	this->bb.h = h;
}
void word::set_cluster_id(unsigned int cluster_id)
{
    this->cluster_id = cluster_id;
}
void word::set_line_id(unsigned int line_id)
{
    this->line_id = line_id;
}
void word::set_column_id(unsigned int column_id)
{
    this->column_id = column_id;
}
void word::set_bow(Mat bow_desc)
{
	this->bow_desc = bow_desc;
}
void word::print_values()
{
	cout<<doc_name<<" "<<record_id<<" "<<region_id<<" "<<line_id<<" "<<column_id<<" "<<cluster_id<<" "<<bb.x<<" "<<bb.y
		<<" "<<bb.w<<" "<<bb.h<<endl;
}
std::string word::get_doc_name()
{
	return doc_name;
}
unsigned int word::get_record_id()
{
	return record_id;
}
unsigned int word::get_region_id()
{
	return region_id;
}
unsigned int word::get_line_id()
{
	return line_id;
}
unsigned int word::get_column_id()
{
	return column_id;
}
unsigned int word::get_cluster_id()
{
    return cluster_id;
}
Mat word::bow()
{
	return bow_desc;
}
myrectangle word::bbox()
{
	return bb;
}
point word::get_centroid()
{
	centroid.x = bb.x+bb.w/2;
	centroid.y = bb.y+bb.h/2;

	return centroid;
}
// Read word file with cluster ids for words
vector<word> ReadFile1(string file_name, unsigned int column_id)
{
	std::string line_str, skip, doc_name;
	char doc_reg[1000], doc_line[1000], word_clid[50];
	unsigned int record_id = 0, region_id, line_id, column_id1, cluster_id, x, y, w, h;

    ifstream file(file_name.c_str());

    if(!file)
    	cerr << "Cannot open file: "<<file_name<<"."<<endl;

    // rewind the file
    file.clear();
    file.seekg(0,ios::beg);

    // skip the first line
    getline(file, skip);

    // store them in vector container class
    vector<word> words;

    while(getline(file, line_str))
    {
    	const char *line = line_str.c_str();

    	sscanf(line," \"%*[^#]#%[^\"]\" , \"%*[^#]#%[^\"]\" , \"%*[^\"]\" , \"%d\" , \"%*[^#]#%[^\"]\" , \"%d\" , "
                "\"%d\" , \"%d\" , \"%d\" ,", doc_reg, doc_line, &column_id1, word_clid, &x, &y, &w, &h);

    	// An area threshold to discard small rectangles
    	if(w*h<2400 || column_id1 != column_id)
    		continue;

    	string reg = string(doc_reg);
    	string lin = string(doc_line);
    	string clid = string(word_clid);

    	size_t pos1 = reg.find("_reg");
    	size_t pos2 = lin.find("_line");
    	size_t pos3 = clid.find("numc");

		doc_name = reg.substr(0,pos1);
		region_id = (unsigned int) atoi(reg.substr(pos1+4).c_str());
		line_id = (unsigned int) atoi(lin.substr(pos2+5).c_str());
		cluster_id = (unsigned int) atoi(clid.substr(pos3+4).c_str());

		word word_tmp;

		word_tmp.set_values(doc_name, record_id, region_id, line_id, column_id1, cluster_id, x, y, w, h);

    	words.push_back(word_tmp);
    }

    file.close();

    return words;
}

// Read word file without cluster ids for words
vector<word> ReadFile2(string file_name, unsigned int column_id)
{
    std::string line_str, skip, doc_name;
    char doc_reg[1000], doc_line[1000];
    unsigned int record_id = 0, region_id, line_id, column_id1, cluster_id = 0, x, y, w, h;

    ifstream file(file_name.c_str());

    if(!file)
        cerr << "Cannot open file: "<<file_name<<"."<<endl;

    // rewind the file
    file.clear();
    file.seekg(0,ios::beg);

    // skip the first line
    getline(file, skip);

    // store them in vector container class
    vector<word> words;

    while(getline(file, line_str))
    {
        const char *line = line_str.c_str();

        sscanf(line," \"%*[^#]#%[^\"]\" , \"%*[^#]#%[^\"]\" , \"%*[^\"]\" , \"%d\" , \"%d\" , \"%d\" , \"%d\" ,"
				" \"%d\" ", doc_reg, doc_line, &column_id1, &x, &y, &w, &h);

        // An area threshold to discard the small rectangles
        if(w*h<2400 || column_id1 != column_id)
            continue;

        string reg = string(doc_reg);
        string lin = string(doc_line);

        size_t pos1 = reg.find("_reg");
        size_t pos2 = lin.find("_line");

		doc_name = reg.substr(0,pos1);
		region_id = (unsigned int) atoi(reg.substr(pos1+4).c_str());
		line_id = (unsigned int) atoi(lin.substr(pos2+5).c_str());

        word word_tmp;

		word_tmp.set_values(doc_name, record_id, region_id, line_id, column_id1, cluster_id, x, y, w, h);

        words.push_back(word_tmp);
    }

    file.close();

    return words;
}

vector<word> ReadXML(string file_name, unsigned int column_id)
{
    std::string line_str, skip, doc_name;
    unsigned int count_record = 0, count_region = 0, count_column = 0, count_line = 0, column_id1, cluster_id = 0;
	float x, y, w, h, x1, x2, y1, y2;

    ifstream file(file_name.c_str());

    if(!file)
        cerr << "Cannot open file: "<<file_name<<"."<<endl;

    // rewind the file
    file.clear();
    file.seekg(0,ios::beg);

	getline(file, skip);

    // store them in vector container class
    vector<word> words;
    vector<word> words_region;
    vector<float> vertical_positions_line_in_region;

    while(getline(file, line_str))
	{
		string line_ = boost::trim_copy( line_str );

//        cout<<line_<<endl;

		if(strcmp(line_.substr(1, 4).c_str(), "Page")==0)
		{
			size_t found1 = line_.find_last_of('/')+1;
			size_t found2 = line_.find_last_of('.');

			doc_name = line_.substr(found1, found2-found1);

			count_record = 0;
            count_region = 0;
		}
		else if(strcmp(line_.substr(1, 6).c_str(), "record")==0)
        {
            count_record++;
            count_column = 0;
        }
		else if(strcmp(line_.substr(1, 6).c_str(), "region")==0)
        {
            sscanf(line_.c_str(),"%*[^\"]\"%*f\" %*[^\"]\"%*f\" %*[^\"]\"%*f\" %*[^\"]\"%*f\" %*[^\"]\"%d\">", &column_id1);
            count_region++;
            count_column++;
			count_line = 0;
        }
        else if(strcmp(line_.substr(0, 9).c_str(), "</region>")==0)
        {
            vector<size_t> indices_lines = sort_ascend( sort_ascend( vertical_positions_line_in_region ) );
            vertical_positions_line_in_region.clear();

            for( size_t t = 0; t < words_region.size(); t++ ) {
                words_region[t].set_line_id((unsigned int) indices_lines[words_region[t].get_line_id()]);
                words.push_back(words_region[t]);
            }
            words_region.clear();
        }
		else if(strcmp(line_.substr(1, 4).c_str(), "line")==0)
        {
            sscanf(line_.c_str(),"%*[^\"]\"%f\" %*[^\"]\"%f\" %*[^\"]\"%f\" %*[^\"]\"%f\" %*[^>]>", &x1, &y1, &x2, &y2);
            vertical_positions_line_in_region.push_back( (float) 0.5 * ( y1 + y2 ) );
            count_line++;
        }

		else if(strcmp(line_.substr(1, 4).c_str(), "word")==0)
		{
			sscanf(line_.c_str(),"%*[^\"]\"%f\" %*[^\"]\"%f\" %*[^\"]\"%f\" %*[^\"]\"%f\" %*[^>]>", &y, &x, &h, &w);
			unsigned int x_ = (unsigned int) fabs(x);
			unsigned int y_ = (unsigned int) fabs(y);
			unsigned int h_ = (unsigned int) fabs(h);
			unsigned int w_ = (unsigned int) fabs(w);

            if( column_id1 == column_id )
            {
                word word_tmp;
                word_tmp.set_values(doc_name, count_record-1, count_region-1, count_line-1, column_id1, cluster_id, x_, y_, w_, h_);
                words_region.push_back(word_tmp);
            }
		}
    }
    file.close();
    return words;
}

vector<unsigned int> CountFreqWordClusters(vector<word> words, unsigned int column_id)
{
    unsigned int num_clust_words = 0;

    for(int i = 0; i < (int) words.size(); i++)
        if (words[i].get_column_id() == column_id && words[i].get_cluster_id() > num_clust_words)
            num_clust_words = words[i].get_cluster_id();

    num_clust_words++;

    cout << "Number of word clusters found in column "<<column_id<<": " << num_clust_words << endl;

    vector<unsigned int> freq_word_clust(num_clust_words);

    for(int i = 0; i < num_clust_words; i++)
        for(int j = 0; j < (int) words.size(); j++)
            if(words[j].get_column_id() == column_id && words[j].get_cluster_id() == i)
                freq_word_clust[i]++;

    return freq_word_clust;
}

bool CheckCompbl(word word1, word word2, word word3)
{
    if(word1.get_region_id() == word2.get_region_id() && word2.get_region_id() == word3.get_region_id() &&
            word3.get_region_id() == word1.get_region_id())
        if(!(word1.get_line_id() == word2.get_line_id() && word2.get_line_id() == word3.get_line_id() &&
                word3.get_line_id() == word1.get_line_id()))
            if(word1.get_doc_name() == word2.get_doc_name() && word2.get_doc_name() == word3.get_doc_name() &&
                    word3.get_doc_name() == word1.get_doc_name())
                return 1;
            else
                return 0;
        else
            return 0;
    else
        return 0;
}

void CreateDictionary(string dir_docs, vector<word> words, float per_words_voc, unsigned int size_dict_words,
                      string file_voc)
{
    cout<<"Number of words: "<<words.size()<<endl;

	vector<int> word_ind;

	for(int i = 0; i<(int)words.size(); i++)
		word_ind.push_back(i);

	random_shuffle(word_ind.begin(), word_ind.end());

	unsigned int nwords_voc = (unsigned int) round(per_words_voc*words.size());

	word_ind.resize(nwords_voc);

	sort(word_ind.begin(), word_ind.end());

	// Extract DenseSIFT from the images
	Ptr<FeatureDetector> detector(new DenseFeatureDetector(3.0f, 3, 0.3f, 6, 0, true, false));
	//create Sift descriptor extractor
	Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);
	Mat all_descriptors;

	int sz_se = 5;

	Mat se = getStructuringElement( MORPH_ELLIPSE, Size(2*sz_se+1,2*sz_se+1), Point( sz_se, sz_se ) );

	for(int i = 0; i < (int) word_ind.size(); i++)
	{
		int iword = word_ind[i];

		Mat im = imread(dir_docs+words[iword].get_doc_name()+".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		Rect roi(words[iword].bbox().x, words[iword].bbox().y, words[iword].bbox().w, words[iword].bbox().h);
		Mat word_image = im(roi);

		Mat eroded_word_image;

		erode( word_image, eroded_word_image, se );

		Mat bin_word_image;

		threshold(eroded_word_image,bin_word_image,0,255,THRESH_BINARY_INV|THRESH_OTSU);

		cout<<"Computing DenseSIFT for word image: "<<i+1<<". "<<flush;
		//std::cout << " File : " << words[iword].doc_name() << endl;
		vector<KeyPoint> keypoints;

		detector->detect(word_image, keypoints, bin_word_image);

		Mat descriptors;
		//extractor->set("edgeThreshold",10.0);
		extractor->compute(word_image, keypoints, descriptors);
		all_descriptors.push_back(descriptors);

		cout<<"Done."<<endl;
	}

	cout<<"Creating dictionary..."<<flush;
	//Construct BOWKMeansTrainer

	//define Term Criteria
	TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
	//retries number
	int retries=1;
	//necessary flags
	int flags = KMEANS_PP_CENTERS;
	//Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(size_dict_words, tc, retries, flags);
	//cluster the feature vectors
	Mat dictionary = bowTrainer.cluster(all_descriptors);

	FileStorage fs(file_voc, FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();

	cout<<"Done."<<endl;
}

void ComputeBOW( vector<word> words, string dir_docs, string file_voc, string file_words, unsigned int column_id )
{
    const unsigned int descriptor_number_of_orientations = 8;
    const unsigned int descriptor_partitions_x = 4;
    const unsigned int descriptor_partitions_y = 4;
    const unsigned int descriptor_step = 5;
    const unsigned int histogram_levels = 1;
    const unsigned int histogram_initial_x = 9;
    const unsigned int histogram_initial_y = 1;
    const double histogram_degree_x = 1.0;
    const double histogram_degree_y = 1.0;
    const double histogram_power = 0.5;
    const double descriptor_sigma_ratio = 0.2;

	const double descriptor_threshold = 5000;
	const unsigned int number_of_threads = 4;
	const unsigned int number_of_neighbors = 3;

    std::vector<std::tuple<int, int> > descriptor_scales_information;
    IntegralHOG hog;
    HistogramPooling pooling;
    Codebook codebook;
    std::vector<std::tuple<double, unsigned int> > * codes;
    std::list<LocalCodeword> window_codewords;
    std::vector<DescriptorLocation> locations;
    std::vector<unsigned char> * descriptors;
    unsigned int number_of_valid_descriptors;
    std::vector<double> bovw_histogram;

    // Initialize the BoVW structures.
    descriptor_scales_information.resize(3);
    descriptor_scales_information[0] = std::make_pair(20, descriptor_step);
    descriptor_scales_information[1] = std::make_pair(30, descriptor_step);
    descriptor_scales_information[2] = std::make_pair(45, descriptor_step);
    const unsigned int maximum_descriptor_size = 45;
    hog.initialize(descriptor_number_of_orientations, descriptor_partitions_x, descriptor_partitions_y, descriptor_scales_information, descriptor_sigma_ratio);
    codebook.load( file_voc.c_str() );
    pooling.initializePyramid(histogram_levels, histogram_initial_x, histogram_initial_y, histogram_degree_x, histogram_degree_y);
    pooling.set(histogram_power, codebook.getNumberOfCodewords());

    string file_words_ = file_words;

    for (size_t iw = 0; iw < words.size(); iw++) {

        Mat im = imread(dir_docs + words[iw].get_doc_name() + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);

        unsigned int x = words[iw].bbox().x;
        unsigned int y = words[iw].bbox().y;
        unsigned int w = words[iw].bbox().w;
        unsigned int h = words[iw].bbox().h;

        cout << "Computing BOW descriptors for word image: " << iw + 1 << " (" << w << "x" << h << ") / " << words.size() << " in column "
             << column_id << ". " << flush;
#if 0
        Rect roi1(x, y, w, h);
#else
        Rect roi1(x - maximum_descriptor_size / 2, y - maximum_descriptor_size / 2, w + maximum_descriptor_size,
                  h + maximum_descriptor_size);
#endif
        Mat word_image = im(roi1);

        descriptors = 0;
        hog.extract(word_image, locations, descriptors, descriptor_threshold, number_of_threads);
        number_of_valid_descriptors = 0;
        for (unsigned int k = 0; k < locations.size(); ++k)
            if (descriptors[k].size() > 0)
                ++number_of_valid_descriptors;
        printf("Processing %d/%d descriptors. ", number_of_valid_descriptors, (int) locations.size());

        // Encode them into visual words.
        codes = new std::vector<std::tuple<double, unsigned int> >[locations.size()];

#pragma omp parallel num_threads(number_of_threads)
        {
            for (int i = omp_get_thread_num(); i < locations.size(); i += number_of_threads) {
                if (descriptors[i].size() > 0) // If is a valid descriptors...
                    codebook.encode(descriptors[i], number_of_neighbors, codes[i]); // ...encode it,
                else codes[i].resize(0); // otherwise set it as as empty codeword.
            }
        }

        // Gather the visual words that their center falls within the coordinates of the snippet (filter margin visual words).
        window_codewords.clear();
#if 0
        WindowCodewords(0, 0, word_image.size().width, word_image.size().height, hog, locations, codes, window_codewords);
#else
        WindowCodewords(maximum_descriptor_size / 2, maximum_descriptor_size / 2, word_image.size().width -
                                                                                  (maximum_descriptor_size -
                                                                                   maximum_descriptor_size / 2),
                        word_image.size().height - (maximum_descriptor_size - maximum_descriptor_size / 2), hog,
                        locations, codes,
                        window_codewords);
#endif

        // Accumulate the BoVW histogram.
        pooling.accumulate(window_codewords, bovw_histogram);

        Mat bow_desc;
        bow_desc = Mat::zeros(1, (int) bovw_histogram.size(), CV_32F);

        for (int k = 0; k < bovw_histogram.size(); k++)
            bow_desc.at<float>(k) = (float) bovw_histogram[k];

        words[iw].set_bow(bow_desc);

        delete[] descriptors;
        delete[] codes;

        cout << "Done." << endl;
    }

    // writing the words into a binary file

    ofstream outFILE( file_words_.c_str(), ios::out | ios::binary);
    boost::archive::binary_oarchive oa(outFILE);
    size_t num_words = words.size();
    oa << num_words;
    for(size_t i = 0; i < words.size(); i++)
        oa << words[i];

    outFILE.close();
}

/********************************************************/
/********************************************************/
/* BUILD THE DISTANCE TABLE                             */
/********************************************************/
/********************************************************/
void auxiliaryBuildDistanceTable(unsigned int distance_idx, const std::list<WordInformation *> &word_information, double * distance_table, unsigned int number_of_threads)
{
    std::vector<std::vector<std::tuple<double, unsigned int> > > inverted_file;
    std::vector<unsigned int> dimension_frequency;
    std::vector<std::vector<double> > distances;
    std::vector<double> l1_norm, l2_norm;
    unsigned int number_of_dimensions, number_of_elements, idx, idx_distance_table;

    // Get the dimensions of the table.
    printf("[Distance table] Building the inverted file...");
    number_of_elements = (unsigned int)word_information.size();
    number_of_dimensions = 0;
    for (std::list<WordInformation *>::const_iterator begin = word_information.begin(), end = word_information.end(); begin != end; ++begin)
        for (unsigned int dim = 0; dim < (*begin)->getSignature().size(); ++dim)
            number_of_dimensions = std::max(number_of_dimensions, std::get<1>((*begin)->getSignature()[dim]));
    ++number_of_dimensions;
    dimension_frequency.resize(number_of_dimensions);
    std::fill(dimension_frequency.begin(), dimension_frequency.end(), 0);
    for (std::list<WordInformation *>::const_iterator begin = word_information.begin(), end = word_information.end(); begin != end; ++begin)
        for (unsigned int dim = 0; dim < (*begin)->getSignature().size(); ++dim)
            ++dimension_frequency[std::get<1>((*begin)->getSignature()[dim])];
    inverted_file.resize(number_of_dimensions);
    l1_norm.resize(number_of_elements);
    l2_norm.resize(number_of_elements);
    for (unsigned int d = 0; d < number_of_dimensions; ++d)
    {
        inverted_file[d].resize(dimension_frequency[d]);
        dimension_frequency[d] = 0;
    }
    idx_distance_table = idx = 0;
    for (std::list<WordInformation *>::const_iterator begin = word_information.begin(), end = word_information.end(); begin != end; ++begin, ++idx)
    {
        l1_norm[idx] = l2_norm[idx] = 0.0;
        for (unsigned int dim = 0; dim < (*begin)->getSignature().size(); ++dim)
        {
            const unsigned int &dimension = std::get<1>((*begin)->getSignature()[dim]);
            const double &value = std::get<0>((*begin)->getSignature()[dim]);

            inverted_file[dimension][dimension_frequency[dimension]] = std::make_pair(value, idx);
            ++dimension_frequency[dimension];

            l1_norm[idx] += std::abs(value);
            l2_norm[idx] += value * value;
        }
    }
    distances.resize(number_of_threads);
    for (unsigned int t = 0; t < number_of_threads; ++t)
        distances[t].resize(number_of_elements);
    cout<<"Done."<<endl;

    printf("[Distance Table] Calculating the distance table...");
    std::fill(dimension_frequency.begin(), dimension_frequency.end(), 0);
    idx = 0;
    for (std::list<WordInformation *>::const_iterator begin = word_information.begin(), end = word_information.end(); begin != end; ++begin, ++idx)
    {
        const std::vector<std::tuple<double, unsigned int> > &current_histogram = (*begin)->getSignature();
        printf("Calculating %d/%d histogram distances.             \r", idx + 1, number_of_elements);
        std::cout.flush();

        #pragma omp parallel num_threads(number_of_threads)
        {
            const unsigned int thread_id = omp_get_thread_num();

            // Initialize the distance vector.
            std::fill(distances[thread_id].begin(), distances[thread_id].end(), 0.0);
        }

        if (distance_idx == 0) // EUCLIDEAN DISTANCE
        {
            #pragma omp parallel num_threads(number_of_threads)
            {
                const unsigned int thread_id = omp_get_thread_num();

                for (unsigned int j = thread_id; j < current_histogram.size(); j += number_of_threads)
                {
                    const unsigned int &dim = std::get<1>(current_histogram[j]);
                    const std::vector<std::tuple<double, unsigned int> > &current = inverted_file[dim];
                    while ((dimension_frequency[dim] < current.size()) && (std::get<1>(current[dimension_frequency[dim]]) <= idx))
                        ++dimension_frequency[dim];
                    for (unsigned int k = dimension_frequency[dim]; k < current.size(); ++k)
                        distances[thread_id][std::get<1>(current[k])] += std::get<0>(current[k]) * std::get<0>(current_histogram[j]);
                }
            }
            for (unsigned int j = 0; j < number_of_elements; ++j)
            {
                for (unsigned int t = 1; t < number_of_threads; ++t)
                    distances[0][j] += distances[t][j];
                distances[0][j] = l2_norm[j] + l2_norm[idx] - 2 * distances[0][j];
            }
        }
        else if ((distance_idx == 1) || (distance_idx == 2)) // BRAY-CURTIS DISSIMILARITY & HISTOGRAM INTERSECTION DISSIMILARITY
        {
            #pragma omp parallel num_threads(number_of_threads)
            {
                const unsigned int thread_id = omp_get_thread_num();

                for (unsigned int j = thread_id; j < current_histogram.size(); j += number_of_threads)
                {
                    const unsigned int &dim = std::get<1>(current_histogram[j]);
                    const double &value = std::get<0>(current_histogram[j]);
                    const std::vector<std::tuple<double, unsigned int> > &current = inverted_file[dim];
                    while ((dimension_frequency[dim] < current.size()) && (std::get<1>(current[dimension_frequency[dim]]) <= idx))
                        ++dimension_frequency[dim];
                    if (value > 0.0)
                    {
                        for (unsigned int k = dimension_frequency[dim]; k < current.size(); ++k)
                        {
                            const double &current_value = std::get<0>(current[k]);
                            if (current_value > 0)
                                distances[thread_id][std::get<1>(current[k])] += std::min(current_value, value);
                        }
                    }
                    else
                    {
                        for (unsigned int k = dimension_frequency[dim]; k < current.size(); ++k)
                        {
                            const double &current_value = std::get<0>(current[k]);
                            if (current_value < 0)
                                distances[thread_id][std::get<1>(current[k])] += -std::max(current_value, value);
                        }
                    }
                }
            }
            if (distance_idx == 1) // BRAY-CURTIS DISSIMILARITY
            {
                for (unsigned int j = 0; j < number_of_elements; ++j)
                {
                    for (unsigned int t = 1; t < number_of_threads; ++t)
                        distances[0][j] += distances[t][j];
                    if (l1_norm[j] + l1_norm[idx] > 0)
                        distances[0][j] = 1.0 - 2.0 * distances[0][j] / (l1_norm[j] + l1_norm[idx]);
                    else distances[0][j] = 1.0;
                }
            }
            else // HISTOGRAM INTERSECTION DISSIMILARITY
            {
                for (unsigned int j = 0; j < number_of_elements; ++j)
                {
                    for (unsigned int t = 1; t < number_of_threads; ++t)
                        distances[0][j] += distances[t][j];
                    if (l1_norm[j] > 0)
                        distances[0][j] = 1.0 - distances[0][j] / l1_norm[j];
                    else distances[0][j] = 1.0;
                }
            }
        }
        else if (distance_idx == 3) // BRAY-CURTIS DISSIMILARITY CALCULATED USING THE SUM OF ABSOLUTE DIFFERENCES (used to check that gives the same result as Bray-Curtis calculated using min operator).
        {
            #pragma omp parallel num_threads(number_of_threads)
            {
                const unsigned int thread_id = omp_get_thread_num();

                for (unsigned int j = thread_id; j < current_histogram.size(); j += number_of_threads)
                {
                    const unsigned int &dim = std::get<1>(current_histogram[j]);
                    const double &value = std::get<0>(current_histogram[j]);
                    const std::vector<std::tuple<double, unsigned int> > &current = inverted_file[dim];
                    while ((dimension_frequency[dim] < current.size()) && (std::get<1>(current[dimension_frequency[dim]]) <= idx))
                        ++dimension_frequency[dim];
                    for (unsigned int k = dimension_frequency[dim]; k < current.size(); ++k)
                        distances[thread_id][std::get<1>(current[k])] += std::abs(std::get<0>(current[k]) - value) - std::abs(std::get<0>(current[k])) - std::abs(value);
                }
            }
            for (unsigned int j = 0; j < number_of_elements; ++j)
            {
                for (unsigned int t = 1; t < number_of_threads; ++t)
                    distances[0][j] += distances[t][j];
                if (l1_norm[j] + l1_norm[idx] > 0)
                    distances[0][j] = (l1_norm[j] + l1_norm[idx] + distances[0][j]) / (l1_norm[j] + l1_norm[idx]);
                else distances[0][j] = 1.0;
            }
        }
        else throwException("The selected distance has not been implemented yet.");

        for (unsigned int j = idx + 1; j < number_of_elements; ++j, ++idx_distance_table)
            distance_table[idx_distance_table] = distances[0][j];
    }
    printf("Done                                                   \n");
}

void ClusterWords( string dir_docs, string dir_clustered_words, string file_words, string file_cluster_words, unsigned int column_id )
{
    cout<<"Loading words for clustering..."<<flush;

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

    unsigned int distance_idx = 1;
    double upgma_threshold = 0.7; //Distance threshold to create the clusters

    UPGMA m_upgma;
    std::vector<int> cluster_ids;
    double * distance_table;
    std::list<WordInformation *> features;

    for (int i = 0; i < (int) words.size(); i++) {
        WordInformation *word_inf = new WordInformation();
        vector<double> word_signaturei((int) words[i].bow().cols);

        for (int j = 0; j < words[i].bow().cols; j++)
            word_signaturei[j] = words[i].bow().at<int>(j);

        word_inf->setSignature(word_signaturei);
        features.push_back(word_inf);
    }

    distance_table = new double[features.size() * (features.size() - 1) / 2];
    auxiliaryBuildDistanceTable(distance_idx, features, distance_table, 4);

    cout<<"Creating Dendogram..."<<flush;
    m_upgma.generate((unsigned int) features.size(), distance_table, true /*verbose information*/);
    delete[] distance_table;
    features.clear();
    cout<<"Done."<<endl;

    cout<<"Extracting Clusters from Dendogram..."<<flush;
    m_upgma.clusterAccumulated(upgma_threshold, cluster_ids); // TODO: Generate the representatives for each cluster using the dendogram internal distances.
    cout<<"Done"<<endl;

    cout<<"Updating labels of the words in column "<<column_id<<"..."<<flush;
    unsigned int number_clusters = 0;
    for(int i = 0; i < words.size(); i++)
    {
        unsigned int curr_cluster = (unsigned int) cluster_ids[i];
        if(curr_cluster > number_clusters)
            number_clusters = curr_cluster;
        words[i].set_cluster_id(curr_cluster);
    }
    number_clusters++;
    cout<<"Done."<<endl;

	// writing the words into a binary file

	ofstream outFILE( file_words.c_str(), ios::out | ios::binary);
	boost::archive::binary_oarchive oa(outFILE);
	num_words = words.size();
	oa << num_words;
	for(size_t i = 0; i < words.size(); i++)
		oa << words[i];

	outFILE.close();

    cout<<"Computing the centers of each cluster in column "<< column_id << "..." << flush;

    Mat1f C_clusters;
    for (int icl = 0; icl < number_clusters; icl++) {
        Mat1f C;
        for (int iw = 0; iw < words.size(); iw++)
            if (words[iw].get_cluster_id() == icl)
                C.push_back(words[iw].bow());
        reduce(C, C, 0, CV_REDUCE_AVG);
        C_clusters.push_back(C);
    }

    FileStorage fs1(file_cluster_words, FileStorage::WRITE);
    fs1 << "C_clusters" << C_clusters;
    fs1.release();

    cout<<"Done"<<endl;

    // Write the words in folders for each clusters

    // Create the folders for the column and the corresponding subfolders for cluster indices

    cout<<"Creating the directory structures for words..."<<flush;

    if( !boost::filesystem::exists( dir_clustered_words ))
    {
        string cmd_mkdir = "mkdir " + dir_clustered_words;
        int ignore = std::system(cmd_mkdir.c_str());
    }
    else
    {
        string cmd_rm = "rm -r " + dir_clustered_words + "*";
        int ignore = std::system(cmd_rm.c_str());
    }

    for(int icl = 0; icl < number_clusters; icl++)
    {
        string cmd_mkdir = "mkdir " + dir_clustered_words + "Cluster_" + to_string(icl);
        int ignore = std::system(cmd_mkdir.c_str());
    }

    cout<<"Done."<<endl;

    cout<<"Writing the words in folders for each clusters..."<<flush;

    for(int iw = 0; iw < words.size(); iw++)
    {
        Mat im = imread(dir_docs+words[iw].get_doc_name()+".jpg", CV_LOAD_IMAGE_COLOR);
        Rect roi(words[iw].bbox().x, words[iw].bbox().y, words[iw].bbox().w, words[iw].bbox().h);
        Mat word_image = im(roi);

        string res_file_iw = dir_clustered_words + "Cluster_"+ to_string(words[iw].get_cluster_id()) + "/Word_"+std::to_string(iw) + ".jpg";

        imwrite(res_file_iw, word_image);
    }

    cout<<"Done."<<endl;
}