#include "../Headers/WordClustering.h"
#include <stdio.h>
#include <string>
#include <istream>
#include <sstream>
#include <iterator>
#include <iostream>
#include <fstream>
#include <algorithm>

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

template <> const char * CWordClustering::convert<const char *>(const std::string &value) const { return value.c_str(); }
template <> unsigned int CWordClustering::convert<unsigned int>(const std::string &value) const { return atoi(value.c_str()); }
template <> double CWordClustering::convert<double>(const std::string &value) const
{
    double result;
    std::istringstream istr(value);
    istr.imbue(std::locale("C"));
    istr >> result;
    return result;
}

CWordClustering::CWordClustering(void) :
    m_save_to_ontology(false)
{
}


CWordClustering::~CWordClustering(void)
{
    for (std::list<WordInformation *>::iterator begin = m_word_information.begin(), end = m_word_information.end(); begin != end; ++begin)
        delete *begin;
}

std::vector<std::string> tokenize_wc(std::string str, char sep)
{
    //std::stringstream strstr(str);
    // use stream iterators to copy the stream to the vector as whitespace separated strings
    /*std::istream_iterator<std::string> it(strstr);
    std::istream_iterator<std::string> end;
    std::vector<std::string> results(it,end);
    return results;*/
    std::vector<std::string> results;
    std::string aux = str;
    int pos = (int)aux.find_first_of(sep);
    while (pos != (int)std::string::npos)
    {
        results.push_back(aux.substr(0,pos));
        aux = aux.substr(pos+1);
        pos = (int)aux.find_first_of(sep);
    }
    results.push_back(aux);
    return results;
}

void CWordClustering::LoadResults(void)
{
    // TODO: The extension of the images should be specified in the configuration file?
    const char * image_folder_const, * image_extension_const;
    char image_folder[4096], image_extension[256];
    std::string line;
    std::ifstream conffile("./results.xml");
    
    image_folder_const = getParameter<const char *>("ImageFolder", 0);
    if (image_folder_const == 0) sprintf(image_folder, "./");
    else
    {
        unsigned int idx;
        
        for (idx = 0; image_folder_const[idx] != '\0'; ++idx)
            image_folder[idx] = image_folder_const[idx];
        if (idx > 0)
        {
            if (image_folder[idx - 1] != '/') image_folder[idx++] = '/';
            image_folder[idx] = '\0';
        }
        else sprintf(image_folder, "./");
    }
    image_extension_const = getParameter<const char *>("ImageExtension", 0);
    if (image_extension_const == 0) sprintf(image_extension, ".jpg");
    else sprintf(image_extension, ".%s", image_extension_const);
    
    if (conffile.is_open())
    {
        int itline = 0;
        while (conffile.good())
        {
            std::getline(conffile, line);
            if (itline > 0)
            {
                if (line != "\0")
                {
                    std::vector<std::string> lineTokens = tokenize_wc(line, ',');
                    if (lineTokens[0].at(0) != '#') // Avoid commentaries.
                    {
                        std::string key = lineTokens[0];
                        int marker_begin, marker_end;
                        
                        BoundingBox bbox;
                        bbox.xmin = atoi(lineTokens[1].c_str());
                        bbox.ymin = atoi(lineTokens[2].c_str());
                        bbox.w = atoi(lineTokens[3].c_str());
                        bbox.h = atoi(lineTokens[4].c_str());
                        marker_begin = (int)key.find('#');
                        marker_end = (int)key.rfind('_');
                        marker_end = (int)key.rfind('_', marker_end - 1);
                        marker_end = (int)key.rfind('_', marker_end - 1);
                        bbox.filename = image_folder + key.substr(marker_begin + 1, marker_end - marker_begin - 1) + image_extension;
                        m_word_information.push_back(new WordInformation(key, bbox));
                    }
                }
            }
            ++itline;
        }
        conffile.close();
    }
}

void CWordClustering::LoadClusters(int /*numcol*/) // This function is used to load cluster information from disk (used in the first DEMO).
{
    /////// std::string line;
    /////// char idnumcol[2];
    /////// sprintf(idnumcol, "%d", numcol);
    /////// std::string dictionary_name = "./col" + std::string(idnumcol) + ".dat";
    /////// std::ifstream conffile(dictionary_name.c_str());
    /////// if (conffile.is_open())
    /////// {
    ///////     int itline = 0;
    ///////     while (conffile.good())
    ///////     {
    ///////         std::getline (conffile,line);
    ///////         
    ///////         if (line!="\0")
    ///////         {
    ///////             std::vector<std::string> lineTokens = tokenize_wc(line,'/');
    ///////             if (lineTokens[0].at(0)=='#')
    ///////             {
    ///////                 //cout << "Comment: " << line << endl;
    ///////             }
    ///////             else
    ///////             {
    ///////                 /*reading true lines*/
    ///////                 std::string key=lineTokens[1];
    ///////                 m_cluster_representatives[key].push_back(lineTokens[2]);
    ///////                 if(find(m_cluster_ids.begin(), m_cluster_ids.end(), key) == m_cluster_ids.end())
    ///////                 {
    ///////                     m_cluster_ids.push_back(key);
    ///////                 }
    ///////             }
    ///////         }
    ///////         itline++;
    ///////     }
    ///////     conffile.close();
    /////// }
}

void auxiliaryBuildDistanceTable(unsigned int distance_idx, const std::list<WordInformation *> &word_information, double * distance_table, unsigned int number_of_threads)
{
    std::vector<std::vector<std::tuple<double, unsigned int> > > inverted_file;
    std::vector<unsigned int> dimension_frequency;
    std::vector<std::vector<double> > distances;
    std::vector<double> l1_norm, l2_norm;
    unsigned int number_of_dimensions, number_of_elements, idx, idx_distance_table;
    
    // Get the dimensions of the table.
    printf("[Distance table] Building the inverted file...\n");
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
    
    printf("[Distance Table] Calculating the distance table.\n");
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

void CWordClustering::ProcessCluster(int numcell)
{
    // -[ Bag of Visual Words signature variables and parameters ]-----------------------------------------------------------------------------------
    const char * codebook_filename;
    unsigned int histogram_levels, histogram_initial_x, histogram_initial_y, descriptor_number_of_orientations;
    unsigned int descriptor_partitions_x, descriptor_partitions_y, descriptor_step, maximum_descriptor_size, number_of_threads, number_of_neighbors;
    unsigned int distance_idx;
    double histogram_power, descriptor_sigma_ratio, descriptor_threshold, histogram_degree_x, histogram_degree_y, upgma_threshold;
    std::vector<std::tuple<int, int> > descriptor_scales_information;
    IntegralHOG hog;
    HistogramPooling pooling;
    Codebook codebook;
    // -[ Other variables ]--------------------------------------------------------------------------------------------------------------------------
    std::map<std::string, std::vector<std::string> >::iterator search_parameter;
    std::map<std::string, std::list<WordInformation *> > word_page_groups;
    std::vector<int> cluster_ids;
    double * distance_table;
    unsigned int idx;
    char colid[3];
    
    // Get the BoVW parameters from the configuration file (or use the default values when possible).
    codebook_filename = getParameter<const char *>("Codebook_Filename", 0);
    if (codebook_filename == 0)
        throwException("The configuration file does not contain a 'Codebook_Filename' field with the route to the codebook used to encode the descriptors.");
    number_of_neighbors               = getParameter<unsigned int>("Number_Of_Codeword_Neighbors", 1);
    histogram_levels                  = getParameter<unsigned int>("Histogram_Levels", 2);
    histogram_initial_x               = getParameter<unsigned int>("Histogram_Initial_X", 3);
    histogram_initial_y               = getParameter<unsigned int>("Histogram_Initial_Y", 2);
    histogram_degree_x                = getParameter<double>("Histogram_Degree_X", 3);
    histogram_degree_y                = getParameter<double>("Histogram_Degree_Y", 1);
    histogram_power                   = getParameter<double>("Histogram_Power", 0.5);
    descriptor_number_of_orientations = getParameter<unsigned int>("Descriptor_Orientations", 8);
    descriptor_partitions_x           = getParameter<unsigned int>("Descriptor_Partitions_X", 4);
    descriptor_partitions_y           = getParameter<unsigned int>("Descriptor_Partitions_Y", 4);
    descriptor_sigma_ratio            = getParameter<double>("Descriptor_Sigma_Ratio", 0.2);
    descriptor_step                   = getParameter<unsigned int>("Descriptor_Step", 4);
    descriptor_threshold              = getParameter<double>("Descriptor_Threshold", 5000);
    number_of_threads                 = getParameter<unsigned int>("Number_Of_Threads", 1);
    distance_idx                      = getParameter<unsigned int>("Histogram_Distance_ID", 1); // Bray-Curtis as default distance.
    upgma_threshold                   = getParameter<double>("UPGMA_Threshold", 0.7);
	search_parameter = m_cfg_params.cf_params.find("Descriptor_Scales");
    if (search_parameter == m_cfg_params.cf_params.end())
    {
        descriptor_scales_information.resize(3);
        descriptor_scales_information[0] = std::make_pair(20, descriptor_step);
        descriptor_scales_information[1] = std::make_pair(30, descriptor_step);
        descriptor_scales_information[2] = std::make_pair(45, descriptor_step);
    }
    else
    {
        descriptor_scales_information.resize(search_parameter->second.size());
        for (unsigned int i = 0; i < search_parameter->second.size(); ++i)
            descriptor_scales_information[i] = std::make_pair(atoi(search_parameter->second[i].c_str()), descriptor_step);
    }
    maximum_descriptor_size = 0;
    for (unsigned int i = 0; i < descriptor_scales_information.size(); ++i)
        maximum_descriptor_size = std::max(maximum_descriptor_size, (unsigned int)std::get<0>(descriptor_scales_information[i]));
    
    col_to_be_processed = numcell;
    printf("Computing Features on Words\n");
    sprintf(colid, "%d", numcell);
    
    ontAPI->QueryModel("SELECT ?word ?xcoord ?ycoord ?wcoord ?hcoord WHERE{ GRAPH <http://158.109.8.95:3030/qidenus/c1>  {?reg myont:Contains ?lin. ?lin myont:Contains ?word. ?word myont:xcoord ?xcoord. ?word myont:ycoord ?ycoord.?word myont:wcoord ?wcoord.?word myont:hcoord ?hcoord. ?reg myont:columnid ?cid. FILTER(?cid="+std::string(colid)+")}");
    LoadResults();
    printf("Initializing BoVW structures.\n");
#if 1
    printf("BoVW PARAMETERS:\n");
    printf("-------------------------------------------------------------------------\n");
    printf("Descriptor orientations: %d\n", descriptor_number_of_orientations);
    printf("Descriptor partitions: %dx%d\n", descriptor_partitions_x, descriptor_partitions_y);
    printf("Descriptor sigma ratio: %f\n", descriptor_sigma_ratio);
    printf("Descriptor scales:\n");
    for (unsigned int i = 0; i < descriptor_scales_information.size(); ++i)
        printf("   · Scale: %d; Step: %d\n", std::get<0>(descriptor_scales_information[i]), std::get<1>(descriptor_scales_information[i]));
    printf("Codebook filename: %s\n", codebook_filename);
    printf("Histogram levels: %d\n", histogram_levels);
    printf("Histogram initial partitions: %dx%d\n", histogram_initial_x, histogram_initial_y);
    printf("Histogram degree: %fx%f\n", histogram_degree_x, histogram_degree_y);
    printf("Histogram power: %f\n", histogram_power);
    printf("-------------------------------------------------------------------------\n");
#endif
    hog.initialize(descriptor_number_of_orientations, descriptor_partitions_x, descriptor_partitions_y, descriptor_scales_information, descriptor_sigma_ratio);
    codebook.load(codebook_filename);
    pooling.initializePyramid(histogram_levels, histogram_initial_x, histogram_initial_y, histogram_degree_x, histogram_degree_y);
    pooling.set(histogram_power, codebook.getNumberOfCodewords());
    
    printf("Calculating the BoVW histograms of each word snippet.\n");
    
    // Grouping the bounding boxes by image to avoid loading multiple times the same file.
    for (std::list<WordInformation *>::iterator begin = m_word_information.begin(), end = m_word_information.end(); begin != end; ++begin)
        word_page_groups[(*begin)->getBoundingBox().filename].push_back(*begin);
    for (std::map<std::string, std::list<WordInformation *> >::iterator begin = word_page_groups.begin(), end = word_page_groups.end(); begin != end; ++begin)
    {
        unsigned int verbose_word_counter;
        cv::Mat image_page;
        
        printf("Processing image '%s'...\n", begin->first.c_str());
        image_page = cv::imread(begin->first.c_str());
        verbose_word_counter = 0;
        for (std::list<WordInformation *>::iterator begin_word = begin->second.begin(), end_word = begin->second.end(); begin_word != end_word; ++begin_word)
        {
            std::vector<std::tuple<double, unsigned int> > * codes;
            std::list<LocalCodeword> window_codewords;
            std::vector<DescriptorLocation> locations;
            std::vector<unsigned char> * descriptors;
            unsigned int number_of_valid_descriptors;
            std::vector<double> bovw_histogram;
            cv::Rect snippet_rectangle;
            cv::Mat snippet_image;
            int rx, ry, rw, rh;
            
            ++verbose_word_counter;
            
            // Get the coordinates of the bounding box and add the maximum descriptor size to extract the descriptors at the margin of the bounding box.
            rx = (*begin_word)->getBoundingBox().xmin - (int)maximum_descriptor_size / 2;
            ry = (*begin_word)->getBoundingBox().ymin - (int)maximum_descriptor_size / 2;
            rw = rx + (*begin_word)->getBoundingBox().w + (int)maximum_descriptor_size;
            rh = ry + (*begin_word)->getBoundingBox().h + (int)maximum_descriptor_size;
            // Make sure that the coordinates of the snippet fall within the image margins.
            rx = std::max(0, rx);
            ry = std::max(0, ry);
            rw = std::min(image_page.size().width, rw) - rx;
            rh = std::min(image_page.size().height, rh) -  ry;
            snippet_rectangle = cv::Rect(rx, ry, rw, rh);
            if ((snippet_rectangle.width < (int)maximum_descriptor_size) || (snippet_rectangle.height < (int)maximum_descriptor_size))    // If the snippet is to small, ignore it and continue.
                continue;
            
            // Extract the descriptors of the word snippet.
            snippet_image = cv::Mat(image_page, snippet_rectangle);
            descriptors = 0;
            hog.extract(snippet_image, locations, descriptors, descriptor_threshold, number_of_threads);
            number_of_valid_descriptors = 0;
            for (unsigned int k = 0; k < locations.size(); ++k)
                if (descriptors[k].size() > 0)
                    ++number_of_valid_descriptors;
            printf("Word %5d/%5d: Processing %d/%d descriptors.               \r", verbose_word_counter, (int)begin->second.size(), number_of_valid_descriptors, (int)locations.size());
            std::cout.flush();
            
            // Encode them into visual words.
            codes = new std::vector<std::tuple<double, unsigned int> >[locations.size()];
            #pragma omp parallel num_threads(number_of_threads)
            {
                for (unsigned int i = omp_get_thread_num(); i < locations.size(); ++i)
                {
                    if (descriptors[i].size() > 0)                                      // If is a valid descriptors...
                        codebook.encode(descriptors[i], number_of_neighbors, codes[i]); // ...encode it,
                    else codes[i].resize(0);                                            // otherwise set it as as empty codeword.
                }
            }
            
            // Gather the visual words that their center falls within the coordinates of the snippet (filter margin visual words).
            WindowCodewords(maximum_descriptor_size / 2, maximum_descriptor_size / 2, snippet_image.size().width - (maximum_descriptor_size - maximum_descriptor_size / 2), snippet_image.size().height - (maximum_descriptor_size - maximum_descriptor_size / 2), hog, locations, codes, window_codewords);
            
            // Accumulate the BoVW histogram.
            pooling.accumulate(window_codewords, bovw_histogram);
            
            // Store the BoVW histogram.
            (*begin_word)->setSignature(bovw_histogram);
            
            // Free allocated memory.
            delete [] descriptors;
            delete [] codes;
        }
        printf("\n");
    }
    
    printf("Building the distance table.\n");
    distance_table = new double[m_word_information.size() * (m_word_information.size() - 1) / 2];
    auxiliaryBuildDistanceTable(distance_idx, m_word_information, distance_table, number_of_threads);
    
    printf("Creating Dendogram\n");
    m_upgma.generate((unsigned int)m_word_information.size(), distance_table, true /*verbose information*/);
    delete [] distance_table;
    
    printf("Extracting Clusters from Dendogram\n");
    m_upgma.clusterAccumulated(upgma_threshold, cluster_ids); // TODO: Generate the representatives for each cluster using the dendogram internal distances.
    idx = 0;
    for (std::list<WordInformation *>::iterator begin = m_word_information.begin(), end = m_word_information.end(); begin != end; ++begin, ++idx)
        (*begin)->setVisualClusterID(cluster_ids[idx]);
    
    if (m_save_to_ontology)
    {
        printf("Updating Ontology\n");
        SavetoOntology();
    }
}

void CWordClustering::SavetoOntology()
{
    std::map<unsigned int, std::list<WordInformation *> > cluster_ids;
    std::vector<WordInformation *> word_array(m_word_information.size());
    std::vector<int> node_left, node_right;
    std::vector<double> node_distance;
    unsigned int idx;
    char colid[3];
    
    sprintf(colid, "%d", col_to_be_processed);
    idx = 0;
    for (std::list<WordInformation *>::iterator begin = m_word_information.begin(), end = m_word_information.end(); begin != end; ++begin, ++idx)
    {
        cluster_ids[(*begin)->getVisualClusterID()].push_back(*begin);
        word_array[idx] = *begin;
    }
    
    for (std::map<unsigned int, std::list<WordInformation *> >::iterator begin = cluster_ids.begin(), end = cluster_ids.end(); begin != end; ++begin)
    {
        std::string value = "col" + std::string(colid) + "numc" + std::to_string(begin->first);
        ontAPI->AddIndividualToModel("Cluster", value);
        for (std::list<WordInformation *>::iterator begin_list = begin->second.begin(), end_list = begin->second.end(); begin_list != end_list; ++begin_list)
        {
            std::string res = (*begin_list)->getWordURI().substr((*begin_list)->getWordURI().find('#') + 1);
            ontAPI->AddPropertytoIndividual(res, "has_Cluster", value);
        }
    }
    
    m_upgma.flat(node_left, node_right, node_distance);
    for (unsigned int i = 0; i < node_left.size(); ++i)
    {
        std::string node_value;
        
        node_value = "col" + std::string(colid) + "upgma" + std::to_string(i);
        ontAPI->AddIndividualToModel("Term", node_value);
    }
    
    for (unsigned int i = 0; i < node_left.size(); ++i)
    {
        std::string node_value, name_left, name_right;
        
        node_value = "col" + std::string(colid) + "upgma" + std::to_string(i);
        if (node_left[i] >= 0)
            name_left = "col" + std::string(colid) + "upgma" + std::to_string(node_left[i]);
        else name_left = word_array[-node_left[i] - 1]->getWordURI().substr(word_array[-node_left[i] - 1]->getWordURI().find('#') + 1);
        if (node_right[i] >= 0)
            name_right = "col" + std::string(colid) + "upgma" + std::to_string(node_right[i]);
        else name_right = word_array[-node_right[i] - 1]->getWordURI().substr(word_array[-node_right[i] - 1]->getWordURI().find('#') + 1);
        
        ontAPI->AddPropertytoIndividual(node_value, "has_dend_descendent", name_left);
        ontAPI->AddPropertytoIndividual(node_value, "has_dend_descendent", name_right);
        ontAPI->AddPropertytoIndividual(name_left, "has_dend_ascendent", node_value);
        ontAPI->AddPropertytoIndividual(name_right, "has_dend_ascendent", node_value);
        ontAPI->AddValueToAttributetoIndividual(node_value, "distancelinkage", std::to_string(node_distance[i]), "decimal");
    }
}

