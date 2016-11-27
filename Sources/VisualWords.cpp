#include "../Headers/VisualWords.hpp"
#include <iomanip>
#if 0
#include <opencv2/imgproc.hpp>
#else
#include <opencv2/imgproc/imgproc.hpp>
#endif

void IntegralHOG::initialize(unsigned int number_of_orientations, unsigned int partitions_x, unsigned int partitions_y, const std::vector<std::tuple<int, int> > &scales_information, double sigma_ratio)
{
    const double step = 6.283185307179586 / ((double)number_of_orientations);
    
    m_number_of_orientations = number_of_orientations;
    m_partitions_x = partitions_x;
    m_partitions_y = partitions_y;
    m_scales_information.resize(scales_information.size());
    for (unsigned int i = 0; i < scales_information.size(); ++i)
        m_scales_information[i].initialize(std::get<0>(scales_information[i]), std::get<1>(scales_information[i]), sigma_ratio, partitions_x, partitions_y);
    m_cosine.resize(number_of_orientations);
    m_sine.resize(number_of_orientations);
    for (unsigned int i = 0; i < number_of_orientations; ++i)
    {
        double angle = step * (double)i - 6.283185307179586 / 2.0;
        m_cosine[i] = std::cos(angle);
        m_sine[i] = std::sin(angle);
    }
}

void IntegralHOG::initializeRegions(unsigned int width, unsigned int height, std::vector<DescriptorLocation> &locations) const
{
    const unsigned int nscales = (unsigned int)m_scales_information.size();
    std::vector<unsigned int> number_of_regions_x(nscales), number_of_regions_y(nscales), region_offset_x(nscales), region_offset_y(nscales);
    unsigned int number_of_regions, total_number_of_regions;
    
    // Calculate the number of local regions ..............................................
    total_number_of_regions = 0;
    for (unsigned int scale = 0; scale < nscales; ++scale)
    {
        const unsigned int step = m_scales_information[scale].getStep();
        const unsigned int region_width = m_scales_information[scale].getWindowWidth();
        const unsigned int region_height = m_scales_information[scale].getWindowHeight();
        number_of_regions_x[scale] = ((step > 0) && (width  > region_width ))?((width  - region_width ) / step + 1):0;
        number_of_regions_y[scale] = ((step > 0) && (height > region_height))?((height - region_height) / step + 1):0;
        if ((number_of_regions_x[scale] > 0) && (number_of_regions_y[scale] > 0))
        {
            region_offset_x[scale] = (width  - ((number_of_regions_x[scale] - 1) * step + region_width )) / 2;
            region_offset_y[scale] = (height - ((number_of_regions_y[scale] - 1) * step + region_height)) / 2;
            number_of_regions = number_of_regions_x[scale] * number_of_regions_y[scale];
        }
        else region_offset_x[scale] = region_offset_y[scale] = number_of_regions = 0;
        total_number_of_regions += number_of_regions;
    }
    
    // Calculate the region locations .....................................................
    locations.resize(total_number_of_regions);
    if (total_number_of_regions > 0)
    {
        for (unsigned int scale = 0, index = 0; scale < m_scales_information.size(); ++scale)
        {
            const unsigned int step = m_scales_information[scale].getStep();
            for (unsigned int y = 0, iy = region_offset_y[scale]; y < number_of_regions_y[scale]; ++y, iy += step)
                for (unsigned int x = 0, ix = region_offset_x[scale]; x < number_of_regions_x[scale]; ++x, ix += step, ++index)
                    locations[index].set(ix, iy, (unsigned short)scale);
        }
    }
}

void IntegralHOG::extract(const cv::Mat &image, std::vector<DescriptorLocation> &locations, std::vector<unsigned char> * &descriptors, double descriptor_threshold, unsigned int number_of_threads) const
{
    const unsigned int number_of_dimensions = m_partitions_x * m_partitions_y * m_number_of_orientations;
    const unsigned int width = image.size().width;
    const unsigned int height = image.size().height;
    cv::Mat image_work(cv::Size(width, height), CV_MAKETYPE(cv::DataType<unsigned char>::depth, 1));
    cv::Mat integral_work(cv::Size(width + 1, height + 1), CV_MAKETYPE(cv::DataType<int>::depth, 1));
    cv::Mat gradient_dx(cv::Size(width, height), CV_MAKETYPE(cv::DataType<short>::depth, 1));
    cv::Mat gradient_dy(cv::Size(width, height), CV_MAKETYPE(cv::DataType<short>::depth, 1));
    std::vector<short *> image_features_ptrs(m_number_of_orientations);
    cv::Mat * image_features, * integral_features;
    
    if (descriptors == 0)
    {
        initializeRegions(width, height, locations);
        descriptors = new std::vector<unsigned char>[locations.size()];
    }
    
    if ((image.channels() == 3) || (image.channels() == 4))
        cv::cvtColor(image, image_work, CV_BGR2GRAY);
    else { int from_to[] = { 0, 0 }; mixChannels(&image, 1, &image_work, 1, from_to, 1); }
    image_features = new cv::Mat[m_number_of_orientations];
    for (unsigned int i = 0; i < m_number_of_orientations; ++i)
        image_features[i].create(height, width, CV_MAKETYPE(cv::DataType<short>::depth, 1));
    integral_features = new cv::Mat[m_number_of_orientations * m_scales_information.size()];
    for (unsigned int i = 0; i < m_number_of_orientations * m_scales_information.size(); ++i)
        integral_features[i].create(height + 1, width + 1, CV_MAKETYPE(cv::DataType<int>::depth, 1));
    
    cv::integral(image_work, integral_work, CV_32S);
    for (unsigned int i = 0, k = 0; i < m_scales_information.size(); ++i)
    {
        // Extract the gradient of the image with the current feature sizes.
        ImageGradient<int, short>(integral_work, m_scales_information[i].getFeatureSize(), gradient_dx, gradient_dy, number_of_threads);
        // Calculate the oriented gradient features for each pixels.
        for (unsigned int y = 0; y < height; ++y)
        {
            const short * dx_ptr = (short *)gradient_dx.ptr(y);
            const short * dy_ptr = (short *)gradient_dy.ptr(y);
            for (unsigned int o = 0; o < m_number_of_orientations; ++o)
                image_features_ptrs[o] = (short *)image_features[o].ptr(y);
            for (unsigned int x = 0; x < width; ++x)
            {
                for (unsigned int o = 0; o < m_number_of_orientations; ++o)
                {
                    *image_features_ptrs[o] = std::max((short)0, (short)((double)*dx_ptr * m_cosine[o] + (double)*dy_ptr * m_sine[o]));
                    ++image_features_ptrs[o];
                }
                ++dx_ptr;
                ++dy_ptr;
            }
        }
        // Create the feature integral images.
        for (unsigned int o = 0; o < m_number_of_orientations; ++o, ++k)
            ImageIntegral<short, int>(image_features[o], integral_features[k]);
    }
    // -----------------------------------[ scales ]-----------------------------------
    // features features features features features features features features features 
    // ++++++++ ++++++++ ++++++++ ++++++++ ++++++++ ++++++++ ++++++++ ++++++++ ++++++++ 
    #pragma omp parallel num_threads(number_of_threads)
    {
        std::vector<double> current_descriptor(number_of_dimensions);
        double l2_norm, l1_norm;
        
        for (unsigned int idx = omp_get_thread_num(); idx < locations.size(); idx += number_of_threads)
        {
            const unsigned int current_x = locations[idx].getX();
            const unsigned int current_y = locations[idx].getY();
            const unsigned int current_width  = m_scales_information[locations[idx].getScale()].getPartitionWidth();
            const unsigned int current_height = m_scales_information[locations[idx].getScale()].getPartitionHeight();
            const unsigned int ci_idx = m_number_of_orientations * locations[idx].getScale();
            // ----------------------------------------------------------------------------------------------------------------------------------
            l1_norm = l2_norm = 0.0;
            for (unsigned int py = 0, cy = current_y, bin_idx = 0; py < m_partitions_y; ++py, cy += current_height)
            {
                for (unsigned int px = 0, cx = current_x; px < m_partitions_x; ++px, cx += current_width)
                {
                    for (unsigned int feature_bin = 0; feature_bin < m_number_of_orientations; ++feature_bin)
                    {
                        const int * __restrict__ top_ptr    = (const int *)integral_features[ci_idx + feature_bin].ptr(cy) + cx;
                        const int * __restrict__ bottom_ptr = (const int *)integral_features[ci_idx + feature_bin].ptr(cy + current_height) + cx;
                        current_descriptor[bin_idx] = (double)(top_ptr[0] + bottom_ptr[current_width] - top_ptr[current_width] - bottom_ptr[0]);
                        l2_norm += current_descriptor[bin_idx] * current_descriptor[bin_idx];
                        l1_norm += current_descriptor[bin_idx];
                        ++bin_idx;
                    }
                }
            }
            // ----------------------------------------------------------------------------------------------------------------------------------
            if (l1_norm >= descriptor_threshold)
            {
                l2_norm = std::sqrt(l2_norm);
                descriptors[idx].resize(number_of_dimensions);
                for (unsigned int d = 0; d < number_of_dimensions; ++d)
                    descriptors[idx][d] = (unsigned char)(255.0 * current_descriptor[d] / l2_norm);
            }
            else descriptors[idx].resize(0);
        }
    }
    
    delete [] image_features;
    delete [] integral_features;
}

void Codebook::load(const char * filename)
{
    unsigned int number_of_codewords;
    std::ifstream file(filename);
    
    if (!file.is_open()) throwException("Codebook file '%s' cannot be loaded.", filename);
    file >> number_of_codewords >> m_number_of_dimensions;
    m_codewords.resize(number_of_codewords);
    m_codewords_norm.resize(number_of_codewords);
    for (unsigned int i = 0; i < number_of_codewords; ++i)
    {
        double l2_norm;
        
        l2_norm = 0;
        m_codewords[i].resize(m_number_of_dimensions);
        for (unsigned int j = 0; j < m_number_of_dimensions; ++j)
        {
            file >> m_codewords[i][j];
            l2_norm += m_codewords[i][j] * m_codewords[i][j];
        }
        m_codewords_norm[i] = l2_norm;
    }
    file.close();
}

void Codebook::set(const std::vector<std::vector<double> > &codewords)
{
    if (codewords.size() > 0)
    {
        m_number_of_dimensions = (unsigned int)codewords[0].size();
        for (unsigned int i = 1; i < codewords.size(); ++i)
            if ((unsigned int)codewords[i].size() != m_number_of_dimensions)
                throwException("All codewords must have the same number of dimensions.");
        
        m_codewords.resize(codewords.size());
        m_codewords_norm.resize(codewords.size());
        for (unsigned int i = 0; i < codewords.size(); ++i)
        {
            double l2_norm;
            
            l2_norm = 0.0;
            m_codewords[i].resize(m_number_of_dimensions);
            for (unsigned int d = 0; d < m_number_of_dimensions; ++d)
            {
                m_codewords[i][d] = codewords[i][d];
                l2_norm += codewords[i][d] * codewords[i][d];
            }
            m_codewords_norm[i] = l2_norm;
        }
    }
    else { m_codewords.resize(0); m_codewords_norm.resize(0); m_number_of_dimensions = 0; }
}

void Codebook::encode(const std::vector<unsigned char> &descriptor, unsigned int number_of_neighbors, std::vector<std::tuple<double, unsigned int> > &code) const
{
    if ((m_codewords.size() > 0) && (number_of_neighbors <= m_codewords.size()))
    {
        if (number_of_neighbors > 1)
        {
            const double beta = 1e-7;
            std::vector<std::tuple<double, unsigned int> > neighbors_queue(number_of_neighbors);
            double diagonal_factor, weight_sum;
            
            for (unsigned int i = 0; i < number_of_neighbors; ++i)
                neighbors_queue[i] = std::make_pair(distance(descriptor, i), i);
            std::make_heap(neighbors_queue.begin(), neighbors_queue.end());
            for (unsigned int i = number_of_neighbors; i < m_codewords.size(); ++i)
            {
                double current_distance;
                
                current_distance = distance(descriptor, i);
                if (current_distance < std::get<0>(neighbors_queue[0]))
                {
                    std::pop_heap(neighbors_queue.begin(), neighbors_queue.end());
                    neighbors_queue[number_of_neighbors - 1] = std::make_pair(current_distance, i);
                    std::push_heap(neighbors_queue.begin(), neighbors_queue.end());
                }
            }
            std::sort_heap(neighbors_queue.begin(), neighbors_queue.end());
                      /*# Rows              , # Cols             , Type*/
            cv::Mat z(number_of_neighbors, m_number_of_dimensions, CV_64F);
            cv::Mat C(number_of_neighbors,    number_of_neighbors, CV_64F);
            cv::Mat w(number_of_neighbors,                      1, CV_64F);
            for (unsigned int n = 0; n < number_of_neighbors; ++n)
            {
                const unsigned int idx = std::get<1>(neighbors_queue[n]);
                double * __restrict__ z_row = (double *)z.ptr(n);
                for (unsigned int d = 0; d < m_number_of_dimensions; ++d)
                    z_row[d] = (double)descriptor[d] - m_codewords[idx][d];
            }
            cv::mulTransposed(z, C, false);                                 // C = z * z';
            diagonal_factor = beta * cv::trace(C)[0];                       // C = C + eye(number_of_neighbors) * beta * trace(C);
            for (unsigned int k = 0; k < number_of_neighbors; ++k)
                C.at<double>(k, k) += diagonal_factor;
            cv::solve(C, cv::Mat::ones(number_of_neighbors, 1, CV_64F), w); // w = C \ ones(number_of_neighbors, 1);
            weight_sum = 0;                                                 // w = w / sum(w);
            for (unsigned int k = 0; k < number_of_neighbors; ++k)
                weight_sum += w.at<double>(k, 0);
            
            code.resize(number_of_neighbors);
            for (unsigned int i = 0; i < number_of_neighbors; ++i)
                code[i] = std::make_pair(w.at<double>(i, 0) / weight_sum, std::get<1>(neighbors_queue[i]));
        }
        else
        {
            unsigned int minimum_index;
            double minimum_distance;
            
            minimum_distance = distance(descriptor, 0);
            minimum_index = 0;
            for (unsigned int i = 1; i < m_codewords.size(); ++i)
            {
                double current_distance;
                
                current_distance = distance(descriptor, i);
                if (current_distance < minimum_distance)
                {
                    minimum_distance = current_distance;
                    minimum_index = i;
                }
            }
            
            code.resize(1);
            code[0] = std::make_pair(1.0, minimum_index);
        }
    }
}

double Codebook::distance(const std::vector<unsigned char> &descriptor, unsigned int idx) const
{
    double dot;
    
    if (descriptor.size() != m_number_of_dimensions)
        throwException("The dimensionality of the descriptor is different than codeword vectors (%d != %d).", (unsigned int)descriptor.size(), m_number_of_dimensions);
    dot = 0;
    for (unsigned int d = 0; d < m_number_of_dimensions; ++d)
        dot += (double)descriptor[d] * m_codewords[idx][d];
    
    return m_codewords_norm[idx] - 2 * dot;
}

void WindowCodewords(int x0, int y0, int x1, int y1, const IntegralHOG &descriptor, const std::vector<DescriptorLocation> &locations, const std::vector<std::tuple<double, unsigned int> > * codes, std::list<LocalCodeword> &window_codewords)
{
    int width = x1 - x0;
    int height = y1 - y0;
    for (unsigned int i = 0; i < locations.size(); ++i)
    {
        if (codes[i].size() > 0)
        {
            float cx, cy;
            
            cx = (float)((int)(locations[i].getX() + descriptor.getScaleInformation(locations[i].getScale()).getWindowWidth() / 2)  - x0) / (float)width;
            cy = (float)((int)(locations[i].getY() + descriptor.getScaleInformation(locations[i].getScale()).getWindowHeight() / 2) - y0) / (float)height;
            if ((cx >= 0.0f) && (cy >= 0.0f) && (cx < 1.0f) && (cy < 1.0f))
                window_codewords.push_back(LocalCodeword(cx, cy, &codes[i]));
        }
    }
}

void HistogramPooling::initializePyramid(unsigned int number_of_levels, unsigned int initial_partitions_x, unsigned int initial_partitions_y, double partition_degree_x, double partition_degree_y)
{
    double degree_x, degree_y;
    
    m_coordinates.resize(number_of_levels);
    m_number_of_spatial_bins = 0;
    degree_x = (double)initial_partitions_x;
    degree_y = (double)initial_partitions_y;
    for (unsigned int level = 0; level < number_of_levels; ++level)
    {
        unsigned int partitions_x, partitions_y, p;
        
        partitions_x = (unsigned int)round(degree_x);
        partitions_y = (unsigned int)round(degree_y);
        p = partitions_x * partitions_y;
        m_number_of_spatial_bins += p;
        m_coordinates[level].resize(p);
        p = 0;
        for (unsigned int py = 0; py < partitions_y; ++py)
            for (unsigned int px = 0; px < partitions_x; ++px, ++p)
                m_coordinates[level][p].set((float)px / (float)partitions_x, (float)py / (float)partitions_y, (float)(px + 1) / (float)partitions_x, (float)(py + 1) / (float)partitions_y);
        degree_x *= partition_degree_x;
        degree_y *= partition_degree_y;
    }
}

void HistogramPooling::accumulate(std::list<LocalCodeword> &window_codewords, std::vector<double> &histogram) const
{
    double l2_norm;
    
    histogram.resize(m_codebook_size * m_number_of_spatial_bins);
    std::fill(histogram.begin(), histogram.end(), 0);
    l2_norm = 0.0;
    for (unsigned int lvl = 0, offset = 0, begin = 0; lvl < m_coordinates.size(); ++lvl, begin = offset)
    {
        double l1_norm;
        
        for (unsigned int bin = 0; bin < m_coordinates[lvl].size(); ++bin, offset += m_codebook_size)
        {
            for (std::list<LocalCodeword>::const_iterator cbegin = window_codewords.begin(), cend = window_codewords.end(); cbegin != cend; ++cbegin)
            {
                if (m_coordinates[lvl][bin].isInside(cbegin->getX(), cbegin->getY()))
                {
                    const std::vector<std::tuple<double, unsigned int> > &current = cbegin->getCodewords();
                    const double factor = (double)m_coordinates[lvl][bin].getArea();
                    for (unsigned int m = 0; m < current.size(); ++m)
                        histogram[offset + std::get<1>(current[m])] += std::get<0>(current[m]) * factor;
                }
            }
        }
        l1_norm = 0.0;
        for (unsigned int k = begin; k < offset; ++k)
            l1_norm += std::abs(histogram[k]);
        for (unsigned int k = begin; k < offset; ++k)
        {
            histogram[k] /= l1_norm;
            l2_norm += histogram[k] * histogram[k];
        }
    }
    l2_norm = std::sqrt(l2_norm);
    for (unsigned int i = 0; i < histogram.size(); ++i)
        histogram[i] /= l2_norm;
    if (m_power_factor != 1.0)
    {
        l2_norm = 0;
        for (unsigned int i = 0; i < histogram.size(); ++i)
        {
            if (histogram[i] >= 0.0) histogram[i] = std::pow(histogram[i], m_power_factor);
            else histogram[i] = -std::pow(-histogram[i], m_power_factor);
            l2_norm += histogram[i] * histogram[i];
        }
        l2_norm = std::sqrt(l2_norm);
        for (unsigned int i = 0; i < histogram.size(); ++i)
            histogram[i] /= l2_norm;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

UPGMA::UPGMA(const UPGMA &other) :
    m_leafs((other.m_number_of_leafs > 0)?new NodeUPGMA[other.m_number_of_leafs]:0),
    m_number_of_leafs(other.m_number_of_leafs),
    m_root((other.m_root != 0)?new NodeUPGMA(*other.m_root):0)
{
    for (unsigned int i = 0; i < other.m_number_of_leafs; ++i)
        m_leafs[i] = other.m_leafs[i];
}

UPGMA::~UPGMA(void)
{
    if (m_root != 0) { m_root->clear(); delete m_root; }
    if (m_leafs != 0) delete [] m_leafs;
}

UPGMA& UPGMA::operator=(const UPGMA &other)
{
    if (this != &other)
    {
        if (m_root != 0) { m_root->clear(); delete m_root; }
        if (m_leafs != 0) delete [] m_leafs;
        if (other.m_number_of_leafs > 0)
        {
            m_leafs = new NodeUPGMA[other.m_number_of_leafs];
            m_number_of_leafs = other.m_number_of_leafs;
            for (unsigned int i = 0; i < other.m_number_of_leafs; ++i)
                m_leafs[i] = other.m_leafs[i];
        }
        else
        {
            m_leafs = 0;
            m_number_of_leafs = 0;
        }
        m_root = (other.m_root != 0)?new NodeUPGMA(*other.m_root):0;
    }
    return *this;
}

void UPGMA::generate(unsigned int number_of_elements, const double * distances, bool verbose)
{
    const unsigned int maximum_number_of_nn = number_of_elements * (number_of_elements - 1) / 2;
    std::list<NodeUPGMA *> nn_table;
    
    // Free current tree information ..................................................................................
    if (m_leafs != 0) delete [] m_leafs;
    if (m_root != 0) { m_root->clear(); delete m_root; }
    if (number_of_elements == 0)
    {
        m_leafs = 0;
        m_number_of_leafs = 0;
        m_root = 0;
        return;
    }
    
    // Initialize the tree structures .................................................................................
    if (verbose) printf("Building the initial distance table (%d elements).\n", maximum_number_of_nn);
    m_leafs = new NodeUPGMA[number_of_elements];
    m_number_of_leafs = number_of_elements;
    for (unsigned int i = 0, k = 0; i < number_of_elements; ++i)
        for (unsigned int j = i + 1; j < number_of_elements; ++j, ++k)
            nn_table.push_back(new NodeUPGMA(&m_leafs[i], &m_leafs[j], distances[k]));
    if (verbose) printf("Done.\n");
    
    if (verbose) printf("Aggregating the descriptors:\n");
    while (nn_table.size() > 1) // Build the tree.
    {
        typename std::list<NodeUPGMA *>::iterator minimum_iter;
        NodeUPGMA * minimum_ptr;
        
        // 1) Search for the shortest distance.
        minimum_iter = nn_table.begin();
        for (typename std::list<NodeUPGMA *>::iterator begin = nn_table.begin(), end = nn_table.end(); begin != end; ++begin)
            if ((*begin)->getDistance() < (*minimum_iter)->getDistance()) minimum_iter = begin;
        minimum_ptr = *minimum_iter;
        nn_table.erase(minimum_iter);
        
        // 2) Update the distances of all elements which interact with this element.
        std::map<NodeUPGMA *, NodeUPGMA *> nodes_to_merge;
        for (typename std::list<NodeUPGMA *>::iterator begin = nn_table.begin(), end = nn_table.end(); begin != end;)
        {
            int neighbor;
            
            neighbor = (*begin)->isNeighbor(minimum_ptr);
            if (neighbor != 0)
            {
                typename std::map<NodeUPGMA *, NodeUPGMA *>::iterator search;
                
                if (neighbor == -1) search = nodes_to_merge.find((*begin)->getRight());
                else search = nodes_to_merge.find((*begin)->getLeft());
                
                if (search == nodes_to_merge.end())
                {
                    if (neighbor == -1) nodes_to_merge[(*begin)->getRight()] = *begin;
                    else nodes_to_merge[(*begin)->getLeft()] = *begin;
                    begin = nn_table.erase(begin);
                }
                else
                {
                    (*begin)->merge(search->second, minimum_ptr);
                    delete search->second;
                    nodes_to_merge.erase(search);
                    ++begin;
                }
            }
            else ++begin;
        }
        minimum_ptr->setDistance(minimum_ptr->getDistance() / 2);
        
        if (verbose)
        {
            printf("[Progress] Min. dist=%f | Remaining distance pairs: %d                                 \r", minimum_ptr->getDistance(), (unsigned int)nn_table.size());
            std::cout.flush();
        }
    }
    m_root = (*nn_table.begin());
    m_root->setDistance(m_root->getDistance() / 2);
    if (verbose) printf("Done                                                                            \n");
}

void UPGMA::NodeUPGMA::flatten(std::vector<int> &node_left, std::vector<int> &node_right, std::vector<unsigned int> &node_counter, std::vector<double> &node_distance, unsigned int &position, const NodeUPGMA * base) const
{
    const unsigned int current_position = position;
    if (m_left->m_left != 0)
    {
        node_left[current_position] = ++position;
        m_left->flatten(node_left, node_right, node_counter, node_distance, position, base);
    }
    else node_left[current_position] = -(int)(m_left - base) - 1;
    if (m_right->m_left != 0)
    {
        node_right[current_position] = ++position;
        m_right->flatten(node_left, node_right, node_counter, node_distance, position, base);
    }
    else node_right[current_position] = -(int)(m_right - base) - 1;
    node_distance[current_position] = m_distance;
    node_counter[current_position] = m_counter;
}

void UPGMA::save(const char * filename) const
{
    if (m_root != 0)
    {
        unsigned int number_of_nodes, position = 0;
        std::vector<unsigned int> node_counter;
        std::vector<int> node_left, node_right;
        std::vector<double> node_distance;
        std::list<std::tuple<const NodeUPGMA *, unsigned int> > stack;
        
        number_of_nodes = m_root->getNumberOfInnerNodes();
        node_left.resize(number_of_nodes);
        node_right.resize(number_of_nodes);
        node_counter.resize(number_of_nodes);
        node_distance.resize(number_of_nodes);
        
        m_root->flatten(node_left, node_right, node_counter, node_distance, position, m_leafs);
        ///// // DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
        ///// // DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
        ///// // DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
        ///// for (unsigned int i = 0; i < number_of_nodes; ++i)
        /////     printf(" %5d", node_left[i]);
        ///// std::cout << std::endl;
        ///// for (unsigned int i = 0; i < number_of_nodes; ++i)
        /////     printf(" %5d", node_right[i]);
        ///// std::cout << std::endl;
        ///// for (unsigned int i = 0; i < number_of_nodes; ++i)
        /////     printf(" %5.2f", node_distance[i]);
        ///// std::cout << std::endl;
        ///// // DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
        ///// // DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
        ///// // DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
        
        std::ofstream file(filename);
        if (!file.is_open()) throwException("Cannot access file '%s'.", filename);
        
        file << number_of_nodes << std::endl;
        for (unsigned int i = 0; i < number_of_nodes; ++i)
            file << node_left[i] << " " << node_right[i] << " " << node_counter[i] << " " << std::setprecision(25) << node_distance[i] << std::endl;
        
        file.close();
    }
}

void UPGMA::load(const char * filename)
{
    unsigned int number_of_nodes;
    
    if (m_root != 0) { m_root->clear(); delete m_root; }
    if (m_leafs != 0) delete [] m_leafs;
    
    std::ifstream file(filename);
    if (!file.is_open()) throwException("Cannot access file '%s'.", filename);
    
    file >> number_of_nodes;
    if (number_of_nodes > 0)
    {
        NodeUPGMA * * node_ptrs;
        
        // Initialize the internal structures.
        node_ptrs = new NodeUPGMA * [number_of_nodes];
        m_leafs = new NodeUPGMA[number_of_nodes + 1];
        m_number_of_leafs = number_of_nodes + 1;
        for (unsigned int i = 0; i < number_of_nodes; ++i)
            node_ptrs[i] = new NodeUPGMA();
        
        for (unsigned int i = 0; i < number_of_nodes; ++i)
        {
            int node_left, node_right;
            unsigned int node_counter;
            double node_distance;
            
            file >> node_left >> node_right >> node_counter >> node_distance;
            node_ptrs[i]->set((node_left >= 0)?(node_ptrs[node_left]):(&m_leafs[-(node_left + 1)]), (node_right >= 0)?(node_ptrs[node_right]):(&m_leafs[-(node_right + 1)]), node_distance, node_counter);
        }
        
        // Set the root node and free allocated memory.
        m_root = node_ptrs[0];
        delete [] node_ptrs;
    }
    else
    {
        m_root = 0;
        m_leafs = 0;
        m_number_of_leafs = 0;
    }
    
    file.close();
}

/////void UPGMA::draw(Image<unsigned char> &image) const
/////{
/////    srv::Draw::Pencil<unsigned char> pencil("FF0000", 0, 1, true);
/////    
/////    image.setValue(255);
/////    for (unsigned int i = 0; i < m_number_of_leafs; ++i)
/////        srv::Draw::Circle(image, i * (image.getWidth() - 20) / (m_number_of_leafs - 1) + 10, image.getHeight() - 15, 5, pencil);
/////    unsigned int position = 0;
/////    m_root->draw(image, pencil, position, (double)(image.getWidth() - 20) / (double)(m_number_of_leafs - 1), m_root->getDistance(), m_leafs);
/////}

/////std::tuple<int, int> UPGMA::NodeUPGMA::draw(srv::Image<unsigned char> &image, srv::Draw::Pencil<unsigned char> &pencil, unsigned int &position, double factor, double maximum_distance, const NodeUPGMA * base) const
/////{
/////    std::tuple<int, int> left, right;
/////    int current_height;
/////    
/////    if (m_left->m_left != 0)
/////        left = m_left->draw(image, pencil, position, factor, maximum_distance, base);
/////    else
/////    {
/////        left = std::tuple<int, int>(round((double)position * factor) + 10, image.getHeight() - 25);
/////        srv::Draw::Text(image, std::get<0>(left) - 2, image.getHeight() - 17, srv::Draw::Pencil<unsigned char>("000000", 0, 1, false), srv::Draw::Font(srv::Draw::TINY_FONT_5x7), "%d", m_left - base);
/////        ++position;
/////    }
/////    if (m_right->m_left != 0)
/////        right = m_right->draw(image, pencil, position, factor, maximum_distance, base);
/////    else
/////    {
/////        right = std::tuple<int, int>(round((double)position * factor) + 10, image.getHeight() - 25);
/////        srv::Draw::Text(image, std::get<0>(right) - 2, image.getHeight() - 17, srv::Draw::Pencil<unsigned char>("000000", 0, 1, false), srv::Draw::Font(srv::Draw::TINY_FONT_5x7), "%d", m_right - base);
/////        ++position;
/////    }
/////    current_height = (int)round((1.0 - m_distance / maximum_distance) * (double)(image.getHeight() - 50) + 25.0);
/////    
/////    srv::Draw::Line(image, std::get<0>(left), std::get<1>(left), std::get<0>(left), current_height, pencil);
/////    srv::Draw::Line(image, std::get<0>(right), std::get<1>(right), std::get<0>(right), current_height, pencil);
/////    srv::Draw::Line(image, std::get<0>(left), current_height, std::get<0>(right), current_height, pencil);
/////    srv::Draw::Text(image, std::get<0>(left) + 2, current_height + 8, srv::Draw::Pencil<unsigned char>("000000", 0, 1, false), srv::Draw::Font(srv::Draw::TINY_FONT_5x7), "%f", m_distance);
/////    
/////    return std::tuple<int, int>((std::get<0>(left) + std::get<0>(right)) / 2, current_height);
/////}

unsigned int UPGMA::NodeUPGMA::clusterAccumulated(unsigned int cluster_id, double threshold, std::vector<int> &cluster_ids, const NodeUPGMA * base) const
{
    if (m_left != 0)
    {
        if (m_distance > threshold)
        {
            cluster_id = m_left->clusterAccumulated(cluster_id, threshold, cluster_ids, base);
            ++cluster_id;
            return m_right->clusterAccumulated(cluster_id, threshold, cluster_ids, base);
        }
        else
        {
            m_left->clusterAccumulated(cluster_id, threshold, cluster_ids, base);
            m_right->clusterAccumulated(cluster_id, threshold, cluster_ids, base);
            return cluster_id;
        }
    }
    else
    {
        cluster_ids[this - base] = cluster_id;
        return cluster_id;
    }
}

double UPGMA::clusterAccumulated(double ratio, std::vector<int> &cluster_ids) const
{
    if (m_root != 0)
    {
        const double threshold = m_root->getDistance() * ratio;
        cluster_ids.resize(m_number_of_leafs);
        std::fill(cluster_ids.begin(), cluster_ids.end(), -1);
        m_root->clusterAccumulated(0, threshold, cluster_ids, m_leafs);
        return threshold;
    }
    else return -1.0;
}

