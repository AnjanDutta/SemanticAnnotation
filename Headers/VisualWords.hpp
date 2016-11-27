/** Classes and functions used to generate visual words.
 */

#ifndef __VISUAL_WORDS_HEADER_FILE__
#define __VISUAL_WORDS_HEADER_FILE__

#include "ImageProcessing.hpp"

/// Class which stores the location of the densely sampled regions where the descriptors are generated from.
class DescriptorLocation
{
public:
    /// Default constructor.
    DescriptorLocation(void) : m_x(0), m_y(0), m_scale(0) {}
    /** Constructor which sets the location of the densely sampled region.
     *  \param[in] x X-coordinate of the top-left corner of the region.
     *  \param[in] y Y-coordinate of the top-left corner of the region.
     *  \param[in] scale identifier of the scale of the region.
     */
    DescriptorLocation(unsigned int x, unsigned int y, unsigned short scale) : m_x(x), m_y(y), m_scale(scale) {}
    /** Function which sets the location of the densely sampled region.
     *  \param[in] x X-coordinate of the top-left corner of the region.
     *  \param[in] y Y-coordinate of the top-left corner of the region.
     *  \param[in] scale identifier of the scale of the region.
     */
    inline void set(unsigned int x, unsigned int y, unsigned short scale) { m_x = x; m_y = y; m_scale = scale; }
    
    /// Returns the X-coordinate of the top-left corner of the region.
    inline unsigned int getX(void) const { return m_x; }
    /// Sets the X-coordinate of the top-left corner of the region.
    inline void setX(unsigned int x) { m_x = x; }
    /// Returns the Y-coordinate of the top-left corner of the region.
    inline unsigned int getY(void) const { return m_y; }
    /// Sets the Y-coordinate of the top-left corner of the region.
    inline void setY(unsigned int y) { m_y = y; }
    /// Returns the identifier of the scale of the region.
    inline unsigned short getScale(void) const { return m_scale; }
    /// Sets the identifier of the scale of the region.
    inline void setScale(unsigned short scale) { m_scale = scale; }
protected:
    /// X-coordinate of the top-left corner of the region.
    unsigned int m_x;
    /// Y-coordinate of the top-left corner of the region.
    unsigned int m_y;
    /// Identifier of the scale of the region.
    unsigned short m_scale;
};

/** This class implements the Integral Histogram of Gradients descriptors presented in:
 *  Q. Zhu, M.C. Yeh, K-T. Cheng, S. Avidan, <b>"Fast Human Detection using a Cascade of Histograms of Oriented Gradients"</b>, <i>CVPR 2006</i>
 */
class IntegralHOG
{
protected:
    /// Container with the information needed to extract the iHOG descriptors at each scale.
    class ScaleInformation
    {
    public:
        /// Default constructor.
        ScaleInformation(void) : m_feature_size(0), m_window_width(0), m_window_height(0), m_step(0), m_partition_width(1), m_partition_height(1) {}
        /// Initializes the information of the scale.
        inline void initialize(unsigned int window_size, unsigned int step, double sigma_ratio, unsigned int partitions_x, unsigned int partitions_y)
        {
            unsigned int feature_size_aux;//, residual_x, residual_y;
            
            feature_size_aux = (unsigned int)/*round*/((double)window_size * sigma_ratio);
            //residual_x = window_size % partitions_x;
            //residual_y = window_size % partitions_y;
            m_feature_size = std::max(2U, (feature_size_aux % 2 == 1)?(feature_size_aux + 1):feature_size_aux);
            //m_window_width = window_size + ((residual_x > 0)?(partitions_x - residual_x):0);
            //m_window_height = window_size + ((residual_y > 0)?(partitions_y - residual_y):0);
            m_window_width = window_size;
            m_window_height = window_size;
            m_step = step;
            m_partition_width  = std::max(1U, m_window_width  / partitions_x);
            m_partition_height = std::max(1U, m_window_height / partitions_y);
        }
        /// Returns the size of the features extracted from the image.
        inline unsigned int getFeatureSize(void) const { return m_feature_size; }
        /// Returns the width of the windows sampled at the scale.
        inline unsigned int getWindowWidth(void) const { return m_window_width; }
        /// Returns the height of the windows sampled at the scale.
        inline unsigned int getWindowHeight(void) const { return m_window_height; }
        /// Returns the separation between consecutive windows.
        inline unsigned int getStep(void) const { return m_step; }
        /// Returns the width of the descriptor spatial bin.
        inline unsigned int getPartitionWidth(void) const { return m_partition_width; }
        /// Returns the height of the descriptor spatial bin.
        inline unsigned int getPartitionHeight(void) const { return m_partition_height; }
    protected:
        /// Size of the features extracted from the image.
        unsigned int m_feature_size;
        /// Width of the windows sampled at the scale.
        unsigned int m_window_width;
        /// Height of the windows sampled at the scale.
        unsigned int m_window_height;
        /// Separation between consecutive windows.
        unsigned int m_step;
        /// Width of the descriptor spatial bin.
        unsigned int m_partition_width;
        /// Height of the descriptor spatial bin.
        unsigned int m_partition_height;
    };
public:
    // -[ Constructors, destructor and assignation operator ]----------------------------------------------------------------------------------------
    /// Default constructor.
    IntegralHOG(void) {}
    /// Function which initializes the descriptor information.
    void initialize(unsigned int number_of_orientations, unsigned int partitions_x, unsigned int partitions_y, const std::vector<std::tuple<int, int> > &scales_information, double sigma_ratio);
    
    // -[ Access functions ]-------------------------------------------------------------------------------------------------------------------------
    /// Returns the number of scales of the descriptor.
    inline unsigned int getNumberOfScales(void) const { return (unsigned int)m_scales_information.size(); }
    /// Information of the descriptors at each scale.
    inline const ScaleInformation& getScaleInformation(unsigned int scale) const { return m_scales_information[scale]; }
    /// Returns the number of horizontal partitions of the descriptor.
    inline unsigned int getPartitionX(void) const { return m_partitions_x; }
    /// Returns the number of vertical partitions of the descriptor.
    inline unsigned int getPartitionY(void) const { return m_partitions_y; }
    
    // -[ Processing functions ]---------------------------------------------------------------------------------------------------------------------
    /// Initializes the locations of the densely sampled descriptors for the given image geometry.
    void initializeRegions(unsigned int width, unsigned int height, std::vector<DescriptorLocation> &locations) const;
    /// Initializes the locations of the densely samples descriptors for the given image.
    inline void initializeRegions(const cv::Mat &image, std::vector<DescriptorLocation> &locations) const { initializeRegions(image.size().width, image.size().height, locations); }
    /** Extract the Integral HOG descriptors from the specified image locations.
     *  \param[in] image unsigned char input image.
     *  \param[in,out] locations vector with the position of the descriptors on the image. The locations on the vector are only used when the \p descriptors pointer parameter is uninitialized (i.e. the pointer variable is set to 0).
     *  \param[out] descriptors array with the descriptor vector extracted at each location. Set the pointer to zero (uninitialized) to extract all possible descriptors.
     *  \param[in] descriptor_threshold minimum value of the accumulated descriptor spatial norm threshold. Descriptors with a norm lower than this threshold have 0 dimensions.
     *  \param[in] number_of_threads number of threads used to concurrently process the image.
     */
    void extract(const cv::Mat &image, std::vector<DescriptorLocation> &locations, std::vector<unsigned char> * &descriptors, double descriptor_threshold, unsigned int number_of_threads) const;
protected:
    /// Number of orientation bins.
    unsigned int m_number_of_orientations;
    /// Number of horizontal partitions of the descriptor.
    unsigned int m_partitions_x;
    /// Number of vertical partitions of the descriptor.
    unsigned int m_partitions_y;
    /// Cosine corresponding to each orientation bin.
    std::vector<double> m_cosine;
    /// Sine corresponding to each orientation bin.
    std::vector<double> m_sine;
    /// Information of the descriptors at each scale.
    std::vector<ScaleInformation> m_scales_information;
};

class Codebook
{
public:
    /// Default constructor.
    Codebook(void) : m_number_of_dimensions(0) {}
    
    /// This function load the codebook from a plain text file.
    void load(const char * filename);
    /// Initializes the codebook with the given set of codewords.
    void set(const std::vector<std::vector<double> > &codewords);
    /// Returns the number of codewords of the codebook.
    inline unsigned int getNumberOfCodewords(void) const { return (unsigned int)m_codewords.size(); }
    /// Returns the number of dimensions of the descriptors encoded by the codebook.
    inline unsigned int getNumberOfDimensions(void) const { return (m_codewords.size() > 0)?(unsigned int)m_codewords[0].size():0; }
    /** Encode a sigle descriptor with the codebook.
     *  \param[in] descriptor vector with the descriptor to encode.
     *  \param[in] number_of_neighbors number of codewords used to encode the descriptor.
     *  \param[in] code vector with the tuples encoding the descriptor (weight, codeword).
     */
    void encode(const std::vector<unsigned char> &descriptor, unsigned int number_of_neighbors, std::vector<std::tuple<double, unsigned int> > &code) const;
    /// Returns a constant reference to the index-th codeword.
    inline const std::vector<double>& getCodeword(unsigned int index) const { return m_codewords[index]; }
protected:
    /** Calculates the Euclidean distance (minus the norm of the query) between a descriptor and a codeword.
     *  \param[in] descriptor vector with the descriptor.
     *  \param[in] idx index of the codeword.
     *  \return the partial Euclidan distance.
     */
    double distance(const std::vector<unsigned char> &descriptor, unsigned int idx) const;
    /// Vector with the codewords of the codebook.
    std::vector<std::vector<double> > m_codewords;
    /// Normals of the codewords.
    std::vector<double> m_codewords_norm;
    /// Number of dimensions of the codewords.
    unsigned int m_number_of_dimensions;
};

/// Container with the information of the encoded descriptor in the window.
class LocalCodeword
{
public:
    /// Default constructor.
    LocalCodeword(void) : m_x(0.0f), m_y(0.0f), m_codes(0) {}
    /// This constructor initializes the information of the codeword in an image window.
    LocalCodeword(float x, float y, const std::vector<std::tuple<double, unsigned int> > * codes) : m_x(x), m_y(y), m_codes(codes) {}
    /// This function initializes the information of the codeword in an image window.
    void set(float x, float y, const std::vector<std::tuple<double, unsigned int> > * codes) { m_x = x; m_y = y; m_codes = codes; }
    /// This function returns the relative X-coordinate of the codeword in the window.
    inline float getX(void) const { return m_x; }
    /// This function returns the relative Y-coordinate of the codeword in the window.
    inline float getY(void) const { return m_y; }
    /// Returns a constant reference to the codewords at the relative location.
    inline const std::vector<std::tuple<double, unsigned int> >& getCodewords(void) const { return *m_codes; }
protected:
    /// Relative X-coordinate of the descriptor in the window.
    float m_x;
    /// Relative Y-coordinate of the descriptor in the window.
    float m_y;
    /// Constant pointer to the encoded descriptor information.
    const std::vector<std::tuple<double, unsigned int> > * m_codes;
};

/** This function gathers the visual words which fall within the specified window.
     *  \param[in] top_left top-left coordinates of the BoVW window in the image.
     *  \param[in] bottom_right bottom-right coordinates of the BoVW window in the image.
     *  \param[in] descriptor object used to extract the descriptors.
     *  \param[in] locations vector with the locations of the descriptors in the image.
     *  \param[in] codes array with the codes obtained from each descriptor.
 */
void WindowCodewords(int x0, int y0, int x1, int y1, const IntegralHOG &descriptor, const std::vector<DescriptorLocation> &locations, const std::vector<std::tuple<double, unsigned int> > * codes, std::list<LocalCodeword> &window_codewords);

/// Information of the histogram bin.
class HistogramBin
{
public:
    /// Default constructor.
    HistogramBin(void) : m_x0(0), m_y0(0), m_x1(0), m_y1(0), m_area(0) {}
    /// Initialization function.
    void set(float x0, float y0, float x1, float y1) { m_x0 = x0; m_y0 = y0; m_x1 = x1; m_y1 = y1; m_area = (x1 - x0) * (y1 - y0); }
    /// Checks if the given relative coordinates are inside the histogram bin.
    inline bool isInside(float x, float y) const { return (x >= m_x0) && (y >= m_y0) && (x < m_x1) && (y < m_y1); }
    /// Returns the X-coordinate of the top-left relative coordinate of the histogram bin.
    inline float getX0(void) const { return m_x0; }
    /// Returns the Y-coordinate of the top-left relative coordinate of the histogram bin.
    inline float getY0(void) const { return m_y0; }
    /// Returns the X-coordinate of the bottom-right relative coordinate of the histogram bin.
    inline float getX1(void) const { return m_x1; }
    /// Returns the Y-coordinate of the bottom-right relative coordinate of the histogram bin.
    inline float getY1(void) const { return m_y1; }
    /// Returns the area of the histogram bin.
    inline float getArea(void) const { return m_area; }
protected:
    /// Relative coordinates of the X-coordinate of the top-left corner of the histogram bin.
    float m_x0;
    /// Relative coordinates of the Y-coordinate of the top-left corner of the histogram bin.
    float m_y0;
    /// Relative coordinates of the X-coordinate of the bottom-right corner of the histogram bin.
    float m_x1;
    /// Relative coordinates of the Y-coordinate of the bottom_right corner of the histogram bin.
    float m_y1;
    /// Relative area of the histogram bin.
    float m_area;
};

/// Information of the histogram.
class HistogramPooling
{
public:
    // -[ Constructors, destructor and assignation operator ]----------------------------------------------------------------------------------------
    /// Default constructor.
    HistogramPooling(void) : m_number_of_spatial_bins(0), m_codebook_size(0), m_power_factor(0) {}
    
    // -[ Initialization functions ]-----------------------------------------------------------------------------------------------------------------
    /** This function initializes the histogram pooling spatial bins as a spatial pyramid.
     *  \param[in] number_of_levels number of levels of the spatial pyramid.
     *  \param[in] initial_partitions_x number of initial horizontal partitions.
     *  \param[in] initial_partitions_y number of initial vertical partitions.
     *  \param[in] partition_degree_x increment of the number of horizontal partitions at each new level of the pyramid.
     *  \param[in] partition_degree_y increment of the number of vertical partitions at each new level of the pyramid.
     */
    void initializePyramid(unsigned int number_of_levels, unsigned int initial_partitions_x, unsigned int initial_partitions_y, double partition_degree_x, double partition_degree_y);
    /** This function sets the remaining parameters needed to accumulate the histogram of visual words.
     *  \param[in] power_factor factor of the power normalization applied at each bin of the histogram.
     *  \param[in] codebook_size number of visual words of the codebook.
     */
    inline void set(double power_factor, unsigned int codebook_size) { m_power_factor = power_factor; m_codebook_size = codebook_size; }
    
    // -[ Access functions ]-------------------------------------------------------------------------------------------------------------------------
    /// Returns a constant reference to the vector of spatial bin coordinates of the index-th level.
    inline const std::vector<HistogramBin>& getCoordinate(unsigned int index) const { return m_coordinates[index]; }
    /// Returns a constant reference to the vector of spatial bin coordinates of the index-th level.
    inline const std::vector<HistogramBin>& operator[](unsigned int index) const { return m_coordinates[index]; }
    /// Returns the number of spatial levels of the spatial bins.
    inline unsigned int getNumberOfLevels(void) const { return (unsigned int)m_coordinates.size(); }
    /// Returns the number of spatial bins of the histogram.
    inline unsigned int getNumberOfSpatialBins(void) const { return m_number_of_spatial_bins; }
    /// Returns the number of visual words of the codebook.
    inline unsigned int getCodebookSize(void) const { return m_codebook_size; }
    /// Sets the number of visual words of the codebook.
    inline void setCodebookSize(unsigned int codebook_size) { m_codebook_size = codebook_size; }
    /// Returns the power factor normalization applied to the histogram bins.
    inline double getPowerFactor(void) const { return m_power_factor; }
    /// Sets the power factor normalization applied to the histogram bins.
    inline void setPowerFactor(double power_factor) { m_power_factor = power_factor; }
    
    // -[ Process functions ]------------------------------------------------------------------------------------------------------------------------
    /** Accumulates the given visual words into a histogram taking into account the different spatial bins.
     *  \param[in] locations vector with the locations of the descriptors in the image.
     *  \param[in] codes array with the codes obtained from each descriptor.
     *  \param[in] top_left top-left coordinates of the BoVW window in the image.
     *  \param[in] bottom_right bottom-right coordinates of the BoVW window in the image.
     *  \param[out] histogram resulting histogram vector.
     */
    void accumulate(std::list<LocalCodeword> &window_codewords, std::vector<double> &histogram) const;
protected:
    // -[ Member variables ]-------------------------------------------------------------------------------------------------------------------------
    /// Array of vectors with the coordinates of the spatial bins of each scale.
    std::vector<std::vector<HistogramBin> > m_coordinates;
    /// Total number of spatial bins in the histogram.
    unsigned int m_number_of_spatial_bins;
    /// Number of visual words of the codebook.
    unsigned int m_codebook_size;
    /// Power normalization applied to the 
    double m_power_factor;
};

/// Simple implementation of the Unweighted Pair Group Method with Arithmetic Mean agglomerative hierarchical clustering algorithm.
class UPGMA
{
public:
    /// Default constructor.
    UPGMA(void) : m_leafs(0), m_number_of_leafs(0), m_root(0) {}
    /// Copy constructor.
    UPGMA(const UPGMA &other);
    /// Destructor.
    ~UPGMA(void);
    /// Assignation operator.
    UPGMA& operator=(const UPGMA &other);
    
    /** Generates the UPGMA tree of the given data.
     *  \param[in] number_of_elements number of clustered elements (not the number of distances).
     *  \param[in] distances (upper) triangular matrix with the pair-wise distances between the clustered elements.
     *  \param[out] verbose boolean flag that enables the function to print generation process information in the standard output when it is true.
     */
    void generate(unsigned int number_of_elements, const double * distances, bool verbose = false);
    
    ///// /** Draws the UPGMA tree on the image.
    /////  *  \param[out] image image where the tree is drawn. The image must be already allocated.
    /////  */
    ///// void draw(Image<unsigned char> &image) const;
    /** This function saves the UPGMA tree into a text file.
     *  \param[in] filename file where the UPGMA tree.
     */
    void save(const char * filename) const;
    /** This function loads the UPGMA tree from a text file.
     *  \param[in] filename file containing the UPGMA tree.
     */
    void load(const char * filename);
    
    // -[ Clustering functions ]---------------------------------------------------------------------------------------------------------------------
    /** Clusters the UPGMA tree leafs using the accumulated distance between nodes as criteria.
     *  \param[in] ratio ratio between the accumulated distance in the root node (maximum distance) and the selected threshold (value between 0 and 1).
     *  \param[out] cluster_ids vector with the id assigned to each leaf of the tree (to each indexed element).
     */
    double clusterAccumulated(double ratio, std::vector<int> &cluster_ids) const;
protected:
    /// Internal node of the tree.
    class NodeUPGMA
    {
    public:
        // -[ Constructors, destructors and assignation operator ]-----------------------------------------------------------------------------------
        /// Default constructor.
        NodeUPGMA(void) : m_left(0), m_right(0), m_distance(0), m_counter(0) {}
        /// Inner node constructor.
        NodeUPGMA(NodeUPGMA * left, NodeUPGMA * right, double distance) :
            m_left((left < right)?left:right),
            m_right((left < right)?right:left),
            m_distance(distance),
            m_counter(1) {}
        /// Copy constructor.
        NodeUPGMA(const NodeUPGMA &other) :
            m_left(0),
            m_right(0),
            m_distance(other.m_distance),
            m_counter(other.m_counter)
        {
            if (other.m_left != 0)
            {
                m_left = new NodeUPGMA(*other.m_left);
                m_right = new NodeUPGMA(*other.m_right);
            }
        }
        /// Destructor (no recursive destructor).
        ~NodeUPGMA(void) {}
        /// Assignation operator.
        NodeUPGMA& operator=(const NodeUPGMA &other)
        {
            if (this != &other)
            {
                if (m_left != 0) clear();
                
                if (other.m_left != 0)
                {
                    m_left = new NodeUPGMA(*other.m_left);
                    m_right = new NodeUPGMA(*other.m_right);
                }
                else m_left = m_right = 0;
                m_distance = other.m_distance;
                m_counter = other.m_counter;
            }
            return *this;
        }
        // -[ Access functions ]---------------------------------------------------------------------------------------------------------------------
        /// Returns a pointer to the left child of the node.
        inline NodeUPGMA * getLeft(void) { return m_left; }
        /// Returns a constant pointer to the left child of the node.
        inline const NodeUPGMA * getLeft(void) const { return m_left; }
        /// Returns a pointer to the right child of the node.
        inline NodeUPGMA * getRight(void) { return m_right; }
        /// Returns a constant pointer to the right child of the node.
        inline const NodeUPGMA * getRight(void) const { return m_right; }
        /// Returns the distance between the two descendant clusters.
        inline double getDistance(void) const { return m_distance; }
        /// Sets the distance between the two descendant clusters.
        inline void setDistance(double distance) { m_distance = distance; }
        /// Returns the number of elements in the cluster.
        inline unsigned int getCounter(void) const { return m_counter; }
        /// Checks if the given node shares a descendant with the node.
        inline int isNeighbor(const NodeUPGMA * other)
        {
            if ((m_left == other->m_left) || (m_left == other->m_right)) return -1;
            else if ((m_right == other->m_left) || (m_right == other->m_right)) return 1;
            else return 0;
        }
        // -[ Node operator functions ]--------------------------------------------------------------------------------------------------------------
        /// Merge the given node with the current node, if possible.
        void merge(NodeUPGMA * other, NodeUPGMA * new_child)
        {
            m_distance = ((double)m_counter * m_distance + (double)other->m_counter * other->m_distance) / (double)(m_counter + other->m_counter);
            m_counter += other->m_counter;
            if ((m_left == new_child->m_left) || (m_left == new_child->m_right))
                m_left = new_child;
            else m_right = new_child;
        }
        /// Recursively removes the memory allocated by the descending nodes.
        void clear(void)
        {
            if (m_left != 0)
            {
                if (m_left->m_left != 0) { m_left->clear(); delete m_left; }
                if (m_right->m_left != 0) { m_right->clear(); delete m_right; }
            }
        }
        ///// /// Recursively draws the nodes in the image.
        ///// std::tuple<int, int> draw(srv::Image<unsigned char> &image, srv::Draw::Pencil<unsigned char> &pencil, unsigned int &position, double factor, double maximum_distance, const NodeUPGMA * base) const;
        /// Returns the number of inner nodes of the tree.
        inline unsigned int getNumberOfInnerNodes(void)
        {
            if (m_left != 0) return 1 + m_left->getNumberOfInnerNodes() + m_right->getNumberOfInnerNodes();
            else return 0;
        }
        /// Auxiliary function used to flatten the tree.
        void flatten(std::vector<int> &node_left, std::vector<int> &node_right, std::vector<unsigned int> &node_counter, std::vector<double> &node_distance, unsigned int &position, const NodeUPGMA * base) const;
        /// Sets the content of the node.
        void set(NodeUPGMA * left, NodeUPGMA * right, double distance, unsigned int counter) { m_left = left; m_right = right; m_distance = distance; m_counter = counter; }
        
        // -[ Node clustering functions ]------------------------------------------------------------------------------------------------------------
        /// Auxiliary function to recursively cluster the UPGMA tree leafs using the accumulated distance between nodes as criteria.
        unsigned int clusterAccumulated(unsigned int cluster_id, double threshold, std::vector<int> &cluster_ids, const NodeUPGMA * base) const;
    protected:
        /// Pointer to the left descendant (set to zero when the node is a leaf).
        NodeUPGMA * m_left;
        /// Pointer to the right descendant.
        NodeUPGMA * m_right;
        /// Distance between the node and the descendant clusters.
        double m_distance;
        /// Number of elements in the cluster.
        unsigned m_counter;
    };
    
    // -[ Member variables ]-------------------------------------------------------------------------------------------------------------------------
    /// Array with the leaf nodes of the tree.
    NodeUPGMA * m_leafs;
    /// Number of leafs or number of elements clustered by the tree.
    unsigned int m_number_of_leafs;
    /// Root node.
    NodeUPGMA * m_root;
};

#endif
