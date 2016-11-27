/* Auxiliary image processing functions.
 */

#ifndef __AUXILIARY_IMAGE_PROCESSING_FUNCTIONS_HEADER_FILE__
#define __AUXILIARY_IMAGE_PROCESSING_FUNCTIONS_HEADER_FILE__

#include <exception>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <ctime>
#include <list>
#include <string>
#include <omp.h>
#include <memory>
#include <limits>
#include <cmath>
#if 0
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

const double c_pi       = 3.141592653589793238462643383279502884;

void throwException(const char * format, ...);

enum RECURSIVE_GAUSSIAN_METHOD
{
    DERICHE = 19301, ///< Method proposed in 'R.\ Deriche, "Recursively implementing the Gaussian and its Derivatives", ICIP 1992.
    VLIET_3RD,       ///< 3rd order filter of the method proposed in 'L.\ van Vliet, I.\ Young, P.\ Verbeek, "Recursive Gaussian derivative filters." ICPR 1998.
    VLIET_4TH,       ///< 4th order filter of the method proposed in 'L.\ van Vliet, I.\ Young, P.\ Verbeek, "Recursive Gaussian derivative filters." ICPR 1998.
    VLIET_5TH        ///< 5th order filter of the method proposed in 'L.\ van Vliet, I.\ Young, P.\ Verbeek, "Recursive Gaussian derivative filters." ICPR 1998.
};

class Gaussian2DRecursiveKernel
{
public:
    // -[ Constructors, destructor and assignation operator ]------------------------------------------------------------------------------------
    /// Default constructor.
    Gaussian2DRecursiveKernel(void);
    /** Constructor where the parameters of the recursive Gaussian kernel are given as parameters.
     *  \param[in] method recursive algorithm used to approximate the Gaussian kernel.
     *  \param[in] sigma sigma of the Gaussian kernel.
     *  \param[in] derivative_x derivative of the Gaussian kernel in the X direction.
     *  \param[in] derivative_y derivative of the Gaussian kernel in the Y direction.
     */
    Gaussian2DRecursiveKernel(RECURSIVE_GAUSSIAN_METHOD method, double sigma, unsigned int derivative_x, unsigned int derivative_y);
    /// Copy constructor.
    Gaussian2DRecursiveKernel(const Gaussian2DRecursiveKernel &other);
    /// Destructor.
    ~Gaussian2DRecursiveKernel(void);
    /// Assignation operator.
    Gaussian2DRecursiveKernel& operator=(const Gaussian2DRecursiveKernel &other);
    /** This function set the parameters of the recursive Gaussian kernel.
     *  \param[in] method recursive algorithm used to approximate the Gaussian kernel.
     *  \param[in] sigma sigma of the Gaussian kernel.
     *  \param[in] derivative_x derivative of the Gaussian kernel in the X direction.
     *  \param[in] derivative_y derivative of the Gaussian kernel in the Y direction.
     */
    void set(RECURSIVE_GAUSSIAN_METHOD method, double sigma, unsigned int derivative_x, unsigned int derivative_y);
    
    // -[ Access functions ]---------------------------------------------------------------------------------------------------------------------
    /// Returns the recursive method used to approximate the Gaussian kernel.
    inline RECURSIVE_GAUSSIAN_METHOD getRecursiveMethod(void) const { return m_method; }
    /// Returns the degree of the Gaussian kernel derivative in the X-direction.
    inline unsigned int getDerivativeX(void) const { return m_derivative_x; }
    /// Returns the degree of the Gaussian kernel derivative in the Y-direction.
    inline unsigned int getDerivativeY(void) const { return m_derivative_y; }
    /// Returns the q-factor corresponding to the sigma of the Gaussian kernel in the X-direction.
    inline double getQFactorX(void) const { return m_q_x; }
    /// Returns the q-factor corresponding to the sigma of the Gaussian kernel in the Y-direction.
    inline double getQFactorY(void) const { return m_q_y; }
    /// Returns the weight applied to the X-direction Gaussian filter so that its "energy" is 1 for the unitary impulse.
    inline double getWeightX(void) const { return m_weight_x; }
    /// Returns the weight applied to the Y-direction Gaussian filter so that its "energy" is 1 for the unitary impulse.
    inline double getWeightY(void) const { return m_weight_y; }
    /// Returns the sigma of the Gaussian function.
    inline double getSigma(void) const { return m_sigma; }
    /// Returns the coefficients of the recursive Gaussian kernel in the X-direction.
    inline const double* getCoefficientsX(void) const { return m_coefficients_x; }
    /// Returns the coefficients of the recursive Gaussian kernel in the Y-direction.
    inline const double* getCoefficientsY(void) const { return m_coefficients_y; }
    /// Returns the number of coefficients of the recursive Gaussian kernel.
    inline unsigned int getNumberOfCoefficients(void) const { return m_number_of_coefficients; }
    /** Applies the X-direction Gaussian kernel derivative to the given array.
     *  \param[in] input array with the input values.
     *  \param[out] output array with the resulting filtered values.
     *  \param[in] size size of the input and output arrays.
     */
    template <class T, class U>
    inline void filterHorizontal(const T * input, U * output, unsigned int size) const { convolution(input, m_coefficients_x, m_derivative_x, m_weight_x, output, size); }
    /** Applies the X-direction Gaussian kernel derivative to the given vector.
     *  \param[in] input input vector.
     *  \param[out] output filtered output vector.
     */
    template <template <class, class> class VECTORA, class T1, class N1, template <class, class> class VECTORB, class T2, class N2>
    inline void filterHorizontal(const VECTORA<T1, N1> &input, VECTORB<T2, N2> &output) const
    {
        if (input.size() != output.size()) output.set((N2)input.size());
        filterHorizontal(input.getData(), output.getData(), (unsigned int)input.size());
    }
    /** Applies the Y-direction Gaussian kernel derivative to the given array.
     *  \param[in] input array with the input values.
     *  \param[out] output array with the resulting filtered values.
     *  \param[in] size size of the input and output arrays.
     */
    template <class T, class U>
    inline void filterVertical(const T * input, U * output, unsigned int size) const  { convolution(input, m_coefficients_y, m_derivative_y, m_weight_y, output, size); }
    /** Applies the Y-direction Gaussian kernel derivative to the given vector.
     *  \param[in] input input vector.
     *  \param[out] output filtered output vector.
     */
    template <template <class, class> class VECTORA, class T1, class N1, template <class, class> class VECTORB, class T2, class N2>
    inline void filterVertical(const VECTORA<T1, N1> &input, VECTORB<T2, N2> &output) const
    {
        if (input.size() != output.size()) output.set((N2)input.size());
        filterVertical(input.getData(), output.getData(), (unsigned int)input.size());
    }
    
private:
    template <class T, class U>
    void convolution(const T * input, const double * coefficients, unsigned int order, double weight, U * output, unsigned int size) const;
    /** This private function sets the coefficients of the recursive filter for the Deriche's algorithm.
     *  \param[in] derivative degree of the derivative of the Gaussian kernel.
     *  \param[in] sigma sigma of the Gaussian kernel.
     *  \param[out] q factor value related to the Gaussian kernel sigma for the recursive filter.
     *  \param[out] weight weight applied to the Gaussian filter so that its "energy" is 1 for the unitary impulse.
     *  \returns an array with the coefficients of the recursive filter.
     */
    double* setDericheCoefficients(unsigned int derivative, double sigma, double &q, double &weight);
    /** This private function sets the coefficients of the recursive filter for the Vliet-Young-Verbeek algorithm.
     *  \param[in] derivative degree of the derivative of the Gaussian kernel.
     *  \param[in] sigma sigma of the Gaussian kernel.
     *  \param[in] order order of the recursive filter (3rd, 4th or 5th order recursive filters).
     *  \param[out] q factor value related to the Gaussian kernel sigma for the recursive filter.
     *  \param[out] weight weight applied to the Gaussian filter so that its "energy" is 1 for the unitary impulse.
     *  \returns an array with the coefficients of the recursive filter.
     */
    double* setVlietYoungVerbeekCoefficients(unsigned int derivative, double sigma, unsigned int order, double &q, double &weight);
    /// Method used by the 
    RECURSIVE_GAUSSIAN_METHOD m_method;
    /// Degree of the Gaussian kernel derivative in the X-direction.
    unsigned int m_derivative_x;
    /// Degree of the Gaussian kernel derivative in the Y-direction.
    unsigned int m_derivative_y;
    /// Value related to the sigma of the recursive Gaussian kernel applied in the X-direction.
    double m_q_x;
    /// Value related to the sigma of the recursive Gaussian kernel applied in the Y-direction.
    double m_q_y;
    /// Weight applied to the X-direction Gaussian filter so that its "energy" is 1 for the unitary impulse.
    double m_weight_x;
    /// Weight applied to the Y-direction Gaussian filter so that its "energy" is 1 for the unitary impulse.
    double m_weight_y;
    /// Sigma of the Gaussian kernel.
    double m_sigma;
    /// Coefficients for the recursive kernel applied in the X-direction.
    double *m_coefficients_x;
    /// Coefficients for the recursive kernel applied in the X-direction.
    double *m_coefficients_y;
    /// Number of coefficients of the recursive kernel.
    unsigned int m_number_of_coefficients;
};

template <class T, class U>
void Gaussian2DRecursiveKernel::convolution(const T * input, const double * coefficients, unsigned int order, double weight, U * output, unsigned int size) const
{
    switch (m_method)
    {
        case DERICHE:
            {
                double y01, y02, y03, y04, x01, x02, x03, x04;
                const double sign = (order == 0)?1.0:-1.0;
                
                y04 = 0.0; y03 = 0.0; y02 = 0.0; y01 = 0.0;
                           x03 = 0.0; x02 = 0.0; x01 = 0.0;
                // 3.2.1) Forward filter.
                for (int x = 0; x < (int)size; ++x)
                {
                    double current_x = (double)input[x];
                    output[x] = (U)(coefficients[0] * current_x + coefficients[1] * x01 + coefficients[2] * x02 + coefficients[3] * x03
                        - coefficients[4] * y01 - coefficients[5] * y02 - coefficients[6] * y03 - coefficients[7] * y04);
                    
                    y04 = y03; y03 = y02; y02 = y01; y01 = output[x];
                               x03 = x02; x02 = x01; x01 = current_x;
                }
                // 3.2.2) Backward filter.
                y04 = 0.0; y03 = 0.0; y02 = 0.0; y01 = 0.0;
                x04 = 0.0; x03 = 0.0; x02 = 0.0; x01 = 0.0;
                for (int x = (int)size - 1; x >= 0; --x)
                {
                    double value = coefficients[8] * x01 + coefficients[9] * x02 + coefficients[10] * x03 + coefficients[11] * x04
                        -coefficients[4] * y01 - coefficients[5] * y02 - coefficients[6] * y03 - coefficients[7] * y04;
                    output[x] = (U)(sign * (output[x] + value) / weight);
                    y04 = y03; y03 = y02; y02 = y01; y01 = value;
                    x04 = x03; x03 = x02; x02 = x01; x01 = (double)input[x];
                }
            }
            break;
        case VLIET_3RD:
        case VLIET_4TH:
        case VLIET_5TH:
            {
                const double alpha = coefficients[m_number_of_coefficients - 1];
                const double * b = coefficients;
                std::vector<double> v(m_number_of_coefficients - 1, 0);
                
                // Forward filter.
                if (order == 0)
                {
                    for (int x = 0; x < (int)size; ++x)
                    {
                        double current = input[x] * alpha;
                        for (unsigned int k = 0; k < m_number_of_coefficients - 1; ++k)
                            current -= v[k] * b[k];
                        output[x] = (U)current;
                        
                        for (unsigned int k = m_number_of_coefficients - 2; k > 0; --k)
                            v[k] = v[k - 1];
                        v[0] = current;
                    }
                }
                else if (order == 1)
                {
                    output[0] = 0; output[size - 1] = 0;
                    for (int x = 1; x < (int)size - 1; ++x)
                    {
                        double current = ((double)input[x - 1] - (double)input[x + 1]) * alpha / 2.0;
                        for (unsigned int k = 0; k < m_number_of_coefficients - 1; ++k)
                            current -= v[k] * b[k];
                        output[x] = (U)current;
                        
                        for (unsigned int k = m_number_of_coefficients - 2; k > 0; --k)
                            v[k] = v[k - 1];
                        v[0] = current;
                    }
                }
                else
                {
                    output[0] = 0;
                    for (int x = 1; x < (int)size; ++x)
                    {
                        double current = ((double)input[x] - (double)input[x - 1]) * alpha;
                        for (unsigned int k = 0; k < m_number_of_coefficients - 1; ++k)
                            current -= v[k] * b[k];
                        output[x] = (U)current;
                        
                        for (unsigned int k = m_number_of_coefficients - 2; k > 0; --k)
                            v[k] = v[k - 1];
                        v[0] = current;
                    }
                }
                
                // Backward filter.
                std::fill(v.begin(), v.end(), 0);
                
                if (order == 2)
                {
                    double previous = 0.0, aux, current;
                    for (int x = (int)size - 1; x >= 0; --x)
                    {
                        aux = (double)output[x];
                        current = (previous - (double)output[x]) * alpha;
                        previous = aux;
                        for (unsigned int k = 0; k < m_number_of_coefficients - 1; ++k)
                            current -= v[k] * b[k];
                        output[x] = (U)(-current / weight);
                        
                        for (unsigned int k = m_number_of_coefficients - 2; k > 0; --k)
                            v[k] = v[k - 1];
                        v[0] = current;
                    }
                }
                else
                {
                    for (int x = (int)size - 1; x >= 0; --x)
                    {
                        double current = (U)(output[x] * alpha);
                        for (unsigned int k = 0; k < m_number_of_coefficients - 1; ++k)
                            current -= v[k] * b[k];
                        output[x] = (U)(current / weight);
                        
                        for (unsigned int k = m_number_of_coefficients - 2; k > 0; --k)
                            v[k] = v[k - 1];
                        v[0] = current;
                    }
                }
            }
            break;
        default:
            throwException("Unknown method identifier '%d'", (int)m_method);
            break;
    }
}

/** Filters the input image with a Gaussian filter.
 *  \tparam TSOURCE underlying  data type of the source image (needed for casts).
 *  \tparam TDESTINATION underlying data type of the destination image (needed for casts).
 *  \param[in] source_image original image.
 *  \param[in] kernel recursive Gaussian kernel used to filter the image.
 *  \param[out] destination_image resulting filtered image. 
 */
template <class TSOURCE, class TDESTINATION>
void gaussianFilter(cv::Mat &image_source, const Gaussian2DRecursiveKernel &kernel, cv::Mat &image_destination, unsigned int number_of_threads)
{
    const unsigned int width = image_source.size().width;
    const unsigned int height = image_source.size().height;
    const unsigned int number_of_channels = image_source.channels();
    const int type_derivatives = cv::DataType<TDESTINATION>::depth;
    ////if (image_destination.size() != image_source.size())
    ////    throwException("[gaussianFilter] Images must have the same geometry.");
    ////if (image_destination.channels() != image_source.channels())
    ////    throwException("[gaussianFilter] Images must have the same number of channels.");
    ////typedef decltype(*(image_destination.ptr(0))) TDESTINATION;
    ////typedef decltype(*(image_source.ptr(0))) TSOURCE;
    ////typedef decltype(std::remove_pointer<image_destination.ptr(0)>::type) TDESTINATION;
    ////typedef decltype(std::remove_pointer<image_source.ptr(0)>::type) TSOURCE;
    TDESTINATION * buffer, * * line_input, * * line_output, * * destination_row;
    
    image_destination.create(cv::Size(width, height), CV_MAKETYPE(type_derivatives, number_of_channels));
    buffer = new TDESTINATION[width * height];
    line_input = new TDESTINATION * [number_of_threads];
    line_output = new TDESTINATION * [number_of_threads];
    for (unsigned int t = 0; t < number_of_threads; ++t)
    {
        line_input[t] = new TDESTINATION[std::max(width, height)];
        line_output[t] = new TDESTINATION[std::max(width, height)];
    }
    destination_row = new TDESTINATION * [height];
    for (unsigned int y = 0; y < height; ++y)
        destination_row[y] = (TDESTINATION *)image_destination.ptr(y);
    for (unsigned int c = 0; c < number_of_channels; ++c)
    {
        #pragma omp parallel num_threads(number_of_threads)
        {
            unsigned int thread_id = omp_get_thread_num();
            for (unsigned int y = thread_id; y < height; y += number_of_threads)
            {
                TSOURCE * ptr_source = (TSOURCE *)image_source.ptr(y) + c;
                for (unsigned int x = 0; x < width; ++x, ptr_source += number_of_channels)
                    line_input[thread_id][x] = *ptr_source;
                kernel.filterHorizontal(line_input[thread_id], line_output[thread_id], width);
                for (unsigned int x = 0; x < width; ++x)
                    buffer[x * height + y] = line_output[thread_id][x];
            }
        }
        #pragma omp parallel num_threads(number_of_threads)
        {
            unsigned int thread_id = omp_get_thread_num();
            for (unsigned int x = thread_id; x < width; x += number_of_threads)
            {
                kernel.filterVertical(buffer + height * x, line_output[thread_id], height);
                for (unsigned int y = 0; y < height; ++y)
                    *(destination_row[y] + x * number_of_channels + c) = line_output[thread_id][y];
            }
        }
    }
    
    delete [] buffer;
    for (unsigned int t = 0; t < number_of_threads; ++t)
    {
        delete [] line_input[t];
        delete [] line_output[t];
    }
    delete [] line_input;
    delete [] line_output;
    delete [] destination_row;
}

/** Filters the input image with a Gaussian filter.
 *  \tparam TIMAGE underlying  data type of the source image (needed for casts).
 *  \tparam TDErivative underlying data type of the derivative images (needed for casts).
 *  \param[in] image_original original image.
 *  \param[in] orientation_sigma sigma of the Gaussian derivative.
 *  \param[out] image_orientation_angle image with the angles of the dominant filters calculated at each pixel.
 *  \param[out] image_orientation_value image with the values of the dominant filters calculated at each pixel.
 *  \param[in] number_of_threads number of threads used to concurrently process the image.
 */
template <class TIMAGE, class TDERIVATIVE>
void calculateOrientations(cv::Mat &image_original, double orientation_sigma, cv::Mat &image_orientation_angle, cv::Mat &image_orientation_value, unsigned int number_of_threads)
{
    const unsigned int number_of_channels = image_original.channels();
    const int type_derivatives = cv::DataType<TDERIVATIVE>::depth;
    const unsigned int width = image_original.size().width;
    const unsigned int height = image_original.size().height;
    const double derivatives_weight_ratio = 1.53;
    cv::Mat image_dx2, image_dxdy, image_dy2;
    
    Gaussian2DRecursiveKernel kernel_dx2 (VLIET_4TH, orientation_sigma, 2, 0);
    Gaussian2DRecursiveKernel kernel_dy2 (VLIET_4TH, orientation_sigma, 0, 2);
    Gaussian2DRecursiveKernel kernel_dxdy(VLIET_4TH, orientation_sigma, 1, 1);
    gaussianFilter<TIMAGE, TDERIVATIVE>(image_original, kernel_dx2 , image_dx2 , number_of_threads);
    gaussianFilter<TIMAGE, TDERIVATIVE>(image_original, kernel_dy2 , image_dy2 , number_of_threads);
    gaussianFilter<TIMAGE, TDERIVATIVE>(image_original, kernel_dxdy, image_dxdy, number_of_threads);
    image_orientation_angle.create(cv::Size(width, height), CV_MAKETYPE(type_derivatives, number_of_channels));
    image_orientation_value.create(cv::Size(width, height), CV_MAKETYPE(type_derivatives, number_of_channels));
    
    #pragma omp parallel num_threads(number_of_threads)
    {
        const unsigned int thread_id = omp_get_thread_num();
        for (unsigned int y = 0; y < height; ++y)
        {
            TDERIVATIVE * ptr_dx2  = (TDERIVATIVE *)image_dx2.ptr(y)  + thread_id;
            TDERIVATIVE * ptr_dxdy = (TDERIVATIVE *)image_dxdy.ptr(y) + thread_id;
            TDERIVATIVE * ptr_dy2  = (TDERIVATIVE *)image_dy2.ptr(y)  + thread_id;
            TDERIVATIVE * ptr_angle = (TDERIVATIVE *)image_orientation_angle.ptr(y) + thread_id;
            TDERIVATIVE * ptr_value = (TDERIVATIVE *)image_orientation_value.ptr(y) + thread_id;
            for (unsigned int x = thread_id; x < width; x += number_of_threads, ptr_dx2 += number_of_threads, ptr_dxdy += number_of_threads, ptr_dy2 += number_of_threads, ptr_angle += number_of_threads, ptr_value += number_of_threads)
            {
                double c1, c2, c3, angle1, angle2, value1, value2, vcos, vsin;
                
                // Get the weighted contribution of each derivative.
                c1 = derivatives_weight_ratio * *ptr_dx2;
                c2 =                            *ptr_dxdy;
                c3 = derivatives_weight_ratio * *ptr_dy2;
                // Calculate the to points where the filter has an extrema.
                angle1 = std::atan(2.0 * c2 / (c3 - c1)) / 2.0;
                vcos = std::cos(angle1);
                vsin = std::sin(angle1);
                value1 = vcos * vcos * c1 - 2.0 * vcos * vsin * c2 + vsin * vsin * c3;
                angle2 = angle1 + c_pi / 2.0;
                vcos = std::cos(angle2);
                vsin = std::sin(angle2);
                value2 = vcos * vcos * c1 - 2.0 * vcos * vsin * c2 + vsin * vsin * c3;
                // Select the local maxima.
                if (std::abs(value1) > std::abs(value2))
                {
                    *ptr_angle = angle1 + c_pi / 2.0;
                    *ptr_value = value1;
                }
                else
                {
                    *ptr_angle = angle2 + c_pi / 2.0;
                    *ptr_value = value2;
                }
                // Make that the angle ranges between 0 and pi.
                while (*ptr_angle >= c_pi) *ptr_angle -= c_pi;
                while (*ptr_angle < 0.0) *ptr_angle += c_pi;
            }
        }
    }
}

/// Class which smooths the given values vector with a Gaussian function.
template <class T>
class smoothVector
{
public:
    /// Default constructor.
    smoothVector(void) : m_sigma(0.0), m_values(0), m_size(0) {}
    /// Constructor which initializes the Gaussian vector filter.
    smoothVector(T sigma) : m_sigma(sigma), m_values(new T[(unsigned int)std::max(0.0, ceil(sigma * 5.0) * 2.0 + 1.0)]), m_size((unsigned int)std::max(0.0, ceil(sigma * 5.0) * 2.0 + 1.0))
    {
        if (m_size > 0)
        {
            const T sigma2 = sigma * sigma;
            T sum = 0;
            
            m_values[m_size / 2] = 1.0;
            for (unsigned int i = 1; i <= m_size / 2; ++i)
                sum += m_values[m_size / 2 - i] = m_values[m_size / 2 + i] = std::exp(-(T)i * (T)i / sigma2);
            sum = 2.0 * sum + 1;
            for (unsigned int i = 0; i < m_size; ++i)
                m_values[i] /= sum;
        }
    }
    /// Copy constructor.
    smoothVector(const smoothVector<T> &other) : m_sigma(other.m_sigma), m_values((other.m_values != 0)?new T[other.m_size]:0), m_size(other.m_size) { for (unsigned int i = 0; i < other.m_size; ++i) m_values[i] = other.m_values[i]; }
    /// Destructor.
    ~smoothVector(void) { if (m_values != 0) delete [] m_values; }
    /// Assignation operator.
    smoothVector<T>& operator=(const smoothVector<T> &other)
    {
        if (this != &other)
        {
            if (m_values != 0) delete [] m_values;
            m_sigma = other.m_sigma;
            if (other.m_values != 0)
            {
                m_values = new T[other.m_size];
                m_size = other.m_size;
                for (unsigned int i = 0; i < other.m_size; ++i) m_values[i] = other.m_values[i];
            }
            else
            {
                m_values = 0;
                m_size = 0;
            }
        }
        return *this;
    }
    /// Function which sets the sigma of the Gaussian function.
    void set(T sigma)
    {
        if (m_values != 0) delete [] m_values;
        m_sigma = sigma;
        m_size = (unsigned int)std::max(0.0, ceil(sigma * 5.0) * 2.0 + 1.0);
        if (m_size > 0)
        {
            const T sigma2 = sigma * sigma;
            T sum = 0;
            
            m_values = new T[m_size];
            m_values[m_size / 2] = 1.0;
            for (unsigned int i = 1; i <= m_size / 2; ++i)
                sum += m_values[m_size / 2 - i] = m_values[m_size / 2 + i] = std::exp(-(T)i * (T)i / sigma2);
            sum = 2.0 * sum + 1;
            for (unsigned int i = 0; i < m_size; ++i)
                m_values[i] /= sum;
        }
        else m_values = 0;
    }
    /** Applies the Gaussian smoothing function on the given vector.
     *  \param[in] input input vector.
     *  \param[out] output vector resulting of applying the Gaussian smooth function.
     *  \param[in] circular the vector is circular.
     */
    template <class TDATA>
    void operator()(const std::vector<TDATA> &input, std::vector<TDATA> &output, bool circular = false)
    {
        const int half_size = (int)m_size / 2;
        output.assign(input.size(), 0);
        if (circular)
        {
            for (int i = 0; i < (int)input.size(); ++i)
            {
                T sum = 0;
                for (int j = -half_size, k = 0; j <= half_size; ++j, ++k)
                {
                    int bin = i + j;
                    if (bin < 0) bin = (int)input.size() + bin;
                    else bin = bin % (int)input.size();
                    sum += input[bin] * m_values[k];
                }
                output[i] = sum;
            }
        }
        else
        {
            for (int i = 0; i < (int)input.size(); ++i)
            {
                T sum = 0;
                for (int j = -half_size, k = 0; j <= half_size; ++j, ++k)
                    if ((i + j >= 0) && (i + j < (int)input.size()))
                        sum += input[i + j] * m_values[k];
                output[i] = sum;
            }
        }
    }
    /** Applies the Gaussian smoothing function on the given vector.
     *  \param[in,out] data vector where the Gaussian vector is applied.
     *  \param[in] circular the vector is circular.
     */
    template <class TDATA> inline void operator()(std::vector<TDATA> &data, bool circular = false) { this->operator()(std::vector<TDATA>(data), data, circular); }
protected:
    T m_sigma;
    T * m_values;
    unsigned int m_size;
};

/** Calculates the gradient in the X and Y direction for the specified image. This function can down-sample the resulting gradient images.
 *  \param[in] integral_image integral image of the original image.
 *  \param[in] size size of the square patch used to approximate the Gaussian kernel.
 *  \param[out] gradient_dx resulting image or sub-image with the gradient in the X direction.
 *  \param[out] gradient_dy resulting image or sub-image with the gradient in the Y direction.
 *  \param[in] number_of_threads number of threads used to process the image concurrently. By default set to the number of threads specified by \b DefaultNumberOfThreads::ImageProcessing().
 */
template <class TINTEGRAL, class TGRADIENT>
void ImageGradient(const cv::Mat &integral_image, unsigned int size, cv::Mat &gradient_dx, cv::Mat &gradient_dy, unsigned int number_of_threads)
{
    const unsigned int width = integral_image.size().width - 1;
    const unsigned int height = integral_image.size().height - 1;
    // Initialize constant variables and check image geometry.
    if ((gradient_dx.size().width != gradient_dy.size().width) || (gradient_dx.size().height != gradient_dy.size().height))
        throwException("Both gradient images must have the same geometry.");
    if ((gradient_dx.channels() != 1) || (gradient_dy.channels() != 1))
        throwException("Gradient function only works with single channel images");
    if (integral_image.channels() != 1)
        throwException("The integral image must have a single channel images.");
    
    if (size % 2 == 1) ++size; // Make the size even.
    const unsigned int size_half = size / 2;
    const unsigned int size2 = (size * size) / 2;
    
    #pragma omp parallel num_threads(number_of_threads)
    {
        unsigned int y = omp_get_thread_num();
        
        // Set to zero the first rows of the resulting image.
        for (; y < size_half; y += number_of_threads)
        {
            TGRADIENT * __restrict__ dx_ptr = (TGRADIENT *)gradient_dx.ptr(y);
            TGRADIENT * __restrict__ dy_ptr = (TGRADIENT *)gradient_dy.ptr(y);
            for (unsigned int x = 0; x < width; ++x)
            {
                *dx_ptr = 0;
                *dy_ptr = 0;
                ++dx_ptr;
                ++dy_ptr;
            }
        }
        
        // Calculate the Gaussian filter approximate for the middle image rows.
        for (; y < height - size_half; y += number_of_threads)
        {
            const TINTEGRAL * __restrict__ top_integral_ptr = (TINTEGRAL *)integral_image.ptr() + (y - size_half) * (width + 1);
            const TINTEGRAL * __restrict__ middle_integral_ptr = (TINTEGRAL *)integral_image.ptr() + y * (width + 1);
            const TINTEGRAL * __restrict__ bottom_integral_ptr = (TINTEGRAL *)integral_image.ptr() + (y + size_half) * (width + 1);
            TGRADIENT * __restrict__ dx_ptr = (TGRADIENT *)gradient_dx.ptr(y);
            TGRADIENT * __restrict__ dy_ptr = (TGRADIENT *)gradient_dy.ptr(y);
            unsigned int x = 0;
            
            // First half columns set to zero.
            for (; x < size_half; ++x)
            {
                *dx_ptr = 0;
                *dy_ptr = 0;
                ++dx_ptr;
                ++dy_ptr;
            }
            
            // Middle columns where the gradient is well defined.
            for (; x < width - size_half; ++x)
            {
                int area =    top_integral_ptr[0] + bottom_integral_ptr[size]      - top_integral_ptr[size]      - bottom_integral_ptr[0];
                int dx   =    top_integral_ptr[0] + bottom_integral_ptr[size_half] - top_integral_ptr[size_half] - bottom_integral_ptr[0];
                int dy   = middle_integral_ptr[0] + bottom_integral_ptr[size]   - middle_integral_ptr[size]      - bottom_integral_ptr[0];
                *dx_ptr = (short)((-(long)area + 2 * (long)dx) / (long)size2);
                *dy_ptr = (short)(( (long)area - 2 * (long)dy) / (long)size2);
                
                ++top_integral_ptr;
                ++bottom_integral_ptr;
                ++middle_integral_ptr;
                ++dx_ptr;
                ++dy_ptr;
            }
            
            // Last half columns set to zero.
            for (; x < width; ++x)
            {
                *dx_ptr = 0;
                *dy_ptr = 0;
                ++dx_ptr;
                ++dy_ptr;
            }
        }
        
        // Set to zero the last rows of the resulting image.
        for (; y < height; y += number_of_threads)
        {
            TGRADIENT * __restrict__ dx_ptr = (TGRADIENT *)gradient_dx.ptr(y);
            TGRADIENT * __restrict__ dy_ptr = (TGRADIENT *)gradient_dy.ptr(y);
            
            for (unsigned int x = 0; x < width; ++x)
            {
                *dx_ptr = 0;
                *dy_ptr = 0;
                ++dx_ptr;
                ++dy_ptr;
            }
        }
    }
}

/** This function creates the integral image of the given input image.
 *  \param[in] input input image.
 *  \param[out] integral resulting integral image.
 *  \note OpenCV is only able to create integral images from and to specific image data types.
 */
template <class TINPUT, class TINTEGRAL>
void ImageIntegral(const cv::Mat &input, cv::Mat &integral)
{
    const unsigned int width = input.size().width;
    const unsigned int height = input.size().height;
    const unsigned int nchannels = input.channels();
    if (((int)width != integral.size().width - 1) || ((int)height != integral.size().height - 1)) throwException("The input and integral images must have the same geometry.");
    if ((int)nchannels != integral.channels()) throwException("the input and integral images must have the same number of channels.");
    
    {   // Initialize the first row of the integral image to 0.
        TINTEGRAL * __restrict__ integral_ptr = (TINTEGRAL *)integral.ptr();
        for (unsigned int x = 0; x < (width + 1) * nchannels; ++x, ++integral_ptr)
            *integral_ptr = 0;
    }
    for (unsigned int channel = 0; channel < nchannels; ++channel)
    {
        for (unsigned int y = 0; y < height; ++y)
        {
            TINTEGRAL * __restrict__ integral_ptr = (TINTEGRAL *)integral.ptr(y + 1) + channel;
            TINTEGRAL * __restrict__ previous_ptr = (TINTEGRAL *)integral.ptr(y    ) + channel;
            const TINPUT * __restrict__ input_ptr = (const TINPUT *)input.ptr(y) + channel;
            register TINTEGRAL accumulated_row;
            
            // Initialize the first column of the integral image to 0.
            *integral_ptr = accumulated_row = 0;
            integral_ptr += nchannels;
            previous_ptr += nchannels;
            for (unsigned int x = 0; x < width; ++x, input_ptr += nchannels, integral_ptr += nchannels, previous_ptr += nchannels)
            {
                accumulated_row += (TINTEGRAL)*input_ptr;
                *integral_ptr = accumulated_row + *previous_ptr;
            }
        }
    }
}

#endif

