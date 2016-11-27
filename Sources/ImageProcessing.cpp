#include "../Headers/ImageProcessing.hpp"

void throwException(const char * format, ...)
{
    char message[1024];
    va_list args;
    
    va_start(args, format);
    vsprintf(message, format, args);
    va_end(args);
    CV_Error(CV_StsError, message);
}

Gaussian2DRecursiveKernel::Gaussian2DRecursiveKernel(void) :
    m_method(VLIET_3RD),
    m_derivative_x(0),
    m_derivative_y(0),
    m_q_x(0.0),
    m_q_y(0.0),
    m_weight_x(1.0),
    m_weight_y(1.0),
    m_sigma(0.0),
    m_coefficients_x(0),
    m_coefficients_y(0),
    m_number_of_coefficients(0)
{}

Gaussian2DRecursiveKernel::Gaussian2DRecursiveKernel(RECURSIVE_GAUSSIAN_METHOD method, double sigma, unsigned int derivative_x, unsigned int derivative_y) :
    m_method(method),
    m_derivative_x(derivative_x),
    m_derivative_y(derivative_y),
    m_sigma(sigma)
{
    if (method == DERICHE)
    {
        m_number_of_coefficients = 12;
        m_coefficients_x = setDericheCoefficients(derivative_x, sigma, m_q_x, m_weight_x);
        m_coefficients_y = setDericheCoefficients(derivative_y, sigma, m_q_y, m_weight_y);
    }
    else
    {
        if (method == VLIET_3RD) m_number_of_coefficients = 4;
        else if (method == VLIET_4TH) m_number_of_coefficients = 5;
        else if (method == VLIET_5TH) m_number_of_coefficients = 6;
        m_coefficients_x = setVlietYoungVerbeekCoefficients(derivative_x, sigma, m_number_of_coefficients - 1, m_q_x, m_weight_x);
        m_coefficients_y = setVlietYoungVerbeekCoefficients(derivative_y, sigma, m_number_of_coefficients - 1, m_q_y, m_weight_y);
    }
}

Gaussian2DRecursiveKernel::Gaussian2DRecursiveKernel(const Gaussian2DRecursiveKernel &other) :
    m_method(other.m_method),
    m_derivative_x(other.m_derivative_x),
    m_derivative_y(other.m_derivative_y),
    m_q_x(other.m_q_x),
    m_q_y(other.m_q_y),
    m_weight_x(other.m_weight_x),
    m_weight_y(other.m_weight_y),
    m_sigma(other.m_sigma),
    m_coefficients_x((other.m_coefficients_x != 0)?new double[other.m_number_of_coefficients]:0),
    m_coefficients_y((other.m_coefficients_y != 0)?new double[other.m_number_of_coefficients]:0),
    m_number_of_coefficients(other.m_number_of_coefficients)
{
    for (unsigned int i = 0; i < other.m_number_of_coefficients; ++i)
    {
        m_coefficients_x[i] = other.m_coefficients_x[i];
        m_coefficients_y[i] = other.m_coefficients_y[i];
    }
}

Gaussian2DRecursiveKernel::~Gaussian2DRecursiveKernel(void)
{
    if (m_coefficients_x != 0) delete [] m_coefficients_x;
    if (m_coefficients_y != 0) delete [] m_coefficients_y;
}

Gaussian2DRecursiveKernel& Gaussian2DRecursiveKernel::operator=(const Gaussian2DRecursiveKernel &other)
{
    if (this != &other)
    {
        // Free .................................................................................................................................
        if (m_coefficients_x != 0) delete [] m_coefficients_x;
        if (m_coefficients_y != 0) delete [] m_coefficients_y;
        
        // Copy .................................................................................................................................
        m_method = other.m_method;
        m_derivative_x = other.m_derivative_x;
        m_derivative_y = other.m_derivative_y;
        m_q_x = other.m_q_x;
        m_q_y = other.m_q_y;
        m_weight_x = other.m_weight_x;
        m_weight_y = other.m_weight_y;
        m_sigma = other.m_sigma;
        m_number_of_coefficients = other.m_number_of_coefficients;
        if (other.m_coefficients_x != 0)
        {
            m_coefficients_x = new double[other.m_number_of_coefficients];
            for (unsigned int i = 0; i < other.m_number_of_coefficients; ++i)
                m_coefficients_x[i] = other.m_coefficients_x[i];
        }
        else m_coefficients_x = 0;
        if (other.m_coefficients_y != 0)
        {
            m_coefficients_y = new double[other.m_number_of_coefficients];
            for (unsigned int i = 0; i < other.m_number_of_coefficients; ++i)
                m_coefficients_y[i] = other.m_coefficients_y[i];
        }
        else m_coefficients_y = 0;
    }
    return *this;
}

void Gaussian2DRecursiveKernel::set(RECURSIVE_GAUSSIAN_METHOD method, double sigma, unsigned int derivative_x, unsigned int derivative_y)
{
    // Free .....................................................................................................................................
    if (m_coefficients_x != 0) delete [] m_coefficients_x;
    if (m_coefficients_y != 0) delete [] m_coefficients_y;
    // Generate .................................................................................................................................
    m_method = method;
    m_derivative_x = derivative_x;
    m_derivative_y = derivative_y;
    m_sigma = sigma;
    
    if (method == DERICHE)
    {
        m_number_of_coefficients = 12;
        m_coefficients_x = setDericheCoefficients(derivative_x, sigma, m_q_x, m_weight_x);
        m_coefficients_y = setDericheCoefficients(derivative_y, sigma, m_q_y, m_weight_y);
    }
    else
    {
        if (method == VLIET_3RD) m_number_of_coefficients = 4;
        else if (method == VLIET_4TH) m_number_of_coefficients = 5;
        else if (method == VLIET_5TH) m_number_of_coefficients = 6;
        m_coefficients_x = setVlietYoungVerbeekCoefficients(derivative_x, sigma, m_number_of_coefficients - 1, m_q_x, m_weight_x);
        m_coefficients_y = setVlietYoungVerbeekCoefficients(derivative_y, sigma, m_number_of_coefficients - 1, m_q_y, m_weight_y);
    }
}

double* Gaussian2DRecursiveKernel::setDericheCoefficients(unsigned int derivative, double sigma, double &q, double &weight)
{
    double y01, y02, y03, y04, x01, x02, x03, x04, *coefficients;
    unsigned int n, m;
    double *sx, *sy;
    
    // Constant coefficients of the Deriches algorithm for each Gaussian derivative.
    const double va0[] = {  1.6800, -0.6472, -1.3310};
    const double va1[] = {  3.7350, -4.5310,  3.6610};
    const double vb0[] = {  1.7830,  1.5270,  1.2400};
    const double vb1[] = {  1.7230,  1.5160,  1.3140};
    const double vc0[] = { -0.6803,  0.6494,  0.3225};
    const double vc1[] = { -0.2598,  0.9557, -1.7380};
    const double vw0[] = {  0.6318,  0.6719,  0.7480};
    const double vw1[] = {  1.9970,  2.0720,  2.1660};
    
    const double a0 = va0[derivative];
    const double a1 = va1[derivative];
    const double b0 = vb0[derivative];
    const double b1 = vb1[derivative];
    const double c0 = vc0[derivative];
    const double c1 = vc1[derivative];
    const double w0 = vw0[derivative];
    const double w1 = vw1[derivative];
    
    // 1) Calculate the Derice's coefficients.
    coefficients = new double[12];
    // From n0 to n3
    coefficients[0] = (derivative == 1)?0:(a0 + c0);
    coefficients[1] = exp(-b1 / sigma) * (c1 * sin(w1 / sigma) -(c0 + 2.0 * a0) * cos(w1 / sigma)) +exp(-b0 / sigma) * (a1 * sin(w0 / sigma) -(2.0 * c0 + a0) * cos(w0 / sigma));
    coefficients[2] = 2.0 * exp(-(b0 + b1) / sigma) * ((a0 + c0) * cos(w1 / sigma) * cos(w0 / sigma) - a1 * cos(w1 / sigma) * sin(w0 / sigma) - c1 * cos(w0 / sigma) * sin(w1 / sigma)) + c0 * exp(-2.0 * b0 / sigma) + a0 * exp(-2.0 * b1 / sigma);
    coefficients[3] = exp(-(b1 + 2.0 * b0) / sigma) * (c1 * sin(w1 / sigma) -c0 * cos(w1 / sigma)) + exp(-(b0 + 2.0 * b1) / sigma) * (a1 * sin(w0 / sigma) -a0 * cos(w0 / sigma));
    // From d1 to d4
    coefficients[4] = -2.0 * exp(-b0 / sigma) * cos(w0 / sigma) -2.0 * exp(-b1 / sigma) * cos(w1 / sigma);
    coefficients[5] = 4.0 * exp(-(b0 + b1) / sigma) * cos(w0 / sigma) * cos(w1 / sigma) + exp(-2.0 * b0 / sigma) + exp(-2.0 * b1 / sigma);
    coefficients[6] = -2.0 * exp(-(b0 + 2.0 * b1) / sigma) * cos(w0 / sigma) - 2.0 * exp(-(b1 + 2.0 * b0) / sigma) * cos(w1 / sigma);
    coefficients[7] = exp(-2.0 * (b0 + b1) / sigma);
    
    // 2) Scale the coefficients to fit the current sigma.
    n = 1 + 2 * (unsigned int)(10.0 * sigma);
    m = (n - 1) / 2;
    sx = new double[n];
    sy = new double[n];
    for (unsigned int i = 0; i < n; ++i) sx[i] = 0.0;
    sx[m] = 1.0;
    
    y04 = 0.0; y03 = 0.0; y02 = 0.0; y01 = 0.0;
               x03 = 0.0; x02 = 0.0; x01 = 0.0;
    for (unsigned int k = 0; k < n; ++k) // Forward filtering...
    {
        double xi = sx[k];
        double yi = coefficients[0] * xi + coefficients[1] * x01 + coefficients[2] * x02 + coefficients[3] * x03
                                         - coefficients[4] * y01 - coefficients[5] * y02 - coefficients[6] * y03 - coefficients[7] * y04;
        sy[k] = yi;
        y04 = y03; y03 = y02; y02 = y01; y01 = yi;
                   x03 = x02; x02 = x01; x01 = xi;
    }
    
    if (derivative == 1) // For non-symmetric filters the negative coefficients are calculated differently.
    {
        coefficients[8]  = -(coefficients[1] - coefficients[4] * coefficients[0]);
        coefficients[9]  = -(coefficients[2] - coefficients[5] * coefficients[0]);
        coefficients[10] = -(coefficients[3] - coefficients[6] * coefficients[0]);
        coefficients[11] = coefficients[7] * coefficients[0];
    }
    else
    {
        coefficients[8]  = coefficients[1] - coefficients[4] * coefficients[0];
        coefficients[9]  = coefficients[2] - coefficients[5] * coefficients[0];
        coefficients[10] = coefficients[3] - coefficients[6] * coefficients[0];
        coefficients[11] = -coefficients[7] * coefficients[0];
    }
    
    y04 = 0.0; y03 = 0.0; y02 = 0.0; y01 = 0.0;
    x04 = 0.0; x03 = 0.0; x02 = 0.0; x01 = 0.0;
    for (int k = (int)(n - 1); k >= 0; --k) // Backward filtering...
    {
        double xi = sx[k];
        double yi = coefficients[8] * x01 + coefficients[9] * x02 + coefficients[10] * x03 + coefficients[11] * x04 -
                    coefficients[4] * y01 - coefficients[5] * y02 - coefficients[6]  * y03 - coefficients[7]  * y04;
        sy[k] += yi;
        y04 = y03; y03 = y02; y02 = y01; y01 = yi;
        x04 = x03; x03 = x02; x02 = x01; x01 = xi;
    }
    
    q = 0;
    if (derivative == 0)
    {
        
        for (unsigned int i = 0, j = n - 1; i < m; ++i, --j)
            q += sy[i] + sy[j];
        q += sy[m];
    }
    else if (derivative == 1)
    {
        for (unsigned int i = 0, j = n - 1; i < m; ++i, --j)
        {
            double factor = sin((-(double)m + (double)i) / sigma);
            q += factor * (sy[i] - sy[j]);
        }
        q *= sigma * 1.6487212707001282; // exp(0.5) = 1.6487212707001282;
    }
    else if (derivative == 2)
    {
        for (unsigned int i = 0, j = n - 1; i < m; ++i, --j)
        {
            double factor = cos((-(double)m + (double)i) * 1.4142135623730951 / sigma); // sqrt(2.0) = 1.4142135623730951;
            q += factor * (sy[i] + sy[j]);
        }
        q = (q + sy[m]) * (-(sigma * sigma) / 2.0) * 2.718281828459045; // exp(1.0) = 2.718281828459045;
    }
    
    q = std::abs(q);
    coefficients[0] /= q;
    coefficients[1] /= q;
    coefficients[2] /= q;
    coefficients[3] /= q;
    
    if (derivative == 1) // For non-symmetric filters the negative coefficients are calculated differently.
    {
        coefficients[8]  = -(coefficients[1] - coefficients[4] * coefficients[0]);
        coefficients[9]  = -(coefficients[2] - coefficients[5] * coefficients[0]);
        coefficients[10] = -(coefficients[3] - coefficients[6] * coefficients[0]);
        coefficients[11] = coefficients[7] * coefficients[0];
    }
    else
    {
        coefficients[8]  = coefficients[1] - coefficients[4] * coefficients[0];
        coefficients[9]  = coefficients[2] - coefficients[5] * coefficients[0];
        coefficients[10] = coefficients[3] - coefficients[6] * coefficients[0];
        coefficients[11] = -coefficients[7] * coefficients[0];
    }
    
    // 3) Calculate the weight of the Gaussian filter.
    y04 = 0.0; y03 = 0.0; y02 = 0.0; y01 = 0.0;
               x03 = 0.0; x02 = 0.0; x01 = 0.0;
    for (unsigned int k = 0; k < n; ++k) // Forward filtering...
    {
        double xi = sx[k];
        double yi = coefficients[0] * xi + coefficients[1] * x01 + coefficients[2] * x02 + coefficients[3] * x03
                                         - coefficients[4] * y01 - coefficients[5] * y02 - coefficients[6] * y03 - coefficients[7] * y04;
        sy[k] = yi;
        y04 = y03; y03 = y02; y02 = y01; y01 = yi;
                   x03 = x02; x02 = x01; x01 = xi;
    }
    
    y04 = 0.0; y03 = 0.0; y02 = 0.0; y01 = 0.0;
    x04 = 0.0; x03 = 0.0; x02 = 0.0; x01 = 0.0;
    for (int k = (int)(n - 1); k >= 0; --k) // Backward filtering...
    {
        double xi = sx[k];
        double yi = coefficients[8] * x01 + coefficients[9] * x02 + coefficients[10] * x03 + coefficients[11] * x04 -
                    coefficients[4] * y01 - coefficients[5] * y02 - coefficients[6]  * y03 - coefficients[7]  * y04;
        sy[k] += yi;
        y04 = y03; y03 = y02; y02 = y01; y01 = yi;
        x04 = x03; x03 = x02; x02 = x01; x01 = xi;
    }
    weight = 0;
    for (unsigned int i = 0; i < n; ++i) weight += fabs(sy[i]);
    
    // 4) Free allocated memory.
    delete [] sx;
    delete [] sy;
    
    return coefficients;
}

double * Gaussian2DRecursiveKernel::setVlietYoungVerbeekCoefficients(unsigned int derivative, double sigma, unsigned int order, double &q, double &weight)
{
    std::complex<double> *d, prod, *previous_values, *values;
    double *coefficients, *module, *angle, b_const;
    unsigned int n, middle;
    double *sx, *sy, *queue;
    
    // 1) Set the poles of the recursive filter.
    d = new std::complex<double>[order];
    if (order == 3)
    {
        if (derivative == 0) d[0] = std::complex<double>(1.41650, 1.00829);
        else if (derivative == 1) d[0] = std::complex<double>(1.31553, 0.97057);
        else if (derivative == 2) d[0] = std::complex<double>(1.22886, 0.93058);
        d[1] = std::conj(d[0]);
        if (derivative == 0) d[2] = std::complex<double>(1.86543, 0.0);
        else if (derivative == 1) d[2] = std::complex<double>(1.77635, 0.0);
        else if (derivative == 2) d[2] = std::complex<double>(1.70493, 0.0);
    }
    else if (order == 4)
    {
        if (derivative == 0) d[0] = std::complex<double>(1.13228, 1.28114);
        else if (derivative == 1) d[0] = std::complex<double>(1.04185, 1.24034);
        else if (derivative == 2) d[0] = std::complex<double>(0.94570, 1.21064);
        d[1] = std::conj(d[0]);
        if (derivative == 0) d[2] = std::complex<double>(1.78534, 0.46763);
        else if (derivative == 1) d[2] = std::complex<double>(1.69747, 0.44790);
        else if (derivative == 2) d[2] = std::complex<double>(1.60161, 0.42647);
        d[3] = std::conj(d[2]);
    }
    else if (order == 5)
    {
        if (derivative == 0) d[0] = std::complex<double>(0.86430, 1.45389);
        else if (derivative == 1) d[0] = std::complex<double>(0.77934, 1.41423);
        else if (derivative == 2) d[0] = std::complex<double>(0.69843, 1.37655);
        d[1] = std::conj(d[0]);
        if (derivative == 0) d[2] = std::complex<double>(1.61433, 0.83134);
        else if (derivative == 1) d[2] = std::complex<double>(1.50941, 0.80828);
        else if (derivative == 2) d[2] = std::complex<double>(1.42631, 0.77399);
        d[3] = std::conj(d[2]);
        if (derivative == 0) d[4] = std::complex<double>(1.87504, 0.0);
        else if (derivative == 1) d[4] = std::complex<double>(1.77181, 0.0);
        else if (derivative == 2) d[4] = std::complex<double>(1.69668, 0.0);
    }
    else throwException("The specified order of the recursive filter is not implemented");
    
    coefficients = new double[order + 1];
    module = new double[order];
    angle = new double[order];
    for (unsigned int i = 0; i < order; ++i)
    {
        module[i] = std::abs(d[i]);
        angle[i] = std::arg(d[i]);
    }
    
    // 2) Rescale the poles to fit the selected sigma.
    q = 1.0;
    for (unsigned int iteration = 0; iteration < 1000; ++iteration)
    {
        double variance, current_module, current_angle;
        std::complex<double> complex_variance;
        
        complex_variance = std::complex<double>(0.0, 0.0);
        for (unsigned int i = 0; i < order; ++i)
        {
            current_module = pow(module[i], 1.0 / q);
            current_angle = angle[i] / q;
            complex_variance += 2.0 * std::polar(current_module, current_angle) * pow(current_module - std::polar(1.0, current_angle), -2.0);
        }
        variance = sqrt(std::real(complex_variance));
        
        q = q * sigma / variance;
        if (fabs(sigma - variance) < sigma * 1.0e-10) break;
    }
    
    for (unsigned int i = 0; i < order; ++i)
        d[i] = std::polar(pow(module[i], 1.0 / q), angle[i] / q);
    
    // 3) Calculate the coefficients of the rescaled poles.
    prod = d[0];
    for (unsigned int i = 1; i < order; ++i) prod *= d[i];
    b_const = 1.0 / std::real(prod);
    
    previous_values = new std::complex<double>[order];
    values = new std::complex<double>[order];
    for (unsigned int i = 0; i < order; ++i) { previous_values[i] = 1.0; values[i] = 0.0; }
    values[0] = 1.0;
    
    for (int index = (int)order - 1, k = 0; index >= 0; --index, ++k)
    {
        prod = values[0];
        for (unsigned int j = 1; j < order; ++j) prod += values[j];
        coefficients[index] = ((index % 2 == 1)?1.0:-1.0) * b_const * std::real(prod);
        //printf("--> %20.15f\n", coefficients[index]);
        for (unsigned int j = 0; j < order; ++j)
        {
            values[j] = d[j] * previous_values[j];
            //printf("%20.15f + i %20.15f\n", std::real(previous_values[j]), std::imag(previous_values[j]));
        }
        //printf("\n\n");
        
        previous_values[k] = 0;
        for (unsigned int j = k + 1; j < order; ++j)
        {
            prod = values[0];
            for (int m = 1; m < (int)j; ++m) prod += values[m];
            previous_values[j] = prod;
            //std::cout << j << "  " << previous_values[j] << std::endl;
        }
    }
    
    // 4) Set the alpha value of the recursive filter in the last coefficient.
    coefficients[order] = 1.0;
    for (unsigned int i = 0; i < order; ++i) coefficients[order] += coefficients[i]; // alpha = 1.0 + sum(b);
    
    // 5) Calculate the weight of the Gaussian filter.
    n = 1 + 2 * (unsigned int)(10.0 * sigma);
    middle = (n - 1) / 2;
    sx = new double[n];
    sy = new double[n];
    for (unsigned int i = 0; i < n; ++i) sx[i] = 0.0;
    sx[middle] = 1.0;
    queue = new double[order];
    
    for (unsigned int i = 0; i < order; ++i) queue[i] = 0.0;
    if (derivative == 0)
    {
        for (unsigned int k = 0, m = 0; k < n; ++k, ++m) // Forward filtering...
        {
            sy[k] = coefficients[order] * sx[k];
            for (unsigned int i = 0; i < order; ++i) sy[k] -= coefficients[i] * queue[i];
            for (unsigned int i = order - 1; i > 0; --i) queue[i] = queue[i - 1];
            queue[0] = sy[k];
        }
    }
    else if (derivative == 1)
    {
        sy[0] = 0.0;
        for (unsigned int k = 1, m = 0; k < n - 1; ++k, ++m) // Forward filtering...
        {
            sy[k] = coefficients[order] * (sx[k + 1] - sx[k - 1]) / 2.0;
            for (unsigned int i = 0; i < order; ++i) sy[k] -= coefficients[i] * queue[i];
            for (unsigned int i = order - 1; i > 0; --i) queue[i] = queue[i - 1];
            queue[0] = sy[k];
        }
        sy[n - 1] = 0.0;
    }
    else // Derivative == 2
    {
        sy[0] = 0.0;
        for (unsigned int k = 1, m = 0; k < n; ++k, ++m) // Forward filtering...
        {
            sy[k] = coefficients[order] * (sx[k] - sx[k - 1]);
            for (unsigned int i = 0; i < order; ++i) sy[k] -= coefficients[i] * queue[i];
            for (unsigned int i = order - 1; i > 0; --i) queue[i] = queue[i - 1];
            queue[0] = sy[k];
        }
    }
    
    for (unsigned int i = 0; i < order; ++i) queue[i] = 0.0;
    if (derivative == 2)
    {
        double previous = 0.0, aux;
        for (int k = (int)(n - 1), m = 0; k >= 0; --k, ++m) // Backward filtering...
        {
            aux = sy[k];
            sy[k] = coefficients[order] * (previous - sy[k]);
            previous = aux;
            for (unsigned int i = 0; i < order; ++i) sy[k] -= coefficients[i] * queue[i];
            for (unsigned int i = order - 1; i > 0; --i) queue[i] = queue[i - 1];
            queue[0] = sy[k];
        }
    }
    else // derivative == 0 OR 1.
    {
        for (int k = (int)(n - 1), m = 0; k >= 0; --k, ++m) // Backward filtering...
        {
            sy[k] = coefficients[order] * sy[k];
            for (unsigned int i = 0; i < order; ++i) sy[k] -= coefficients[i] * queue[i];
            for (unsigned int i = order - 1; i > 0; --i) queue[i] = queue[i - 1];
            queue[0] = sy[k];
        }
    }
    weight = 0;
    for (unsigned int i = 0; i < n; ++i) weight += fabs(sy[i]);
    
    // 6) Free allocated memory.
    delete [] d;
    delete [] module;
    delete [] angle;
    delete [] previous_values;
    delete [] values;
    delete [] queue;
    delete [] sx;
    delete [] sy;
    
    return coefficients;
}

