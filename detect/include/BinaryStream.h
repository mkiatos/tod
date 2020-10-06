#ifndef __BINARY_STREAM_H__
#define __BINARY_STREAM_H__

#include <iostream>
#include <vector>

#include <boost/cstdint.hpp>
#include <iterator>

#include <eigen3/Eigen/Core>

#include <opencv2/opencv.hpp>


// Lightweight portable binary stream on top of std::stream
// Portability is not guaranteed on platforms with char != 8 bits and those using floating point representation other that IEEE-754

class BinaryStream
{
public:

    // Initialize using a stream. The stream should be open in binary mode. By default data is saved in little endian but this may be modified when writing the stream

    BinaryStream(std::istream &strm, bool format_little_endian_ = true) ;
    BinaryStream(std::ostream &strm, bool format_little_endian_ = true) ;

    // Write operations

    void write(bool i) ;
    void write(int8_t i)  ;
    void write(uint8_t i) ;
    void write(int16_t i) ;
    void write(uint16_t i) ;
    void write(int32_t i) ;
    void write(uint32_t i) ;
    void write(int64_t i) ;
    void write(uint64_t i) ;
    void write(float i) ;
    void write(double i) ;
    void write(const std::string& str) ;
    void write(const char *str) ;

    // Google protobuf variable-length encoding (see http://code.google.com/apis/protocolbuffers/docs/encoding.html)
    // For signed integers do not cast them to unsigned but use the signed versions of the function belowvoid read_v(uint64_t &val) ;
    void write_v(uint64_t val) ;
    void write_v(uint32_t val) ;
    void write_v(int64_t val) ;
    void write_v(int32_t val) ;

    void write(const double *t, size_t n) ;
    void write(const float *t, size_t n)  ;
    void write(const int8_t *t, size_t n)   ;
    void write(const uint8_t *t, size_t n)  ;
    void write(const int16_t *t, size_t n)  ;
    void write(const uint16_t *t, size_t n) ;
    void write(const int32_t *t, size_t n)  ;
    void write(const uint32_t *t, size_t n) ;
    void write(const int64_t *t, size_t n)  ;
    void write(const uint64_t *t, size_t n) ;

    // read Operations

    void read(bool &i) ;
    void read(int8_t &i)  ;
    void read(uint8_t &i) ;
    void read(int16_t &i) ;
    void read(uint16_t &i) ;
    void read(int32_t &i) ;
    void read(uint32_t &i) ;
    void read(int64_t &i) ;
    void read(uint64_t &i) ;
    void read(float &i) ;
    void read(double &i) ;
    void read(std::string& str) ;

    void read_v(uint64_t &val) ;
    void read_v(int64_t &val) ;
    void read_v(uint32_t &val) ;
    void read_v(int32_t &val) ;

    void read(double *t, size_t n) ;
    void read(float *t, size_t n)  ;
    void read(int8_t *t, size_t n)   ;
    void read(uint8_t *t, size_t n)  ;
    void read(int16_t *t, size_t n)  ;
    void read(uint16_t *t, size_t n) ;
    void read(int32_t *t, size_t n)  ;
    void read(uint32_t *t, size_t n) ;
    void read(int64_t *t, size_t n)  ;
    void read(uint64_t *t, size_t n) ;

    void read_bytes(char *data, size_t size) ;
    void write_bytes(const char *data, size_t size) ;

    template<class T>
    BinaryStream & 	operator<< ( const T &i ) {
        write((const T &)i) ;
        return *this ;
    }

    template<class T>
    BinaryStream & 	operator >> ( T &i ) {
        read(i) ;
        return *this ;
    }

    template<class T>
    void read( std::vector<T> &v ) {
        uint64_t n ;
        read_v(n) ;

        v.resize(n) ;

        for(int i=0 ; i<n ; i++)
            read(v[i]) ;
    }

    template<class T>
    void write( const std::vector<T> &v ) {
        uint64_t n ;
        write_v(v.size()) ;

        for(int i=0 ; i<v.size() ; i++)
            write((T)v[i]) ;
    }


    template<class _S, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    void write(const Eigen::Matrix<_S, _Rows, _Cols, _Options, _MaxRows, _MaxCols> &m )
    {
        uint32_t rows = m.rows(), cols = m.cols();
        write(rows);
        write(cols);

        write_bytes((const char *)m.data(), rows * cols * sizeof(_S)) ;
    }

    template<class _S, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    void read(Eigen::Matrix<_S, _Rows, _Cols, _Options, _MaxRows, _MaxCols> &m )
    {
        uint32_t rows, cols ;

        read(rows) ;
        read(cols) ;

        m.resize(rows, cols);

        read_bytes((char *)m.data(), rows * cols * sizeof(_S)) ;
    }

    void write(const cv::Point2d &p) {
        write(p.x) ;
        write(p.y) ;
    }

    void read(cv::Point2d &p) {
        read(p.x) ;
        read(p.y) ;
    }

    void write(const cv::Mat &m) ;
    void read(cv::Mat &m) ;

private:
    std::istream *the_istream_ ;
    std::ostream *the_ostream_ ;
    bool compatible_endianess_ ;
} ;


#endif
