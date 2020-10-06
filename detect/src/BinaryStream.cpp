#include "BinaryStream.h"
#include <cassert>
#include <iterator>
#include <string.h>

// boost implementation of C++11 operation
#include <boost/algorithm/cxx11/copy_n.hpp>
using boost::algorithm::copy_n ;

#include <stdexcept>

using namespace std ;


#ifdef __CHAR_BIT__ 
#if __CHAR_BIT__ != 8 
#error "unsupported char size"
#endif
#else
#ifdef CHAR_BIT
#if CHAR_BIT != 8 
#error "unsupported char size"
#endif
#endif
#endif

enum
{
    O32_LITTLE_ENDIAN = 0x03020100ul,
    O32_BIG_ENDIAN = 0x00010203ul,
    O32_PDP_ENDIAN = 0x01000302ul
};

static const union { unsigned char bytes[4]; uint32_t value; } o32_host_order =
{ { 0, 1, 2, 3 } };



BinaryStream::BinaryStream(std::istream &strm, bool format_little_endian): the_istream_(&strm), the_ostream_(NULL)
{
    bool platform_is_little_endian_ = ( o32_host_order.value == 0x03020100ul ) ;
    compatible_endianess_ =  ( platform_is_little_endian_ == format_little_endian ) ;
}

BinaryStream::BinaryStream(std::ostream &strm, bool format_little_endian):
    the_ostream_(&strm), the_istream_(NULL)
{
    bool platform_is_little_endian_ = ( o32_host_order.value == 0x03020100ul ) ;
    compatible_endianess_ =  ( platform_is_little_endian_ == format_little_endian ) ;
}

void BinaryStream::write_bytes(const char *data, size_t sz)
{
    assert( the_ostream_ ) ;
    the_ostream_->write(data, sz);
}


void BinaryStream::read_bytes(char *data, size_t sz)
{
    assert( the_istream_ );
    the_istream_->read(data, sz);
}

void BinaryStream::write(const string& str)
{
    assert( the_ostream_ ) ;
    uint32_t len = str.length();

    write(len);
    write_bytes(str.c_str(), len);
}

void BinaryStream::write(const char *str)
{
    assert( the_ostream_ ) ;
    uint32_t len = strlen(str);

    write(len);
    write_bytes(str, len);
}

void BinaryStream::write_v(uint64_t uv)
{
    assert( the_ostream_ ) ;

    while ( uv > 0x7F ) {
        the_ostream_->put((static_cast<uint8_t>(uv) & 0x7F) | 0x80);
        uv >>= 7 ;
    }

    the_ostream_->put(static_cast<uint8_t>(uv) & 0x7F) ;
}

void BinaryStream::write_v(int64_t uv)
{
    write_v((uint64_t)((uv << 1) ^ (uv >> 63))) ;
}

void BinaryStream::write_v(uint32_t uv)
{
    assert( the_ostream_ ) ;

    while ( uv > 0x7F ) {
        the_ostream_->put((static_cast<uint8_t>(uv) & 0x7F) | 0x80);
        uv >>= 7 ;
    }

    the_ostream_->put(static_cast<uint8_t>(uv) & 0x7F) ;
}

void BinaryStream::write_v(int32_t uv)
{
    write_v((uint32_t)((uv << 1) ^ (uv >> 31))) ;
}


void BinaryStream::read_v(uint64_t &uv)
{
    assert( the_istream_ ) ;

    uv = UINT64_C(0);
    unsigned int bits = 0;
    uint32_t b ;

    while ( 1 )
    {
        char c ;
        the_istream_->get(c) ;
        b = c ;

        uv |= ( b & 0x7F ) << bits;

        if ( !(b & 0x80) ) break ;

        bits += 7 ;

        if ( bits > 63 ) throw runtime_error("Variable length integer is too long") ;
    }

}

void BinaryStream::read_v(uint32_t &uv)
{
    assert( the_istream_ ) ;

    uv = UINT32_C(0);
    unsigned int bits = 0;
    uint32_t b ;

    while ( 1 )
    {

        char c ;
        the_istream_->get(c) ;
        b = c ;

        uv |= ( b & 0x7F ) << bits;

        if ( !( b & 0x80 ) ) break ;

        bits += 7 ;

        if ( bits > 31 ) throw runtime_error("Variable length integer is too long") ;
    }

}

void BinaryStream::read_v(int64_t &uv)
{
    uint64_t v ;
    read_v(v) ;
    uv = (v >> 1) ^ -static_cast<int64_t>(v & 1) ;
}

void BinaryStream::read_v(int32_t &uv)
{
    uint32_t v ;
    read_v(v) ;
    uv = (v >> 1) ^ -static_cast<int32_t>(v & 1) ;
}


void BinaryStream::read(string& str)
{
    assert( the_istream_ ) ;
    uint32_t count = 0;

    read(count) ;
    str.reserve(count) ;

    std::istreambuf_iterator<char> it(*the_istream_) ;

	for( int i=0 ; i<count ; i++ )
		str.push_back(*it++) ;

}

static void byte_swap_32(uint32_t &data)
{
    union u {uint32_t v; uint8_t c[4];};
    u un, vn;
    un.v = data ;
    vn.c[0]=un.c[3];
    vn.c[1]=un.c[2];
    vn.c[2]=un.c[1];
    vn.c[3]=un.c[0];
    data = vn.v ;
}

static void byte_swap_64(uint64_t &data)
{
    union u {uint64_t v; uint8_t c[8];};
    u un, vn;
    un.v = data ;
    vn.c[0]=un.c[7];
    vn.c[1]=un.c[6];
    vn.c[2]=un.c[5];
    vn.c[3]=un.c[4];
    vn.c[4]=un.c[3];
    vn.c[5]=un.c[2];
    vn.c[6]=un.c[1];
    vn.c[7]=un.c[0];
    data = vn.v ;
}

static void byte_swap_16(uint16_t &nValue)
{
    nValue = ((( nValue>> 8)) | (nValue << 8));
}

void BinaryStream::write(bool i) {
    write((uint8_t)i) ;
}

void BinaryStream::write(int8_t i) {
    assert( the_ostream_ ) ;
    the_ostream_->write((const char *)&i, 1) ;
}

void BinaryStream::write(uint8_t i) {
    assert( the_ostream_ ) ;
    the_ostream_->write((const char *)&i, 1) ;
}

void BinaryStream::write(int16_t i) {
    write((uint16_t)i) ;
}

void BinaryStream::write(uint16_t i)
{
    assert( the_ostream_ ) ;

    if ( compatible_endianess_ )
        the_ostream_->write((const char *)&i, 2) ;
    else {
        byte_swap_16(i) ;
        the_ostream_->write((const char *)&i, 2) ;
    }
}

void BinaryStream::write(int32_t i) {
    write((uint32_t)i) ;
}

void BinaryStream::write(uint32_t i)
{
    assert( the_ostream_ ) ;

    if ( compatible_endianess_ )
        the_ostream_->write((const char *)&i, 4) ;
    else {
        byte_swap_32(i) ;
        the_ostream_->write((const char *)&i, 4) ;
    }
}

void BinaryStream::write(int64_t i) {
    write((uint64_t)i) ;
}

void BinaryStream::write(uint64_t i)
{
    assert( the_ostream_ ) ;

    if ( compatible_endianess_ )
        the_ostream_->write((const char *)&i, 8) ;
    else {
        byte_swap_64(i) ;
        the_ostream_->write((const char *)&i, 8) ;
    }
}

void BinaryStream::write(float val) {
    union {float f; uint32_t i;} ;
    f = val ;
    write(i) ;
}

void BinaryStream::write(double val) {
    union {double f; uint64_t i;} ;
    f = val ;
    write(i) ;
}

void BinaryStream::write(const double *t, size_t n) {
    assert( the_ostream_ ) ;

    if ( compatible_endianess_ )
        the_ostream_->write((const char *)t, n * 8) ;
    else {
        uint64_t *p = (uint64_t *)t ;
        for(int i=0 ; i<n ; i++)
        {
            uint64_t s = *p++ ;
            byte_swap_64(s) ;
            the_ostream_->write((const char *)&s, 8) ;
        }
    }
}

void BinaryStream::write(const float *t, size_t n) {
    assert( the_ostream_ ) ;

    if ( compatible_endianess_ )
        the_ostream_->write((const char *)t, n * 4) ;
    else {
        uint32_t *p = (uint32_t *)t ;
        for(int i=0 ; i<n ; i++)
        {
            uint32_t s = *p++ ;
            byte_swap_32(s) ;
            the_ostream_->write((const char *)&s, 4) ;
        }
    }
}

void BinaryStream::write(const uint8_t *t, size_t n) {
    assert( the_ostream_ ) ;

    the_ostream_->write((const char *)t, n) ;
}

void BinaryStream::write(const int8_t *t, size_t n) {
    write((uint8_t *)t, n) ;
}

void BinaryStream::write(const uint16_t *t, size_t n) {
    assert( the_ostream_ ) ;

    if ( compatible_endianess_ )
        the_ostream_->write((const char *)t, n * 2) ;
    else {
        const uint16_t *p = t ;
        for(int i=0 ; i<n ; i++)
        {
            uint16_t s = *p++ ;
            byte_swap_16(s) ;
            the_ostream_->write((const char *)&s, 2) ;
        }
    }
}

void BinaryStream::write(const int16_t *t, size_t n) {
    write((uint16_t *)t, n) ;
}

void BinaryStream::write(const uint32_t *t, size_t n) {
    assert( the_ostream_ ) ;

    if ( compatible_endianess_ )
        the_ostream_->write((const char *)t, n * 4) ;
    else {
        const uint32_t *p = t ;
        for(int i=0 ; i<n ; i++)
        {
            uint32_t s = *p++ ;
            byte_swap_32(s) ;
            the_ostream_->write((const char *)&s, 4) ;
        }
    }
}

void BinaryStream::write(const int32_t *t, size_t n) {
    write((uint32_t *)t, n) ;
}

void BinaryStream::write(const uint64_t *t, size_t n) {
    assert( the_ostream_ ) ;

    if ( compatible_endianess_ )
        the_ostream_->write((const char *)t, n * 8) ;
    else {
        const uint64_t *p = t ;
        for(int i=0 ; i<n ; i++)
        {
            uint64_t s = *p++ ;
            byte_swap_64(s) ;
            the_ostream_->write((const char *)&s, 8) ;
        }
    }
}

void BinaryStream::write(const int64_t *t, size_t n) {
    write((uint64_t *)t, n) ;
}


// Read Operations

void BinaryStream::read(bool &i) {
    assert( the_istream_ ) ;

    char s ;
    the_istream_->get(s) ;
    i = s ;
}


void BinaryStream::read(uint8_t &i)          {
    assert( the_istream_ ) ;

    char s ;
    the_istream_->get(s) ;
    i = (uint8_t)s ;
}

void BinaryStream::read(int8_t &i)          {
    read((uint8_t &)i) ;
}


void BinaryStream::read(uint16_t &i) {
    assert( the_istream_ ) ;

    if ( compatible_endianess_ )
        the_istream_->read((char *)&i, 2) ;
    else {
        the_istream_->read((char *)&i, 2) ;
        byte_swap_16(i) ;
    }
}

void BinaryStream::read(int16_t &i)  {
    read((uint16_t &)i) ;
}


void BinaryStream::read(uint32_t &i) {
    assert( the_istream_ ) ;

    if ( compatible_endianess_ )
        the_istream_->read((char *)&i, 4) ;
    else {
        the_istream_->read((char *)&i, 4) ;
        byte_swap_32(i) ;
    }
}

void BinaryStream::read(int32_t &i)  {
    read((uint32_t &)i) ;
}


void BinaryStream::read(uint64_t &i) {
    assert( the_istream_ ) ;

    if ( compatible_endianess_ )
        the_istream_->read((char *)&i, 8) ;
    else {
        the_istream_->read((char *)&i, 8) ;
        byte_swap_64(i) ;
    }
}

void BinaryStream::read(int64_t &i)  {
    read((uint64_t &)i) ;
}

void BinaryStream::read(float &val) {
    read((uint32_t &)val) ;
}

void BinaryStream::read(double &val) {
    read((uint64_t &)val) ;
}

void BinaryStream::read(double *t, size_t n) {
    assert(the_istream_) ;

    if ( compatible_endianess_ )
        the_istream_->read((char *)t, n * 8) ;
    else {
        uint64_t *p = (uint64_t *)t ;
        for(int i=0 ; i<n ; i++)
        {
            the_istream_->read((char *)p, 8) ;
            byte_swap_64(*p++) ;
        }
    }
}

void BinaryStream::read(float *t, size_t n) {
    assert(the_istream_) ;

    if ( compatible_endianess_ )
        the_istream_->read((char *)t, n * 4) ;
    else {
        uint32_t *p = (uint32_t *)t ;
        for(int i=0 ; i<n ; i++)
        {
            the_istream_->read((char *)p, 4) ;
            byte_swap_32(*p++) ;
        }
    }
}

void BinaryStream::read(uint8_t *t, size_t n) {
    assert(the_istream_) ;

    the_istream_->read((char *)t, n) ;
}

void BinaryStream::read(int8_t *t, size_t n) {
    read((uint8_t *)t, n) ;
}

void BinaryStream::read(uint16_t *t, size_t n) {
    assert(the_istream_) ;

    if ( compatible_endianess_ )
        the_istream_->read((char *)t, n * 2) ;
    else {
        uint16_t *p = t ;
        for(int i=0 ; i<n ; i++)
        {
            the_istream_->read((char *)p, 2) ;
            byte_swap_16(*p++) ;
        }
    }
}

void BinaryStream::read(int16_t *t, size_t n) {
    read((uint16_t *)t, n) ;
}

void BinaryStream::read(uint32_t *t, size_t n) {
    assert(the_istream_) ;

    if ( compatible_endianess_ )
        the_istream_->read((char *)t, n * 4) ;
    else {
        uint32_t *p = t ;
        for(int i=0 ; i<n ; i++)
        {
            the_istream_->read((char *)p, 4) ;
            byte_swap_32(*p++) ;
        }
    }
}

void BinaryStream::read(int32_t *t, size_t n) {
    read((uint32_t *)t, n) ;
}

void BinaryStream::read(uint64_t *t, size_t n) {
    assert(the_istream_) ;

    if ( compatible_endianess_ )
        the_istream_->read((char *)t, n * 8) ;
    else {
        uint64_t *p = t ;
        for(int i=0 ; i<n ; i++)
        {
            the_istream_->read((char *)p, 8) ;
            byte_swap_64(*p++) ;
        }
    }
}

void BinaryStream::read(int64_t *t, size_t n) {
    read((uint64_t *)t, n) ;
}


void BinaryStream::write(const cv::Mat &m)
{
    size_t elem_size = m.elemSize();
    size_t elem_type = m.type();

    write((int)m.cols) ;
    write((int)m.rows) ;
    write(elem_size);
    write(elem_type);

    const size_t data_size = m.cols * m.rows * elem_size;
    write_bytes((const char *)m.ptr(), data_size) ;
}


void BinaryStream::read(cv::Mat &m)
{
    int cols, rows;
    size_t elem_size, elem_type;

    read(cols) ; read(rows) ;
    read(elem_size) ; read(elem_type) ;

    m.create(rows, cols, elem_type);

    size_t data_size = m.cols * m.rows * elem_size;
    read_bytes((char *)m.ptr(), data_size) ;

}

