//
// Created by auv on 7/19/19.
//

#include "common_utils/NumpyConverter.h"
#include <iostream>

#ifndef CV_MAX_DIM
#define CV_MAX_DIM 32
#endif

static void* init()
{
    import_array();
    return NULL;
}

static int failmsg(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}

class PyAllowThreads
{
public:
    PyAllowThreads() : _state(PyEval_SaveThread()) {}
    ~PyAllowThreads()
    {
        PyEval_RestoreThread(_state);
    }
private:
    PyThreadState* _state;
};

class PyEnsureGIL
{
public:
    PyEnsureGIL() : _state(PyGILState_Ensure()) {}
    ~PyEnsureGIL()
    {
        //std::cout << "releasing"<< std::endl;
        PyGILState_Release(_state);
    }
private:
    PyGILState_STATE _state;
};

using namespace cv;

static PyObject* failmsgp(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}


class NumpyAllocator : public MatAllocator
{
public:
#if ( CV_MAJOR_VERSION < 3)
    NumpyAllocator() {}
    ~NumpyAllocator() {}

    void allocate(int dims, const int* sizes, int type, int*& refcount,
                  uchar*& datastart, uchar*& data, size_t* step)
    {

        //PyEnsureGIL gil;

        int depth = CV_MAT_DEPTH(type);
        int cn = CV_MAT_CN(type);

        const int f = (int)(sizeof(size_t)/8);
        int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
                                                   depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
                                                                                  depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
                                                                                                              depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;
        int i;

        npy_intp _sizes[CV_MAX_DIM+1];
        for( i = 0; i < dims; i++ )
        {
            _sizes[i] = sizes[i];
        }

        if( cn > 1 )
        {
            _sizes[dims++] = cn;
        }
        PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
        if(!o)
        {

            CV_Error_(CV_StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
        }
        refcount = refcountFromPyObject(o);

        npy_intp* _strides = PyArray_STRIDES(o);
        for( i = 0; i < dims - (cn > 1); i++ )
            step[i] = (size_t)_strides[i];

        datastart = data = (uchar*)PyArray_DATA(o);

    }

    void deallocate(int* refcount, uchar*, uchar*)
    {
        //PyEnsureGIL gil;
        if( !refcount )
            return;
        PyObject* o = pyObjectFromRefcount(refcount);
        Py_INCREF(o);
        Py_DECREF(o);
    }
#else

    NumpyAllocator() {
        stdAllocator = Mat::getStdAllocator();
    }
    ~NumpyAllocator() {
        }

        UMatData* allocate(PyObject* o, int dims, const int* sizes, int type,
                           size_t* step) const {
            UMatData* u = new UMatData(this);
            u->data = u->origdata = (uchar*) PyArray_DATA((PyArrayObject*) o);
            npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
            for (int i = 0; i < dims - 1; i++)
                step[i] = (size_t) _strides[i];
            step[dims - 1] = CV_ELEM_SIZE(type);
            u->size = sizes[0] * step[0];
            u->userdata = o;
            return u;
        }

        UMatData* allocate(int dims0, const int* sizes, int type, void* data,
                           size_t* step, AccessFlag flags, UMatUsageFlags usageFlags) const {
            if (data != 0) {
                CV_Error(Error::StsAssert, "The data should normally be NULL!");
                // probably this is safe to do in such extreme case
                return stdAllocator->allocate(dims0, sizes, type, data, step, flags,
                                              usageFlags);
            }
            PyEnsureGIL gil;

            int depth = CV_MAT_DEPTH(type);
            int cn = CV_MAT_CN(type);
            const int f = (int) (sizeof(size_t) / 8);
            int typenum =
                    depth == CV_8U ? NPY_UBYTE :
                    depth == CV_8S ? NPY_BYTE :
                    depth == CV_16U ? NPY_USHORT :
                    depth == CV_16S ? NPY_SHORT :
                    depth == CV_32S ? NPY_INT :
                    depth == CV_32F ? NPY_FLOAT :
                    depth == CV_64F ?
                    NPY_DOUBLE :
                    f * NPY_ULONGLONG + (f ^ 1) * NPY_UINT;
            int i, dims = dims0;
            cv::AutoBuffer<npy_intp> _sizes(dims + 1);
            for (i = 0; i < dims; i++)
                _sizes[i] = sizes[i];
            if (cn > 1)
                _sizes[dims++] = cn;
            PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
            if (!o)
                CV_Error_(Error::StsError,
                          ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
            return allocate(o, dims0, sizes, type, step);
        }

        bool allocate(UMatData* u, AccessFlag accessFlags,
                      UMatUsageFlags usageFlags) const {
            return stdAllocator->allocate(u, accessFlags, usageFlags);
        }

        void deallocate(UMatData* u) const {
            if (u) {
                PyEnsureGIL gil;
                PyObject* o = (PyObject*) u->userdata;
                Py_XDECREF(o);
                delete u;
            }
        }

        const MatAllocator* stdAllocator;
#endif
};

NumpyAllocator g_numpyAllocator;

NDArrayConverter::NDArrayConverter() { init(); }

void* NDArrayConverter::init()
{
    import_array();
    return NULL;
}


cv::Mat NDArrayConverter::toMat( PyObject *o)
{
    cv::Mat m;

    if(!o || o == Py_None)
    {
        if( !m.data )
            m.allocator = &g_numpyAllocator;
    }

    if( !PyArray_Check(o) )
    {
        failmsg("toMat: Object is not a numpy array");
    }

    int typenum = PyArray_TYPE(o);
    int type = typenum == NPY_UBYTE ? CV_8U : typenum == NPY_BYTE ? CV_8S :
                                              typenum == NPY_USHORT ? CV_16U : typenum == NPY_SHORT ? CV_16S :
                                                                               typenum == NPY_INT || typenum == NPY_LONG ? CV_32S :
                                                                               typenum == NPY_FLOAT ? CV_32F :
                                                                               typenum == NPY_DOUBLE ? CV_64F : -1;

    if( type < 0 )
    {
        failmsg("toMat: Data type = %d is not supported", typenum);
    }

    int ndims = PyArray_NDIM(o);

    if(ndims >= CV_MAX_DIM)
    {
        failmsg("toMat: Dimensionality (=%d) is too high", ndims);
    }

    int size[CV_MAX_DIM+1];
    size_t step[CV_MAX_DIM+1], elemsize = CV_ELEM_SIZE1(type);
    const npy_intp* _sizes = PyArray_DIMS(o);
    const npy_intp* _strides = PyArray_STRIDES(o);
    bool transposed = false;

    for(int i = 0; i < ndims; i++)
    {
        size[i] = (int)_sizes[i];
        step[i] = (size_t)_strides[i];
    }

    if( ndims == 0 || step[ndims-1] > elemsize ) {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }

    if( ndims >= 2 && step[0] < step[1] )
    {
        std::swap(size[0], size[1]);
        std::swap(step[0], step[1]);
        transposed = true;
    }

    if( ndims == 3 && size[2] <= CV_CN_MAX && step[1] == elemsize*size[2] )
    {
        ndims--;
        type |= CV_MAKETYPE(0, size[2]);
    }

    if( ndims > 2)
    {
        failmsg("toMat: Object has more than 2 dimensions");
    }

    m = Mat(ndims, size, type, PyArray_DATA(o), step);

    if( m.data )
    {
#if ( CV_MAJOR_VERSION < 3)
        m.refcount = refcountFromPyObject(o);
        m.addref(); // protect the original numpy array from deallocation
        // (since Mat destructor will decrement the reference counter)
#else
        m.u = g_numpyAllocator.allocate(o, ndims, size, type, step);
        m.addref();
        Py_INCREF(o);
        //m.u->refcount = *refcountFromPyObject(o);
#endif

    };
    m.allocator = &g_numpyAllocator;

    if( transposed )
    {
        Mat tmp;
        tmp.allocator = &g_numpyAllocator;
        transpose(m, tmp);
        m = tmp;
    }
    return m;
}

PyObject* NDArrayConverter::toNDArray(const cv::Mat& m)
{
    if( !m.data )
        Py_RETURN_NONE;
    Mat temp;
    Mat *p = (Mat*)&m;
#if ( CV_MAJOR_VERSION < 3)
    if(!p->refcount || p->allocator != &g_numpyAllocator)
    {
        temp.allocator = &g_numpyAllocator;
        m.copyTo(temp);
        p = &temp;
    }
    p->addref();
    return pyObjectFromRefcount(p->refcount);
#else
    if(!p->u || p->allocator != &g_numpyAllocator)
    {
        temp.allocator = &g_numpyAllocator;
        m.copyTo(temp);
        p = &temp;
    }
    //p->addref();
    //return pyObjectFromRefcount(&p->u->refcount);
    PyObject* o = (PyObject*) p->u->userdata;
    Py_INCREF(o);
    return o;
#endif

}
