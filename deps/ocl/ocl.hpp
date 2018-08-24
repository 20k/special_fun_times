#ifndef OCL_HPP_INCLUDED
#define OCL_HPP_INCLUDED

#include <gl/glew.h>
#include <windows.h>
#include <gl/gl.h>
#include <gl/glext.h>
#include <cl/cl.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include "logging.hpp"
#include <vec/vec.hpp>
#include <assert.h>
#include <mutex>
#include <memory>

namespace cl
{
    struct kernel;

    bool supports_extension(cl_device_id device, const std::string& ext_name);

    cl_int get_platform_ids(cl_platform_id* clSelectedPlatformID);

    struct event
    {
        cl_event cevent = nullptr;
        bool invalid = true;

        bool finished()
        {
            cl_int status;

            clGetEventInfo(cevent, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, nullptr);

            return status == CL_COMPLETE;
        }

        bool bad() const
        {
            return invalid;
        }

        template<typename T, typename U>
        void set_completion_callback(T t, const U& data)
        {
            if(bad())
                return;

            cl_int ret = clSetEventCallback(cevent, CL_COMPLETE, t, (void*)data);
        }
    };

    ///hmm. problem is, if we destroy a read event while reading...
    ///then again, that's definitely bad code. But still, it might be better
    ///for the read to go nowhere and leak (free eventually using callbacks? hmm? reclaim?)
    ///rather than crash
    template<typename T>
    struct read_event : event
    {
        std::vector<T>* data = nullptr;

        void allocate_num(int s)
        {
            data = new std::vector<T>();

            data->resize(s);
        }

        void del()
        {
            if(data)
                delete data;
        }

        T* front_ptr()
        {
            if(data == nullptr)
                return nullptr;

            return &(*data)[0];
        }

        const T& operator[](std::size_t idx) const {return (*data)[idx];}
    };

    template<typename T>
    struct write_event : event
    {
        std::vector<T>* data = nullptr;

        void allocate_with(const std::vector<T>& dat)
        {
            data = new std::vector<T>();

            *data = dat;
        }

        void del()
        {
            if(data)
                delete data;
        }

        T* front_ptr()
        {
            if(data == nullptr)
                return nullptr;

            return &(*data)[0];
        }
    };

    inline
    void wait_for(const std::vector<event>& events)
    {
        if(events.size() == 0)
            return;

        std::vector<cl_event> clevents;

        for(const event& e : events)
        {
            if(e.bad())
                continue;

            clevents.push_back(e.cevent);
        }

        if(clevents.size() == 0)
            return;

        cl_int ret = clWaitForEvents(clevents.size(), &clevents[0]);

        if(ret != CL_SUCCESS)
        {
            std::cout << "Wait for events err " << ret << std::endl;
        }
    }

    inline
    void wait_for(const std::vector<event*>& events)
    {
        if(events.size() == 0)
            return;

        std::vector<cl_event> clevents;

        for(auto e : events)
        {
            if(e->bad())
                continue;

            clevents.push_back(e->cevent);
        }

        if(clevents.size() == 0)
            return;

        cl_int ret = clWaitForEvents(clevents.size(), &clevents[0]);

        if(ret != CL_SUCCESS)
        {
            std::cout << "Wait for events err " << ret << std::endl;
        }
    }

    struct program;
    struct kernel;

    struct context
    {
        cl_platform_id platform;
        cl_device_id devices[100] = {0};
        cl_device_id selected_device;
        cl_context ccontext;

        std::string device_name;

        std::vector<program> programs;
        std::map<std::string, kernel> kernels;

        context();

        void rebuild();

        cl_context& get(){return ccontext;}

        operator cl_context() {return ccontext;}

        void register_program(program& p);
    };

    struct program
    {
        context& saved_context;
        std::string saved_fname;

        cl_program cprogram;
        bool built = false;
        program(context& ctx, const std::string& fname, bool is_file = true);

        void rebuild();

        cl_program& get()
        {
            return cprogram;
        }

        cl_program get() const
        {
            return cprogram;
        }

        void operator=(const program& other)
        {
            cprogram = other.cprogram;
            built = other.built;
            saved_fname = other.saved_fname;
        }

        operator cl_program() {return cprogram;}

        void ensure_build();

        void build_with(context& ctx, const std::string& options);
    };

    struct kernel
    {
        cl_kernel ckernel;
        std::string name;
        bool loaded = false;
        //cl_uint work_size;

        kernel(program& p, const std::string& kname);
        kernel(cl_kernel&);
        kernel(){}

        cl_kernel& get(){return ckernel;}
    };

    struct arg_info
    {
        void* ptr = nullptr;
        int64_t size = 0;
    };

    struct args
    {
        std::vector<arg_info> arg_list;

        template<typename T>
        inline
        void push_back(T& val)
        {
            arg_info inf;
            inf.ptr = &val;
            inf.size = sizeof(T);

            arg_list.push_back(inf);
        }
    };

    struct buffer;

    template<typename T>
    struct map_info
    {
        buffer& v;
        T* ptr = nullptr;

        map_info(T* ptr, buffer& v) : ptr(ptr), v(v)
        {

        }
    };

    struct command_queue
    {
        cl_command_queue cqueue;
        context& ctx;

        command_queue(context& ctx);

        ///size defaults to -1 which means map the whole buffer
        void* map(buffer& v, cl_map_flags flag, int64_t size = -1);
        void unmap(buffer& v, void* ptr);

        template<typename T>
        map_info<T> map_type(buffer& v, cl_map_flags flag, int64_t size = -1)
        {
            void* ptr = map(v, flag, size);

            map_info<T> ret(ptr, v);

            return ret;
        }

        template<typename T>
        void unmap(map_info<T>& info)
        {
            unmap(info.v, info.ptr);
        }

        ///make this finally non stupid
        template<typename T, int dim>
        void exec(kernel& kname, args& pack, const T(&global_ws)[dim], const T(&local_ws)[dim], cl::event* evt = nullptr, std::vector<cl::event*> evts = std::vector<cl::event*>())
        {
            for(int i=0; i < (int)pack.arg_list.size(); i++)
            {
                clSetKernelArg(kname.ckernel, i, pack.arg_list[i].size, pack.arg_list[i].ptr);
            }

            size_t g_ws[dim] = {0};
            size_t l_ws[dim] = {0};

            for(int i=0; i < dim; i++)
            {
                l_ws[i] = local_ws[i];
                g_ws[i] = global_ws[i];

                if(l_ws[i] == 0)
                    continue;

                if((g_ws[i] % l_ws[i]) != 0)
                {
                    int rem = g_ws[i] % l_ws[i];

                    g_ws[i] -= rem;
                    g_ws[i] += l_ws[i];
                }

                if(g_ws[i] == 0)
                {
                    g_ws[i] += l_ws[i];
                }
            }

            cl_int err = CL_SUCCESS;

            std::vector<cl_event> events;

            for(cl::event* e : evts)
            {
                if(!e->bad())
                    events.push_back(e->cevent);
            }

            cl_event* out = nullptr;

            cl_event* first = nullptr;

            if(events.size() > 0)
                first = &events[0];

            if(evt != nullptr)
                out = &evt->cevent;

            #ifndef GPU_PROFILE
                err = clEnqueueNDRangeKernel(cqueue, kname.get(), dim, nullptr, g_ws, l_ws, events.size(), first, out);
            #else

            cl_event local;

            if(out == nullptr)
                out = &local;

            err = clEnqueueNDRangeKernel(cqueue, kname.get(), dim, nullptr, g_ws, l_ws, 0, nullptr, out);

            cl_ulong start;
            cl_ulong finish;

            block();

            clGetEventProfilingInfo(*out, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
            clGetEventProfilingInfo(*out, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &finish, nullptr);

            cl_ulong diff = finish - start;

            double ddiff = diff / 1000. / 1000.;

            std::cout << "kernel " << kname.name << " ms " << ddiff << std::endl;

            #endif // GPU_PROFILE

            if(err != CL_SUCCESS)
            {
                lg::log("clEnqueueNDRangeKernel Error with", kname.name);
                lg::log(err);
            }
            else
            {
                if(evt != nullptr)
                    evt->invalid = false;
            }
        }

        template<typename T, int dim>
        void exec(const std::string& kname, args& pack, const T(&global_ws)[dim], const T(&local_ws)[dim], cl::event* evt = nullptr, std::vector<cl::event*> evts = std::vector<cl::event*>())
        {
            ///needs to be made thread safe
            auto it = ctx.kernels.find(kname);

            if(it == ctx.kernels.end())
            {
                lg::log("Kernel with name ", kname, " not found");
                return;
            }

            return exec(it->second, pack, global_ws, local_ws, evt, evts);
        }

        template<typename T, int dim>
        void exec(const std::string& kname, args& pack, const vec<dim, T>& global_ws, const vec<dim, T>& local_ws, cl::event* evt = nullptr, std::vector<cl::event*> evts = std::vector<cl::event*>())
        {
            T g_ws[dim] = {0};
            T l_ws[dim] = {0};

            for(int i=0; i < dim; i++)
            {
                g_ws[i] = global_ws.v[i];
                l_ws[i] = local_ws.v[i];
            }

            return exec(kname, pack, g_ws, l_ws, evt, evts);
        }

        void block()
        {
            clFinish(cqueue);
        }

        operator cl_command_queue() {return cqueue;}
    };

    ///need a centralised way to invalidate all buffers
    ///associated with a context
    ///and then reallocate them
    ///NEEDS TO WORK WITH IMAGES WHEN WE DO RECONTEXTING
    ///OTHERWISE ALL GL INTEROP WILL BREAK
    struct buffer
    {
        cl_mem cmem;
        int64_t alloc_size = 0;
        context& ctx;

        size_t image_dims[3] = {1,1,1};
        int64_t image_dimensionality = 1;
        int byte_per_pixel = 1;

        enum internal_format
        {
            BUFFER,
            IMAGE,
        };

        internal_format format = BUFFER;

        buffer(context& ctx) : ctx(ctx) {}

        cl_mem& get()
        {
            return cmem;
        }

        template<typename T>
        read_event<T> async_read(command_queue& read_on, vec2i location, vec2i dim = {1, 1}, bool invert = false, std::vector<cl::event*> dependents = std::vector<cl::event*>())
        {
            read_event<T> data;

            ///TODO: WORK FOR BOTH
            //assert(format == IMAGE);

            if(format == BUFFER)
            {
                if(location.x() < 0 || location.x() >= alloc_size)
                    return data;

                data.allocate_num(dim.x());

                std::vector<cl_event> cl_events;

                for(cl::event* e : dependents)
                {
                    if(e->bad())
                        continue;

                    cl_events.push_back(e->cevent);
                }

                cl_event* first = nullptr;

                if(cl_events.size() > 0)
                    first = &cl_events[0];

                cl_int ret = clEnqueueReadBuffer(read_on, cmem, CL_FALSE, location.x(), dim.x() * sizeof(T), &(*data.data)[0], cl_events.size(), first, &data.cevent);

                if(ret != CL_SUCCESS)
                {
                    std::cout << "Error in async buffer read " << ret << std::endl;

                    data.invalid = true;
                }
                else
                {
                    data.invalid = false;
                }

                return data;
            }

            ///doesn't do a full invalid check yet with sizes
            if(location.x() < 0 || location.x() >= image_dims[0] || location.y() < 0 || location.y() >= image_dims[1])
            {
                data.invalid = true;

                return data;
            }

            if(format == IMAGE)
            {
                data.allocate_num(dim.x() * dim.y());

                if(invert)
                {
                    location.y() = image_dims[1] - location.y();
                }

                size_t origin[3] = {location.x(), location.y(), 0};
                size_t region[3] = {dim.x(), dim.y(), 1};

                cl_int ret = clEnqueueReadImage(read_on, cmem, CL_FALSE, origin, region, 0, 0, &(*data.data)[0], 0, nullptr, &data.cevent);

                if(ret != CL_SUCCESS)
                {
                    std::cout << "Error in async read " << ret << std::endl;

                    data.invalid = true;
                }
                else
                {
                    data.invalid = false;
                }

                return data;
            }

            return data;
        }

        ///for 2d writes.. double template me?
        template<typename T>
        write_event<T> async_write(command_queue& write_on, const std::vector<T>& in_dat, vec2i location = {0,0}, bool invert = false)
        {
            write_event<T> data;

            if(in_dat.size() == 0)
                return data;

            assert(format == BUFFER);

            if(location.x() < 0 || location.y() < 0)
                return data;

            data.allocate_with(in_dat);

            if(format == BUFFER)
            {
                assert(location.x() * in_dat.size() * sizeof(T) < alloc_size);

                cl_int ret = clEnqueueWriteBuffer(write_on, cmem, CL_FALSE, location.x(), in_dat.size() * sizeof(T), data.front_ptr(), 0, nullptr, &data.cevent);

                if(ret != CL_SUCCESS)
                {
                    std::cout << "Error in async write " << ret << std::endl;

                    data.invalid = true;
                }
                else
                {
                    data.invalid = false;
                }
            }

            return data;
        }

        void clear_to_zero(command_queue& write_on)
        {
            cl_uint zeros[4] = {0};

            if(format == BUFFER)
            {
                clEnqueueFillBuffer(write_on, cmem, &zeros[0], sizeof(cl_uchar), 0, alloc_size, 0, nullptr, nullptr);
            }
            else
            {
                size_t origin[3] = {0};

                ///thanks ieee! might be the only time this was said non sarcastically
                clEnqueueFillImage(write_on, cmem, &zeros[0], origin, image_dims, 0, nullptr, nullptr);
            }
        }

        void write_all(command_queue& write_on, const void* ptr)
        {
            cl_int val = CL_SUCCESS;

            if(format == BUFFER)
            {
                val = clEnqueueWriteBuffer(write_on, cmem, CL_TRUE, 0, alloc_size, ptr, 0, nullptr, nullptr);
            }
            else
            {
                size_t origin[3] = {0};

                val = clEnqueueWriteImage(write_on, cmem, CL_TRUE, origin, image_dims, 0, 0, ptr, 0, nullptr, nullptr);
            }

            if(val != CL_SUCCESS)
            {
                lg::log("Error writing to image", val);
            }
        }

        template<typename T>
        void write_all(command_queue& write_on, std::vector<T>& data)
        {
            if(data.size() == 0)
                return;

            write_all(write_on, &data[0]);

            /*clEnqueueWriteBuffer(write_on, cmem, CL_TRUE, 0, data.size() * sizeof(T), &data[0], 0, nullptr, nullptr);

            if(format == BUFFER)
            {
                clEnqueueWriteBuffer(write_on, cmem, CL_TRUE, 0, alloc_size, ptr, 0, nullptr, nullptr);
            }
            else
            {
                size_t origin[3] = {0};

                clEnqueueWriteImage(write_on, cmem, CL_TRUE, origin, image_dims, 0, 0, ptr, 0, nullptr, nullptr);
            }*/
        }

        template<typename T>
        std::vector<T> read_all(command_queue& read_on)
        {
            std::vector<T> ret;

            if(alloc_size == 0)
                return ret;

            ret.resize(alloc_size / sizeof(T));

            cl_int val = CL_SUCCESS;

            if(format == BUFFER)
            {
                val = clEnqueueReadBuffer(read_on, cmem, CL_TRUE, 0, alloc_size, &ret[0], 0, nullptr, nullptr);
            }
            else
            {
                size_t origin[3] = {0};

                val = clEnqueueReadImage(read_on, cmem, CL_TRUE, origin, image_dims, 0, 0, &ret[0], 0, nullptr, nullptr);
            }

            if(val != CL_SUCCESS)
            {
                lg::log("Error writing to image", val);
            }

            return ret;
        }

        void alloc_bytes(int bytes)
        {
            alloc_size = bytes;

            cl_int err;
            cmem = clCreateBuffer(ctx, CL_MEM_READ_WRITE, alloc_size, nullptr, &err);

            if(err != CL_SUCCESS)
            {
                lg::log("Error allocating buffer");
                return;
            }
        }

        template<typename T>
        void alloc_n(command_queue& write_on, const T* data, int num)
        {
            format = BUFFER;

            alloc_size = sizeof(T) * num;

            alloc_bytes(alloc_size);

            write_all(write_on, data);
        }

        template<typename T>
        void alloc(command_queue& write_on, const std::vector<T>& data)
        {
            format = BUFFER;

            if(data.size() == 0)
                return;

            alloc_n(write_on, &data[0], data.size());
        }

        template<typename T, int N>
        void alloc_n_img(command_queue& write_on, const T* data, const vec<N, int>& dims, cl_channel_order channel_order = CL_RGBA, cl_channel_type channel_type = CL_FLOAT)
        {
            format = IMAGE;
            image_dimensionality = N;

            int sum = 1;

            ///eg for a 2 dimension vector we get
            ///x * y
            ///or a 3d vec we get x * y * z;
            for(int i=0; i < N; i++)
            {
                sum = sum * dims.v[i];
            }

            alloc_size = sizeof(T) * sum;

            for(int i=0; i < N; i++)
            {
                image_dims[i] = dims.v[i];
            }

            cl_image_format format;
            format.image_channel_order = channel_order;
            format.image_channel_data_type = channel_type;

            ///TODO: REMOVE THIS CHECK
            static_assert(N == 2);

            cl_int err;
            cmem = clCreateImage2D(ctx, CL_MEM_READ_WRITE, &format, dims.x(), dims.y(), 0, nullptr, &err);

            if(err != CL_SUCCESS)
            {
                lg::log("Error creating image2d");
                return;
            }

            write_all(write_on, data);
        }

        template<typename T, int N>
        void alloc_img(command_queue& write_on, const std::vector<T>& data, const vec<N, int>& dims, cl_channel_order channel_order = CL_RGBA, cl_channel_type channel_type = CL_FLOAT)
        {
            format = IMAGE;

            if(data.size() == 0)
                return;

            /*for(const T& i : data)
            {
                std::cout << i << std::endl;
            }*/

            alloc_n_img(write_on, &data[0], dims, channel_order, channel_type);
        }

        void release()
        {
            clReleaseMemObject(cmem);
        }

        operator cl_mem() {return cmem;}
    };

    struct buffer_manager
    {
        std::map<buffer*, buffer*> buffers;

        template<typename U, typename... T>
        U* fetch(context& ctx, U* old, T&&... args)
        {
            if(old == nullptr)
            {
                U* buf = new U(ctx, std::forward<T>(args)...);

                buffers[buf] = buf;

                return buf;
            }
            else
            {
                return (U*)buffers[old];
            }
        }
    };

    struct cl_gl_storage_base
    {
        virtual void allocate_storage(){}
        virtual void deallocate_storage(){}

        virtual cl::cl_gl_storage_base* shallow_clone() const
        {
            return nullptr;
        }

        template<typename T>
        T* fetch_storage_as()
        {
            return (T*)fetch();
        }

    private:
        virtual void* fetch()
        {
            throw;
        }
    };

    template<typename T>
    struct cl_gl_storage : cl_gl_storage_base
    {
        T* storage = nullptr;

        cl_gl_storage()
        {
            storage = new T();
        }

        cl::cl_gl_storage_base* shallow_clone() const override
        {
            cl_gl_storage<T>* n = new cl_gl_storage<T>();
            n->storage = storage;
            return n;
        }

        void allocate_storage() override {storage = new T();}
        void deallocate_storage() override {if(storage){delete storage;} storage = nullptr;}

    private:
        void* fetch() override
        {
            return (void*)storage;
        }
    };

    struct cl_gl_interop_texture : buffer
    {
        cl_gl_interop_texture(context& ctx);

        void create_from_renderbuffer(GLuint renderbuf);
        void create_renderbuffer(int w, int h);
        void create_from_texture(GLuint tex, const cl_gl_storage_base& storage);

        int w, h;

        bool acquired = false;

        GLuint renderbuffer_id;

        cl_gl_storage_base* storage = nullptr;

        void gl_blit_raw(GLuint target, GLuint source);
        void gl_blit_me(GLuint target, command_queue& cqueue);

        ///to opencl
        void acquire(command_queue& cqueue);
        ///release to opengl
        void unacquire(command_queue& cqueue);
    };

    //kernel load_kernel(context& ctx, program& p, const std::string& name);
}

template<>
inline
void cl::args::push_back<cl::buffer>(cl::buffer& val)
{
    cl::arg_info inf;
    inf.ptr = &val.get();
    inf.size = sizeof(val.get());

    arg_list.push_back(inf);
}

template<>
inline
void cl::args::push_back<cl::buffer*>(cl::buffer*& val)
{
    cl::arg_info inf;
    inf.ptr = &val->get();
    inf.size = sizeof(val->get());

    arg_list.push_back(inf);
}

template<>
inline
void cl::args::push_back<cl::cl_gl_interop_texture*>(cl::cl_gl_interop_texture*& val)
{
    cl::arg_info inf;
    inf.ptr = &val->get();
    inf.size = sizeof(val->get());

    arg_list.push_back(inf);
}

#endif // OCL_HPP_INCLUDED
