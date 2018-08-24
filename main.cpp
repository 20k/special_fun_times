#include <iostream>
#include <cl/cl.h>
#include <ocl/ocl.hpp>
#include <ocl/logging.hpp>
#include <SFML/Graphics.hpp>

using type_t = double;

std::vector<type_t> GenerateDoubleDataSet(int numElement, int numDimension)
{
    std::vector<type_t> dataSet;
    dataSet.resize(numElement * numDimension);

    for (int i = 0; i < numElement; i++)
    {
        //dataSet[i] = new double[numDimension];
        for (int j = 0; j < numDimension; j++)
        {
            dataSet[i * numDimension + j] = rand() % 5;
        }
    }

    return dataSet;
}

int main()
{
    lg::set_logfile("./out.txt");
    lg::redirect_to_stdout();

    sf::Context sf_ctx;

    cl::context ctx;

    cl::program program(ctx, "Kernels.cl");
    program.build_with(ctx, "");

    ctx.register_program(program);

    cl::command_queue cqueue(ctx);

    cl::buffer_manager buffer_manage;

    cl::buffer* b_dataset = buffer_manage.fetch<cl::buffer>(ctx, nullptr);
    cl::buffer* distance_out = buffer_manage.fetch<cl::buffer>(ctx, nullptr);

    cl::kernel kernel(program, "csk");

    int rows = 2000;
    int columns = 5000;

    auto data = GenerateDoubleDataSet(rows, columns);

    sf::Clock clk;

    b_dataset->alloc_bytes(data.size() * sizeof(data[0]));
    distance_out->alloc_bytes(rows * rows * sizeof(type_t));

    auto write_package = b_dataset->async_write(cqueue, data);

    cl::args arg_pack;
    arg_pack.push_back(b_dataset);
    arg_pack.push_back(distance_out);
    arg_pack.push_back(rows);
    arg_pack.push_back(columns);

    cl::event evt;
    cqueue.exec(kernel, arg_pack, {rows * rows}, {256}, &evt, {&write_package});

    cl::read_event<type_t> read_package = distance_out->async_read<type_t>(cqueue, (vec2i){0, 0}, (vec2i){rows * rows, 0}, false, {&evt});

    cl::wait_for({read_package});

    std::cout << "done in " << clk.getElapsedTime().asMicroseconds() / 1000. << std::endl;

    return 0;
}
