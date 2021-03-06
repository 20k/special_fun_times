#include <iostream>
#include <cl/cl.h>
#include "ocl.hpp"
#include "logging.hpp"
#include <SFML/Graphics.hpp>

int main()
{
    lg::set_logfile("./out.txt");
    lg::redirect_to_stdout();

    lg::log("Test");

    sf::RenderWindow win;
    win.create(sf::VideoMode(800, 600), "Test");

    cl::context ctx;

    cl::program program(ctx, "test_cl.cl");
    program.build_with(ctx, "");

    //cl::kernel test_kernel(program, "test_kernel");

    cl::command_queue cqueue(ctx);

    cl::buffer_manager buffer_manage;

    cl::buffer* buf = buffer_manage.fetch<cl::buffer>(ctx, nullptr);

    std::vector<int> data;

    for(int i=0; i < 800*600; i++)
    {
        data.push_back(i);
    }

    buf->alloc(cqueue, data);

    cl::cl_gl_interop_texture* interop = buffer_manage.fetch<cl::cl_gl_interop_texture>(ctx, nullptr, win.getSize().x, win.getSize().y);
    interop->acquire(cqueue);

    cl::args none;
    none.push_back(buf);
    none.push_back(interop);

    cqueue.exec(program, "test_kernel", none, {128}, {16});

    cqueue.block();

    while(win.isOpen())
    {
        sf::Event event;

        while(win.pollEvent(event))
        {

        }

        cqueue.exec(program, "test_kernel", none, {800, 600}, {16, 16});
        cqueue.block();

        interop->gl_blit_me(0, cqueue);

        win.display();
        win.clear();
    }

    return 0;
}
