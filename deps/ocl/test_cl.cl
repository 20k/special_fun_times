__kernel
void test_kernel(__global int* test, __write_only image2d_t test_screen)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);

    int gw = get_image_width(test_screen);
    int gh = get_image_height(test_screen);

    float4 col = (float4){(float)idx / gw, (float)idy / gh, 0.f, 1.f};

    write_imagef(test_screen, (int2){idx, idy}, col);
}
