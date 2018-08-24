#define IDX(a, b) a * rows + b
#define IDXC(a, b) a * columns + b

__kernel
void csk(__global double* dataset, __global double* distances, int rows, int columns)
{
    int index = get_global_id(0);

    int i = index / rows; ///canonical y
    int j = index % rows; ///canonical x

    if (i < j) return;

    if(index >= rows * rows)
        return;

    double dotProduct = 0;
    double magnitudeOne = 0;
    double magnitudeTwo = 0;
    for (int k = 0; k < columns; k++)
    {
        double d1 = dataset[IDXC(i, k)];
        double d2 = dataset[IDXC(j, k)];

        dotProduct += d1 * d2;
        magnitudeOne += d1 * d1;
        magnitudeTwo += d2 * d2;
    }
    double distance = 0;

    double divisor = sqrt(magnitudeOne * magnitudeTwo);

    if (divisor != 0)
    {
        distance = max((double)0., (double)(1. - (dotProduct / divisor)));
    }

    distances[IDX(i, j)] = distance;
    distances[IDX(j, i)] = distance;
}
