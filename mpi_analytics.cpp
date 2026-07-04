//This code was developed by our team.
// We used limited AI assistance (for syntax correction and optimization suggestions).
// All logic and implementation decisions were made and verified by team members.


#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>

std::vector<double> movingAverage(const std::vector<double>& data, int window) {
    std::vector<double> result;

    if (data.size() < window) return result;

    for (size_t i = 0; i <= data.size() - window; i++) {
        double sum = 0.0;
        for (int j = 0; j < window; j++) {
            sum += data[i + j];
        }
        result.push_back(sum / window);
    }

    return result;
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long N = 1000000;
    if (rank == 0 && argc > 1)
        N = std::stoll(argv[1]);

    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    std::vector<double> data;

    if (rank == 0) {
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> dist(0.0, 10000.0);

        data.resize(N);
        for (long long i = 0; i < N; i++)
            data[i] = dist(rng);
    }

    long long base = N / size;
    long long rem = N % size;

    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);

    for (int i = 0; i < size; i++)
        sendcounts[i] = base + (i < rem ? 1 : 0);

    displs[0] = 0;
    for (int i = 1; i < size; i++)
        displs[i] = displs[i - 1] + sendcounts[i - 1];

    long long localN = sendcounts[rank];
    std::vector<double> localData(localN);

    MPI_Scatterv(
        rank == 0 ? data.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_DOUBLE,
        localData.data(),
        localN,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    double localSum = 0, localSq = 0;
    double localMin = 1e18, localMax = -1e18;

    for (double v : localData) {
        localSum += v;
        localSq += v * v;
        localMin = std::min(localMin, v);
        localMax = std::max(localMax, v);
    }

    double globalSum, globalSq, globalMin, globalMax;

    MPI_Reduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localSq, &globalSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double mean = 0, var = 0, stddev = 0;

    if (rank == 0) {
        mean = globalSum / N;
        var = globalSq / N - mean * mean;
        stddev = std::sqrt(var);
    }

    MPI_Bcast(&mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int bins = 10;
    std::vector<int> localHist(bins, 0);

    for (double v : localData) {
        int b = std::min(bins - 1, (int)(v / (10000.0 / bins)));
        localHist[b]++;
    }

    std::vector<int> globalHist(bins, 0);

    MPI_Reduce(localHist.data(), globalHist.data(), bins, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    std::vector<double> allData;

    if (rank == 0) allData.resize(N);

    MPI_Gather(localData.data(), localN, MPI_DOUBLE,
               allData.data(), localN, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0)
        std::sort(allData.begin(), allData.end());

    std::vector<double> ma;

    if (rank == 0) {
        int window = 5;
        ma = movingAverage(allData, window);
    }

    int localOut = 0;

    for (double v : localData) {
        if (std::fabs((v - mean) / stddev) > 3)
            localOut++;
    }

    int globalOut;

    MPI_Reduce(&localOut, &globalOut, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    std::vector<double> local2(localN);

    for (long long i = 0; i < localN; i++)
        local2[i] = localData[i] * 0.5;

    double localMean2 = std::accumulate(local2.begin(), local2.end(), 0.0) / localN;

    double lnum = 0, lden1 = 0, lden2 = 0;

    for (long long i = 0; i < localN; i++) {
        lnum += (localData[i] - mean) * (local2[i] - localMean2);
        lden1 += (localData[i] - mean) * (localData[i] - mean);
        lden2 += (local2[i] - localMean2) * (local2[i] - localMean2);
    }

    double gnum, gden1, gden2;

    MPI_Reduce(&lnum, &gnum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&lden1, &gden1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&lden2, &gden2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double corr = 0;

    if (rank == 0)
        corr = gnum / std::sqrt(gden1 * gden2);

    double end = MPI_Wtime();

    if (rank == 0) {

        std::cout << "MPI Analytics\n";
        std::cout << "N = " << N << "\n\n";

        std::cout << "Mean : " << mean << "\n";
        std::cout << "Variance : " << var << "\n";
        std::cout << "StdDev : " << stddev << "\n";
        std::cout << "Min : " << globalMin << "\n";
        std::cout << "Max : " << globalMax << "\n\n";

        std::cout << "Histogram : ";
        for (int i = 0; i < bins; i++) std::cout << globalHist[i] << " ";
        std::cout << "\n";

        std::cout << "Sorted (first 10): ";
        for (int i = 0; i < 10; i++) std::cout << allData[i] << " ";
        std::cout << "\n";

        std::cout << "Moving Average (window=5, first 10): ";
        for (int i = 0; i < 10 && i < ma.size(); i++)
            std::cout << ma[i] << " ";
        std::cout << "\n";

        std::cout << "Correlation : " << corr << "\n";
        std::cout << "Outliers : " << globalOut << "\n";
        std::cout << "Time : " << end - start << " sec\n";
    }

    MPI_Finalize();
    return 0;
}