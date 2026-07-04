
//This code was developed by our team.
// We used limited AI assistance (for syntax correction and optimization suggestions).
// All logic and implementation decisions were made and verified by team members.


#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <string>

using namespace std;


void runAnalysis(const string& label, long long N, int bins, ofstream& log)
{
    cout << "\n#####################################\n";
    cout << "Run: " << label << "  (N = " << N << ")\n";
    cout << "#####################################\n";

    cout << "Generating synthetic dataset...\n";

    // Generate synthetic dataset (Column A)
    mt19937_64 rng(42);
    uniform_real_distribution<double> dist(0.0, 10000.0);

    vector<double> data(N);
    for (long long i = 0; i < N; i++)
        data[i] = dist(rng);

    // Generate a second synthetic column for correlation (Column B)
    mt19937_64 rng2(123);
    normal_distribution<double> noise(0.0, 500.0);

    vector<double> data2(N);
    for (long long i = 0; i < N; i++)
        data2[i] = data[i] * 0.5 + noise(rng2);

    double totalElapsed = 0.0;

    // =====================================================
    // Task 1 : Basic Statistics
    // =====================================================
    auto start = chrono::high_resolution_clock::now();

    double sum = accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / N;

    double sq_sum = inner_product(
        data.begin(), data.end(), data.begin(), 0.0);

    double variance = sq_sum / N - mean * mean;
    double stddev = sqrt(variance);

    double minVal = *min_element(data.begin(), data.end());
    double maxVal = *max_element(data.begin(), data.end());

    auto end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double, milli>(end - start).count();
    totalElapsed += elapsed;

    cout << "\n===== Task 1: Basic Statistics =====\n";
    cout << "Mean      : " << mean << endl;
    cout << "Variance  : " << variance << endl;
    cout << "Std Dev   : " << stddev << endl;
    cout << "Minimum   : " << minVal << endl;
    cout << "Maximum   : " << maxVal << endl;
    cout << "Time      : " << elapsed << " ms" << endl;

    log << label << ",BasicStatistics," << N << "," << mean << "," << elapsed << "\n";

    // =====================================================
    // Task 2 : Histogram Generation (configurable bin count)
    // =====================================================
    start = chrono::high_resolution_clock::now();

    vector<long long> histogram(bins, 0);
    double range = maxVal - minVal;
    double binWidth = range / bins;

    for (double value : data)
    {
        int bin = (int)((value - minVal) / binWidth);
        if (bin >= bins) bin = bins - 1;
        if (bin < 0) bin = 0;
        histogram[bin]++;
    }

    end = chrono::high_resolution_clock::now();
    elapsed = chrono::duration<double, milli>(end - start).count();
    totalElapsed += elapsed;

    cout << "\n===== Task 2: Histogram (" << bins << " bins) =====\n";
    for (int i = 0; i < bins; i++)
        cout << "  Bin " << i << " [" << (minVal + i * binWidth) << " - "
             << (minVal + (i + 1) * binWidth) << "): " << histogram[i] << "\n";
    cout << "Time      : " << elapsed << " ms" << endl;

    log << label << ",Histogram," << N << "," << bins << " bins," << elapsed << "\n";

    // =====================================================
    // Task 3 : Sorting
    // (Sequential baseline. The parallel version implements
    //  a distributed sort, e.g. Sample Sort / Bitonic Sort,
    //  across MPI ranks.)
    // =====================================================
    start = chrono::high_resolution_clock::now();

    vector<double> sorted = data;
    sort(sorted.begin(), sorted.end());

    end = chrono::high_resolution_clock::now();
    elapsed = chrono::duration<double, milli>(end - start).count();
    totalElapsed += elapsed;

    cout << "\n===== Task 3: Sorting =====\n";
    cout << "Sorting completed.\n";
    cout << "Time      : " << elapsed << " ms" << endl;

    log << label << ",Sorting," << N << ",Completed," << elapsed << "\n";

    // =====================================================
    // Task 4 : Pearson Correlation (between data and data2)
    // =====================================================
    start = chrono::high_resolution_clock::now();

    double mean2 = accumulate(data2.begin(), data2.end(), 0.0) / N;

    double numerator = 0.0;
    double denominator1 = 0.0;
    double denominator2 = 0.0;

    for (long long i = 0; i < N; i++)
    {
        double da = data[i] - mean;
        double db = data2[i] - mean2;
        numerator += da * db;
        denominator1 += da * da;
        denominator2 += db * db;
    }

    double correlation = numerator / sqrt(denominator1 * denominator2);

    end = chrono::high_resolution_clock::now();
    elapsed = chrono::duration<double, milli>(end - start).count();
    totalElapsed += elapsed;

    cout << "\n===== Task 4: Pearson Correlation =====\n";
    cout << "Correlation : " << correlation << endl;
    cout << "Time        : " << elapsed << " ms" << endl;

    log << label << ",Correlation," << N << "," << correlation << "," << elapsed << "\n";

    // =====================================================
    // Task 5 : Moving Average (rolling window)
    // =====================================================
    start = chrono::high_resolution_clock::now();

    int window = 5;
    vector<double> movingAverage(N - window + 1);
    double windowSum = accumulate(data.begin(), data.begin() + window, 0.0);
    movingAverage[0] = windowSum / window;

    for (long long i = 1; i <= N - window; i++)
    {
        windowSum += data[i + window - 1] - data[i - 1];
        movingAverage[i] = windowSum / window;
    }

    end = chrono::high_resolution_clock::now();
    elapsed = chrono::duration<double, milli>(end - start).count();
    totalElapsed += elapsed;

    cout << "\n===== Task 5: Moving Average (window=" << window << ") =====\n";
    cout << "Moving Average completed.\n";
    cout << "Time      : " << elapsed << " ms" << endl;

    log << label << ",MovingAverage," << N << ",Window=" << window << "," << elapsed << "\n";

    // =====================================================
    // Task 6 : Outlier Detection (Z-score method)
    // =====================================================
    start = chrono::high_resolution_clock::now();

    long long outliers = 0;
    for (double value : data)
    {
        double z = (value - mean) / stddev;
        if (fabs(z) > 3.0)
            outliers++;
    }

    end = chrono::high_resolution_clock::now();
    elapsed = chrono::duration<double, milli>(end - start).count();
    totalElapsed += elapsed;

    cout << "\n===== Task 6: Outlier Detection (Z-score) =====\n";
    cout << "Outliers  : " << outliers << endl;
    cout << "Time      : " << elapsed << " ms" << endl;

    log << label << ",OutlierDetection," << N << "," << outliers << "," << elapsed << "\n";

    cout << "\n-------------------------------------\n";
    cout << label << " Total Time: " << totalElapsed << " ms ("
         << totalElapsed / 1000.0 << " s)\n";
    cout << "-------------------------------------\n";
}

int main()
{
    cout << "=====================================\n";
    cout << "Sequential Data Analytics\n";
    cout << "=====================================\n";

    // -----------------------------------------------------
    // Hardcoded run sizes — no input needed.
    // Small  : 1,000,000    (expected < 5 s)
    // Medium : 10,000,000   (expected 10-60 s)
    // Large  : 100,000,000  (expected 1-10 min)
    // -----------------------------------------------------
    const int bins = 10;

    filesystem::create_directory("results");
    ofstream log("results/sequential.csv");
    log << "RunSize,Task,DatasetSize,Result,ExecutionTime(ms)\n";

    runAnalysis("Small",  1000000,   bins, log);
    runAnalysis("Medium", 10000000,  bins, log);
    runAnalysis("Large",  100000000, bins, log);

    log.close();

    cout << "\n=====================================\n";
    cout << "All runs completed.\n";
    cout << "Results saved to results/sequential.csv\n";
    cout << "=====================================\n";

    return 0;
}