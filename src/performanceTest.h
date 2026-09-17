#pragma once

#include <string>
#include <vector>
#include <functional>

class PerformanceSuite {
public:
	static void runPerformanceTest(int numSamples);

private:
	static void runIndividualPerformanceTest(const std::string& testName,
        std::function<void(int, int*, int*)> functionToTest,
        std::function<float()> timerFunction,
        int numSamples,
        std::vector<int> sizes);

	static void getMeanStandardDeviation(std::vector<float> runtimes, float& outMean, float& outStandardDeviation);
};