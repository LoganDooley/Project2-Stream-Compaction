#pragma once

#include <string>
#include <vector>
#include <functional>

class PerformanceSuite {
public:
	static void runPerformanceTest(int numSamples, int SIZE, int NPOT, int* a, int* b, int* c);

private:
	static void runIndividualPerformanceTest(const std::string& testName,
		std::function<void(int, int*, int*)> functionToTest,
		std::function<float()> timerFunction,
		int numSamples, int SIZE, int NPOT, int* a, int* b, int* c);

	static void getMeanStandardDeviation(std::vector<float> runtimes, float& outMean, float& outStandardDeviation);
};