#define XMA_IMPLEMENTATION

#include <XMA/Brain.hpp>

using namespace XMA;
using namespace XMA::Brain;

int main()
{
    // Neuron

    XMA_TEST("Neuron should be construct with 50 weights and changes", []() {
        MLP::Neuron neuron(50);
        XMA_ASSERT_EQUAL(neuron.weights.size(), 50);
        XMA_ASSERT_EQUAL(neuron.changes.size(), 50);
    });

    XMA_TEST("Neuron should have random bias", []() {
        float_t prevBias = { 0 };
        for(auto i(0u); i < 100; i++) {
            MLP::Neuron neuron(0);
            XMA_ASSERT_NOT_EQUAL(neuron.bias, 0);
            XMA_ASSERT_NOT_EQUAL(neuron.bias, prevBias);
            XMA_ASSERT_HIGHER_OR_EQUAL(neuron.bias, -1.f);
            XMA_ASSERT_LOWER_OR_EQUAL(neuron.bias, 1.f);
            prevBias = neuron.bias;
        }
    });

    XMA_TEST("Neuron should have random weights", []() {
        MLP::Neuron neuron(200);
        for(auto& weight : neuron.weights) {
            XMA_ASSERT_NOT_EQUAL(weight, 0);
            XMA_ASSERT_HIGHER_OR_EQUAL(weight, -1.f);
            XMA_ASSERT_LOWER_OR_EQUAL(weight, 1.f);
        }
    });

    XMA_TEST("Neuron output should throw bad input size", []() {
        MLP::Neuron neuron(2);
        XMA_ASSERT_THROW([&]() { neuron.output({ 1.f, 2.f, 3.f, 5.f }); });
    });

    XMA_TEST("Neuron should output good value", []() {
        MLP::Neuron neuron(2);
        neuron.bias = 0.f;
        neuron.weights[0] = 1.f;
        neuron.weights[1] = -1.f;
        XMA_ASSERT_EQUAL(neuron.output({ 1.f, 1.f }), 0.5f);
        XMA_ASSERT_HIGHER_OR_EQUAL((float)neuron.output({ 0.f, 1.f }), 0.2689f);
        XMA_ASSERT_LOWER_OR_EQUAL((float)neuron.output({ 0.f, 1.f }), 0.269f);
        XMA_ASSERT_HIGHER_OR_EQUAL((float)neuron.output({ 1.f, 0.f }), 0.731f);
        XMA_ASSERT_LOWER_OR_EQUAL((float)neuron.output({ 1.f, 0.f }), 0.732f);
    });

    // Network

    XMA_TEST("Network topology should be 2 3 1", []() {
        MLP::Network net({ 2, 3, 1 });
        XMA_ASSERT_EQUAL(net.getInputCount(), 2);
        XMA_ASSERT_EQUAL(net.getLayerCount(), 3);
        XMA_ASSERT_EQUAL(net.getOutputCount(), 1);
    });

    XMA_TEST("Network topology should be 10 51 23 8", []() {
        MLP::Network net({ 10, 51, 23, 8 });
        XMA_ASSERT_EQUAL(net.getInputCount(), 10);
        XMA_ASSERT_EQUAL(net.getOutputCount(), 8);
        XMA_ASSERT_EQUAL(net.getLayerCount(), 4);
    });

    XMA_TEST("Network output should throw bad input size", []() {
        MLP::Network net({ 2, 3, 3 });
        XMA_ASSERT_THROW([&]() { net.output({ 5, 6, 7 }); });
    });

    XMA_TEST("Network output should have 3 results", []() {
        MLP::Network net({ 2, 3, 5 });
        XMA_ASSERT_EQUAL(net.output({ 5, 6 }).size(), 5);
    });

    XMA_TEST("Network output should return Hard limit values", []() {
        MLP::Network net({ 3, 5, 20 });
        XMA::float_vector_t results = net.output({ 1.f, 0.f, 0.5f }, Transfers::hardLimit);
        XMA_ASSERT(results.size() == 20);
        XMA_ASSERT(results[0 ] == 0 || results[0 ] == 1);
        XMA_ASSERT(results[1 ] == 0 || results[1 ] == 1);
        XMA_ASSERT(results[2 ] == 0 || results[2 ] == 1);
        XMA_ASSERT(results[3 ] == 0 || results[3 ] == 1);
        XMA_ASSERT(results[4 ] == 0 || results[4 ] == 1);
        XMA_ASSERT(results[5 ] == 0 || results[5 ] == 1);
        XMA_ASSERT(results[6 ] == 0 || results[6 ] == 1);
        XMA_ASSERT(results[7 ] == 0 || results[7 ] == 1);
        XMA_ASSERT(results[8 ] == 0 || results[8 ] == 1);
        XMA_ASSERT(results[9 ] == 0 || results[9 ] == 1);
        XMA_ASSERT(results[10] == 0 || results[10] == 1);
        XMA_ASSERT(results[11] == 0 || results[11] == 1);
        XMA_ASSERT(results[12] == 0 || results[12] == 1);
        XMA_ASSERT(results[13] == 0 || results[13] == 1);
        XMA_ASSERT(results[14] == 0 || results[14] == 1);
        XMA_ASSERT(results[15] == 0 || results[15] == 1);
        XMA_ASSERT(results[16] == 0 || results[16] == 1);
        XMA_ASSERT(results[17] == 0 || results[17] == 1);
        XMA_ASSERT(results[18] == 0 || results[18] == 1);
        XMA_ASSERT(results[19] == 0 || results[19] == 1);
    });

    XMA_TEST("Network train for OR", []() {

        for(auto i(0u); i < 100; i++) {

            MLP::Network net({ 2, 1, 1 });

            MLP::training_t training;

            training.learningRate = 9.f;
            training.treshold = 0.01f;

            training.patterns.push_back({ { 0,0 }, { 0 } });
            training.patterns.push_back({ { 0,1 }, { 1 } });
            training.patterns.push_back({ { 1,0 }, { 1 } });
            training.patterns.push_back({ { 1,1 }, { 1 } });

            MLP::progress_t res = net.train(training);

            XMA_ASSERT_LOWER_OR_EQUAL(res.iterations, 150);

            for(auto& pattern : training.patterns) {
                XMA_ASSERT_EQUAL(net.output(pattern.inputs, Transfers::boolean), pattern.targets);
            }

        }

    });

    XMA_TEST("Network train for AND", []() {

        for(auto i(0u); i < 100; i++) {

            MLP::Network net({ 2, 1, 1 });

            MLP::training_t training;

            training.learningRate = 9.f;
            training.treshold = 0.01f;

            training.patterns.push_back({ { 0,0 }, { 0 } });
            training.patterns.push_back({ { 0,1 }, { 0 } });
            training.patterns.push_back({ { 1,0 }, { 0 } });
            training.patterns.push_back({ { 1,1 }, { 1 } });

            MLP::progress_t res = net.train(training);

            XMA_ASSERT_LOWER_OR_EQUAL(res.iterations, 150);

            for(auto& pattern : training.patterns) {
                XMA_ASSERT_EQUAL(net.output(pattern.inputs, Transfers::boolean), pattern.targets);
            }

        }

    });

    XMA_TEST("Network train for XOR", []() {

        for(auto i(0u); i < 100; i++) {

            MLP::Network net({ 2, 5, 1 });

            MLP::training_t training;

            training.learningRate = 6.f;
            training.treshold = 0.001f;

            training.patterns.push_back({ { 0,0 }, { 0 } });
            training.patterns.push_back({ { 1,1 }, { 0 } });
            training.patterns.push_back({ { 1,0 }, { 1 } });
            training.patterns.push_back({ { 0,1 }, { 1 } });

            MLP::progress_t res = net.train(training);

            for(auto& pattern : training.patterns) {

                float_vector_t prediction = net.output(pattern.inputs, Transfers::logistic);

                if(pattern.targets[0] == 1) {
                    XMA_ASSERT_HIGHER(prediction[0], 0.9f);
                } else {
                    XMA_ASSERT_LOWER(prediction[0], 0.1f);
                }
            }

        }

    });
}