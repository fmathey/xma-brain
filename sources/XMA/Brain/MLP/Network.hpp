#ifndef _XMA_BRAIN_MLP_NETWORK_HPP
#define _XMA_BRAIN_MLP_NETWORK_HPP

#include <XMA/Brain/MLP/Layer.hpp>

namespace XMA { namespace Brain { namespace MLP {

// ---------------------------------------------------------------------------------------------------------------------

typedef struct {
    float_vector_t inputs;
    float_vector_t targets;

} training_pattern_t;

// ---------------------------------------------------------------------------------------------------------------------

using training_pattern_vector_t = std::vector<training_pattern_t>;

// ---------------------------------------------------------------------------------------------------------------------

typedef struct {
    count_t iterations { 20000 };
    float_t learningRate { 0.3f };
    float_t momentum { 0.1f };
    float_t treshold { 0.005f };
    training_pattern_vector_t patterns;

} training_t;

// ---------------------------------------------------------------------------------------------------------------------

typedef struct
{
    count_t iterations { 0 };
    float_t error { 0 };

} progress_t;

// ---------------------------------------------------------------------------------------------------------------------

class Network
{
    public:

        Network(const float_vector_t& topology);

        count_t getInputCount();
        count_t getOutputCount();
        count_t getLayerCount();

        float_vector_t output(const float_vector_t& inputs, transfer_t transfer = Transfers::logistic);

        progress_t train(
                const training_t& training,
                std::function<void(const progress_t&)> = nullptr,
                count_t interval = 1
        );

    private:

        float_t trainPattern(const training_pattern_t& pattern, float_t learningRate, float_t momentum);
        std::vector<float_vector_t> getOutputs(const float_vector_t& inputs);
        float_t getErrorSum(const float_vector_t& errors);

    private:

        std::vector<Layer> m_layers;

        float_vector_t m_topology;
};

// ---------------------------------------------------------------------------------------------------------------------

}}}

// ---------------------------------------------------------------------------------------------------------------------

#ifdef XMA_IMPLEMENTATION
    #include "Network.cxx"
#endif

// ---------------------------------------------------------------------------------------------------------------------

#endif
