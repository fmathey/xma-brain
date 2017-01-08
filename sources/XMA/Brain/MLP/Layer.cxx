#include <XMA/Brain/MLP/Layer.hpp>

namespace XMA { namespace Brain { namespace MLP {

// ---------------------------------------------------------------------------------------------------------------------

Layer::Layer(count_t numberOfNeurons, count_t sizeOfNeurons)
{
    for(auto i(0u); i < numberOfNeurons; i++) {
        neurons.push_back(Neuron(sizeOfNeurons));
    }
}

// ---------------------------------------------------------------------------------------------------------------------

float_vector_t Layer::output(const float_vector_t& inputs, transfer_t transfer)
{
    float_vector_t results;

    for(auto& neuron : neurons) {
        results.push_back(neuron.output(inputs, transfer));
    }

    return results;
}

// ---------------------------------------------------------------------------------------------------------------------

}}}