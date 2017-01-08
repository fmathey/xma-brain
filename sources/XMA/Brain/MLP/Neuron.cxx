#include <XMA/Brain/MLP/Neuron.hpp>

namespace XMA { namespace Brain { namespace MLP {

// ---------------------------------------------------------------------------------------------------------------------


Neuron::Neuron(count_t size)
{
    bias = Core::random(-1.f, 1.f);

    for(auto i(0u); i < size; i++) {
        weights.push_back(Core::random(-1.f, 1.f));
        changes.push_back(0);
    }
}

float_t Neuron::output(const float_vector_t& inputs, transfer_t transfer)
{
    XMA_ASSERT(inputs.size() == weights.size());

    float_t result = bias;

    for(auto i(0u); i < inputs.size(); i++) {
        result += weights[i] * inputs[i];
    }

    return transfer(result);
}

// ---------------------------------------------------------------------------------------------------------------------

}}}