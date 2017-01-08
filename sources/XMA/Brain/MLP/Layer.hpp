#ifndef _XMA_BRAIN_MLP_LAYER_HPP
#define _XMA_BRAIN_MLP_LAYER_HPP

#include <XMA/Brain/MLP/Neuron.hpp>

namespace XMA { namespace Brain { namespace MLP {

// ---------------------------------------------------------------------------------------------------------------------

struct Layer
{
    std::vector<Neuron> neurons;

    Layer(count_t numberOfNeurons, count_t sizeOfNeurons);

    float_vector_t output(const float_vector_t& inputs, transfer_t transfer = Transfers::logistic);
};

// ---------------------------------------------------------------------------------------------------------------------

}}}

// ---------------------------------------------------------------------------------------------------------------------

#ifdef XMA_IMPLEMENTATION
    #include "Layer.cxx"
#endif

// ---------------------------------------------------------------------------------------------------------------------

#endif
