#ifndef _XMA_BRAIN_MLP_NEURON_HPP
#define _XMA_BRAIN_MLP_NEURON_HPP

#include <XMA/Brain/Config.hpp>
#include <XMA/Brain/Transfers.hpp>

namespace XMA { namespace Brain { namespace MLP {

// ---------------------------------------------------------------------------------------------------------------------

struct Neuron
{
    float_t bias { 0 };
    float_t error { 0 };
    float_t delta { 0 };
    float_vector_t weights;
    float_vector_t changes;

    Neuron(count_t size);

    float_t output(const float_vector_t& inputs, transfer_t transfer = Transfers::logistic);
};

// ---------------------------------------------------------------------------------------------------------------------

}}}

// ---------------------------------------------------------------------------------------------------------------------

#ifdef XMA_IMPLEMENTATION
    #include "Neuron.cxx"
#endif

// ---------------------------------------------------------------------------------------------------------------------

#endif
