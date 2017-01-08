#ifndef _XMA_BRAIN_TRANSFERS_HPP
#define _XMA_BRAIN_TRANSFERS_HPP

#include <XMA/Core.hpp>

namespace XMA { namespace Brain { namespace Transfers {

// ---------------------------------------------------------------------------------------------------------------------

static float_t identity(float_t sum) { return sum; }

static float_t logistic(float_t sum) { return 1.f / (1.f + exp(-sum)); }

static float_t hardLimit(float_t sum) { return sum > 0.f ? 1.f : 0.f; }

static float_t relu(float_t sum) { return sum > 0.f ? sum : 0.f; }

static float_t boolean(float_t sum) { return sum >= 0.5f ? 1.f : 0.f; }

static float_t tanh(float_t sum) {
    float_t eP = exp(sum);
    float_t eN = 1 / eP;
    return (eP - eN) / (eP + eN);
}

// ---------------------------------------------------------------------------------------------------------------------

}}}

#endif
