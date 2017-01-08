#include <XMA/Brain/MLP/Network.hpp>

namespace XMA { namespace Brain { namespace MLP {

// ---------------------------------------------------------------------------------------------------------------------

Network::Network(const float_vector_t& topology) : m_topology(topology)
{
    m_layers.push_back(Layer(topology[0], 1));

    for(auto i(1u); i < topology.size(); i++) {
        m_layers.push_back(Layer(topology[i], topology[i-1]));
    }
}

// ---------------------------------------------------------------------------------------------------------------------

count_t Network::getInputCount()
{
    return m_topology[0];
}

// ---------------------------------------------------------------------------------------------------------------------

count_t Network::getLayerCount()
{
    return m_topology.size();
}

// ---------------------------------------------------------------------------------------------------------------------

count_t Network::getOutputCount()
{
    return m_topology[m_topology.size()-1];
}

// ---------------------------------------------------------------------------------------------------------------------

float_vector_t Network::output(const float_vector_t& inputs, transfer_t transfer)
{
    XMA_ASSERT(inputs.size() == getInputCount());

    float_vector_t results;
    float_vector_t prevInputs = inputs;

    for(auto i(1u); i < getLayerCount(); i++) {
        results = prevInputs = m_layers[i].output(prevInputs, transfer);
    }

    return results;
}

// ---------------------------------------------------------------------------------------------------------------------

progress_t Network::train(const training_t& training, std::function<void(const progress_t&)> callback, count_t interval)
{
    progress_t result;

    result.error = training.treshold + 0.1f;

    for (auto i(0u); i < training.iterations && result.error > training.treshold; i++) {

        float_t iterationError { 0 };

        for(auto& pattern : training.patterns)
            iterationError += trainPattern(pattern, training.learningRate, training.momentum);

        result.error = iterationError / training.patterns.size();

        if(callback != nullptr && (i % interval == 0)) {
            callback(result);
        }

        result.iterations++;
    }

    return result;
}

// ---------------------------------------------------------------------------------------------------------------------

float_t Network::trainPattern(const training_pattern_t& pattern, float_t learningRate, float_t momentum)
{
    std::vector<float_vector_t> outputs = getOutputs(pattern.inputs);

    // Calculate errors and deltas

    size_t lastLayerId = m_layers.size()-1;

    for (int layerId = lastLayerId; layerId >= 0; layerId--) {

        for (auto neuronId(0u); neuronId < m_layers[layerId].neurons.size(); neuronId++) {

            float_t output { outputs[layerId][neuronId] };
            float_t neuronError { 0 };

            if (layerId == lastLayerId) {
                neuronError = pattern.targets[neuronId] - output;
            } else {
                for(auto& n : m_layers[layerId + 1].neurons)
                    neuronError += n.delta * n.weights[neuronId];
            }

            m_layers[layerId].neurons[neuronId].error = neuronError;
            m_layers[layerId].neurons[neuronId].delta = neuronError * output * (1 - output);
        }
    }

    // Back propagate

    for (auto layerId(1u); layerId < m_layers.size(); layerId++) {

        float_vector_t incoming = outputs[layerId-1];

        for (auto neuronId(0u); neuronId < m_layers[layerId].neurons.size(); neuronId++) {

            Neuron& neuron = m_layers[layerId].neurons[neuronId];

            for (auto k(0u); k < incoming.size(); k++) {
                float_t& change = neuron.changes[k];
                change = (learningRate * neuron.delta * incoming[k]) + (momentum * change);
                neuron.weights[k] += change;
            }

            neuron.bias += learningRate * neuron.delta;
        }
    }

    float_vector_t errors;

    for(auto& n : m_layers[lastLayerId].neurons) errors.push_back(n.error);

    return getErrorSum(errors);
}

// ---------------------------------------------------------------------------------------------------------------------

std::vector<float_vector_t> Network::getOutputs(const float_vector_t& inputs)
{
    std::vector<float_vector_t> results;

    float_vector_t outputs = inputs;

    results.push_back(outputs);

    for (auto i(1u); i < m_layers.size(); i++) {
        outputs = m_layers[i].output(outputs);
        results.push_back(outputs);
    }

    return results;
}

// ---------------------------------------------------------------------------------------------------------------------

float_t Network::getErrorSum(const float_vector_t& errors)
{
    float_t sum = 0;
    for (auto& error : errors) {
        sum += pow(error, 2);
    }
    return sum / errors.size();
}

// ---------------------------------------------------------------------------------------------------------------------

}}}