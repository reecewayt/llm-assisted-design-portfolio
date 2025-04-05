module network #(
    parameter int LAYER1_NEURONS = 3,          // Number of neurons in layer 1
    parameter int LAYER2_NEURONS = 3,          // Number of neurons in layer 2
    parameter int CURRENT_WIDTH = 8,           // Width of current signals
    parameter int WEIGHT_WIDTH = 8,            // Width of synaptic weights
    parameter int POTENTIAL_WIDTH = 8,         // Width of membrane potential
    parameter int LEAK_FACTOR_WIDTH = 8,       // Width of the leak factor
    parameter int THRESHOLD_WIDTH = 8,         // Width of the threshold parameter
    parameter int REFRACTORY_COUNTER_WIDTH = 4 // Width of refractory counter
) (
    input logic clk,                                                        // Clock signal
    input logic rst_n,                                                      // Active low reset
    input logic [CURRENT_WIDTH-1:0] inputs[LAYER1_NEURONS],                 // Input currents to first layer
    input logic [WEIGHT_WIDTH-1:0] weights[LAYER2_NEURONS][LAYER1_NEURONS], // Programmable weights
    output logic spikes_layer2[LAYER2_NEURONS]                              // Spikes from layer 2
);

    logic spikes_layer1[LAYER1_NEURONS];                                    // Spikes from layer 1 neurons
    logic [CURRENT_WIDTH-1:0] currents_layer2[LAYER2_NEURONS];              // Currents to layer 2 neurons



    // Instantiate layer 1 neurons
    generate
        for (genvar i = 0; i < LAYER1_NEURONS; i++) begin : layer1_neurons
            neuron #(
                .CURRENT_WIDTH(CURRENT_WIDTH),
                .POTENTIAL_WIDTH(POTENTIAL_WIDTH),
                .LEAK_FACTOR_WIDTH(LEAK_FACTOR_WIDTH),
                .THRESHOLD_WIDTH(THRESHOLD_WIDTH),
                .REFRACTORY_COUNTER_WIDTH(REFRACTORY_COUNTER_WIDTH)
            ) neuron_inst (
                .clk(clk),
                .rst_n(rst_n),
                .current_in(inputs[i]),
                .spike_out(spikes_layer1[i])
            );
        end
    endgenerate

    // Instantiate multi-input synapses for layer 2
    generate
        for (genvar j = 0; j < LAYER2_NEURONS; j++) begin : layer2_synapses
            multi_input_synapse #(
                .NUM_INPUTS(LAYER1_NEURONS),
                .WEIGHT_WIDTH(WEIGHT_WIDTH),
                .CURRENT_WIDTH(CURRENT_WIDTH)
            ) synapse_inst (
                .spikes_in(spikes_layer1),            // All layer 1 spikes
                .weights(weights[j]),         // Weights for this specific layer 2 neuron
                .current_out(currents_layer2[j])      // Current output to layer 2 neuron
            );
        end
    endgenerate

    // Instantiate layer 2 neurons
    generate
        for (genvar j = 0; j < LAYER2_NEURONS; j++) begin : layer2_neurons
            neuron #(
                .CURRENT_WIDTH(CURRENT_WIDTH),
                .POTENTIAL_WIDTH(POTENTIAL_WIDTH),
                .LEAK_FACTOR_WIDTH(LEAK_FACTOR_WIDTH),
                .THRESHOLD_WIDTH(THRESHOLD_WIDTH),
                .REFRACTORY_COUNTER_WIDTH(REFRACTORY_COUNTER_WIDTH)
            ) neuron_inst (
                .clk(clk),
                .rst_n(rst_n),
                .current_in(currents_layer2[j]),
                .spike_out(spikes_layer2[j])
            );
        end
    endgenerate

endmodule
