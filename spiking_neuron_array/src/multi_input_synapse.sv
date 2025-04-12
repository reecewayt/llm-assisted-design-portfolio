module multi_input_synapse #(
    parameter int NUM_INPUTS = 3,           // Number of input spikes
    parameter int WEIGHT_WIDTH = 8,         // Width of the weights
    parameter int CURRENT_WIDTH = 8         // Width of output current
) (
    input logic spikes_in[NUM_INPUTS],                       // Input spikes array
    input logic [WEIGHT_WIDTH-1:0] weights[NUM_INPUTS],      // Programmable 8 bit weights array
    output logic [CURRENT_WIDTH-1:0] current_out            // Output current (sum of weighted spikes)
);

    assign current_out = spikes_in[0] * weights[0] + spikes_in[1] * weights[1] + spikes_in[2] * weights[2];


endmodule
