module neuron #(
    parameter int CURRENT_WIDTH = 8,           // Width of the current input
    parameter int POTENTIAL_WIDTH = 8,        // Width of membrane potential
    parameter int LEAK_FACTOR_WIDTH = 8,       // Width of the leak factor
    parameter int THRESHOLD_WIDTH = 8,        // Width of the threshold parameter
    parameter int REFRACTORY_COUNTER_WIDTH = 4 // Width of refractory counter (determines max refractory period)
) (
    input logic clk,                                  // Clock signal
    input logic rst_n,                                // Active low reset
    input logic [CURRENT_WIDTH-1:0] current_in,       // Input current
    output logic spike_out                            // Output spike
);

    // Internal registers - all unsigned
    logic [POTENTIAL_WIDTH-1:0] membrane_potential;  // Vm[t]
    logic [REFRACTORY_COUNTER_WIDTH-1:0] refractory_counter;
    logic in_refractory_period;
    logic [LEAK_FACTOR_WIDTH-1:0] leak_value;        // Internal constant leak value (Ileak)
    logic [THRESHOLD_WIDTH-1:0] threshold;           // Firing threshold (Vth)
    logic spike_state;                              // S[t]

    // Initialize constants (these could be made inputs if desired)
    initial begin
        leak_value = 8'd1;  // Default leak value
        threshold = 8'd100; // Default threshold value
    end

    // Determine if in refractory period
    assign in_refractory_period = (refractory_counter > 0);
    
    // Assign spike_state to indicate if the neuron is currently spiking
    assign spike_state = (membrane_potential > threshold);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset state
            membrane_potential <= '0;
            refractory_counter <= '0;
            spike_out <= 1'b0;
        end else begin
            // Set output spike based on spike state
            spike_out <= spike_state;
            
            // Handle refractory period counter
            if (refractory_counter > 0) begin
                refractory_counter <= refractory_counter - 1'b1;
            end

            // Update membrane potential based on the formula:
            // Vm[t] = { max(Iin[t] + Vm[t-1] - Ileak, 0), S[t] = 0
            //         { 0,                              S[t] = 1
            // S[t] = Vm[t] > Vth[t]
            
            if (spike_state) begin
                // If spiking, reset membrane potential and enter refractory period
                membrane_potential <= '0;
                refractory_counter <= 4'd5; // Default refractory period of 5 cycles
            end else if (!in_refractory_period) begin
                // If not spiking and not in refractory period, update potential
                
                // Handle membrane potential update with careful handling of underflow
                if (membrane_potential + current_in < leak_value) begin
                    // If potential would go below 0, clamp to 0
                    membrane_potential <= '0;
                end else begin
                    // Normal case: current + old_potential - leak
                    membrane_potential <= membrane_potential + current_in - leak_value;
                end
                
                // Check for potential overflow and clamp if necessary
                // This handles the case where current_in + membrane_potential would overflow
                if (current_in > {POTENTIAL_WIDTH{1'b1}} - membrane_potential) begin
                    // If addition would overflow, clamp to max value
                    membrane_potential <= {POTENTIAL_WIDTH{1'b1}};
                end
            end
        end
    end

endmodule