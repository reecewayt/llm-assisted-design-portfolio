#include <iostream>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vnetwork.h"
#include <array>
#include <vector>
#include <cstdlib>
#include <ctime>

#define MAX_SIM_TIME 1000
#define LAYER1_NEURONS 3
#define LAYER2_NEURONS 3
#define CURRENT_WIDTH 8
#define WEIGHT_WIDTH 8

vluint64_t sim_time = 0;

int main(int argc, char** argv) {
    // Initialize Verilator
    Verilated::commandArgs(argc, argv);

    // Create an instance of our module under test
    Vnetwork *dut = new Vnetwork;

    // Trace initialization
    Verilated::traceEverOn(true);
    VerilatedVcdC* trace = new VerilatedVcdC;
    dut->trace(trace, 99);
    trace->open("network.vcd");

    // Initialize random seed
    std::srand(std::time(nullptr));

    // Simulation variables
    std::vector<std::array<uint8_t, LAYER1_NEURONS>> input_patterns = {
        {30, 0, 0},     // Stimulate neuron 0
        {0, 30, 0},     // Stimulate neuron 1
        {0, 0, 30},     // Stimulate neuron 2
        {20, 20, 20},   // Stimulate all neurons
        {10, 20, 30}    // Gradient stimulus
    };

    size_t pattern_idx = 0;
    size_t pattern_duration = 200;
    size_t pattern_start_time = 0;

    // Reset sequence
    dut->rst_n = 0;
    dut->clk = 0;

    // Configure synapse weights
    // First neuron in layer 2 strongly connected to neuron 0 in layer 1
    dut->weights[0][0] = 50;  // Strong connection
    dut->weights[0][1] = 10;  // Weak connection
    dut->weights[0][2] = 10;  // Weak connection

    // Second neuron in layer 2 strongly connected to neuron 1 in layer 1
    dut->weights[1][0] = 50;  // Weak connection
    dut->weights[1][1] = 50;  // Strong connection
    dut->weights[1][2] = 10;  // Weak connection

    // Third neuron in layer 2 strongly connected to neuron 2 in layer 1
    dut->weights[2][0] = 10;  // Weak connection
    dut->weights[2][1] = 10;  // Weak connection
    dut->weights[2][2] = 50;  // Strong connection

    // Apply initial inputs
    for (int i = 0; i < LAYER1_NEURONS; i++) {
        dut->inputs[i] = 0;
    }

    // Toggle clock for a few cycles during reset
    for (int i = 0; i < 5; i++) {
        dut->clk = !dut->clk;
        dut->eval();
        trace->dump(sim_time++);
    }

    // Release reset
    dut->rst_n = 1;

    // Start actual simulation
    std::cout << "Starting neural network simulation with " << LAYER1_NEURONS
              << " input neurons and " << LAYER2_NEURONS << " output neurons" << std::endl;

    while (sim_time < MAX_SIM_TIME) {
        // Apply stimulus when clock is low
        if (dut->clk == 0) {
            // Check if it's time to switch patterns
            if (sim_time - pattern_start_time >= pattern_duration) {
                pattern_idx = (pattern_idx + 1) % input_patterns.size();
                pattern_start_time = sim_time;

                std::cout << "\nSwitching to input pattern " << pattern_idx << " at time " << sim_time << std::endl;
                std::cout << "Pattern: [";
                for (int i = 0; i < LAYER1_NEURONS; i++) {
                    std::cout << (int)input_patterns[pattern_idx][i];
                    if (i < LAYER1_NEURONS - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }

            // Apply current pattern
            for (int i = 0; i < LAYER1_NEURONS; i++) {
                dut->inputs[i] = input_patterns[pattern_idx][i];
            }
        }

        // Toggle clock
        dut->clk = !dut->clk;

        // Evaluate model
        dut->eval();

        // Dump trace and advance time
        trace->dump(sim_time);

        // Sample and print outputs on positive edge
        if (dut->clk == 1) {
            // Display any spikes for monitoring
            bool any_spike = false;
            std::string spike_str = "Time=" + std::to_string(sim_time) + " Spikes: ";

            // Only check layer 2 spikes since layer 1 is no longer accessible
            for (int i = 0; i < LAYER2_NEURONS; i++) {
                if (dut->spikes_layer2[i]) {
                    spike_str += "L2N" + std::to_string(i) + " ";
                    any_spike = true;
                }
            }

            if (any_spike) {
                std::cout << spike_str << std::endl;
            }
        }

        sim_time++;
    }

    // Clean up
    trace->close();
    delete trace;
    delete dut;

    std::cout << "Simulation completed after " << sim_time << " ticks" << std::endl;
    std::cout << "Waveform data saved to network.vcd" << std::endl;

    return 0;
}
