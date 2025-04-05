#include <iostream>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vneuron.h"
#include <vector>
#include <cstdlib>
#include <ctime>

#define MAX_SIM_TIME 1000
vluint64_t sim_time = 0;

int main(int argc, char** argv) {
    // Initialize Verilator
    Verilated::commandArgs(argc, argv);
    
    // Create an instance of our module under test
    Vneuron *dut = new Vneuron;
    
    // Trace initialization
    Verilated::traceEverOn(true);
    VerilatedVcdC* trace = new VerilatedVcdC;
    dut->trace(trace, 99);
    trace->open("neuron.vcd");
    
    // Initialize random seed
    std::srand(std::time(nullptr));
    
    // Simulation variables
    std::vector<uint8_t> stimulus = {
        0, 0, 0, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,   // Hold at 20 to accumulate
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30,                        // Stronger input
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5     // Weaker input
    };
    
    size_t stim_idx = 0;
    bool last_spike = false;
    
    // Reset sequence
    dut->rst_n = 0;
    dut->clk = 0;
    dut->current_in = 0;
    
    // Toggle clock for a few cycles during reset
    for (int i = 0; i < 5; i++) {
        dut->clk = !dut->clk;
        dut->eval();
        trace->dump(sim_time++);
    }
    
    // Release reset
    dut->rst_n = 1;
    
    // Start actual simulation
    std::cout << "Time,Current,Spike\n";
    
    while (sim_time < MAX_SIM_TIME && stim_idx < stimulus.size()) {
        // Apply stimulus when clock is low
        if (dut->clk == 0) {
            dut->current_in = stimulus[stim_idx];
        }
        
        // Toggle clock
        dut->clk = !dut->clk;
        
        // Evaluate model
        dut->eval();
        
        // Dump trace and advance time
        trace->dump(sim_time);
        sim_time++;
        
        // Sample and print outputs on positive edge
        if (dut->clk == 1) {
            std::cout << sim_time << "," << (int)dut->current_in << "," << (int)dut->spike_out << std::endl;
            
            // Detect rising edge of spike (useful for counting spikes)
            if (dut->spike_out == 1 && last_spike == false) {
                std::cout << "SPIKE DETECTED at time " << sim_time << std::endl;
            }
            last_spike = dut->spike_out;
            
            // Increment stimulus index on positive edge
            if (dut->clk == 1) {
                stim_idx = (stim_idx + 1 < stimulus.size()) ? stim_idx + 1 : stim_idx;
            }
        }
    }
    
    // Clean up
    trace->close();
    delete trace;
    delete dut;
    
    std::cout << "Simulation completed after " << sim_time << " ticks" << std::endl;
    return 0;
}