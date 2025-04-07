#include <iostream>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vspi_interface.h"
#include <cstdint>
#include <string>

#define MAX_SIM_TIME 1000
vluint64_t sim_time = 0;

// Helper class for SPI testing
class SPITester {
private:
    Vspi_interface *dut;
    VerilatedVcdC* trace;

public:
    SPITester(Vspi_interface *dut_ptr, VerilatedVcdC* trace_ptr)
        : dut(dut_ptr), trace(trace_ptr) {}

    // Function to toggle SCK once (low->high->low or high->low->high)
    void toggle_sck() {
        dut->spi_sck = !dut->spi_sck;
        dut->eval();
        trace->dump(sim_time++);
    }

    // Function to delay for n simulation ticks
    void delay(int n) {
        for (int i = 0; i < n; i++) {
            dut->eval();
            trace->dump(sim_time++);
        }
    }

    // Function to assert or deassert CS
    void set_cs(bool active) {
        // CS is active low
        dut->spi_cs_n = !active;
        dut->eval();
        trace->dump(sim_time++);
    }

    // Function to send one byte over MOSI and return received MISO
    uint8_t transfer_byte(uint8_t send_byte) {
        uint8_t receive_byte = 0;

        // Transfer 8 bits, MSB first
        for (int i = 7; i >= 0; i--) {
            // Set up MOSI with bit to send
            dut->spi_mosi = (send_byte >> i) & 0x01;
            dut->eval();

            // Clock low
            dut->spi_sck = 0;
            dut->eval();
            trace->dump(sim_time++);

            // Clock high - SPI module samples MOSI here
            dut->spi_sck = 1;
            dut->eval();
            trace->dump(sim_time++);

            // Sample MISO (capture the bit)
            receive_byte = (receive_byte << 1) | dut->spi_miso;
        }

        return receive_byte;
    }

    // Function to prepare transmit data in the SPI module
    void set_transmit_data(uint8_t tx_byte) {
        dut->tx_data = tx_byte;
        dut->tx_ready = 1;
        dut->eval();
        trace->dump(sim_time++);

        // Pulse tx_ready
        delay(1);
        dut->tx_ready = 0;
        dut->eval();
        trace->dump(sim_time++);
    }

    // Function to run a complete SPI transaction with a single byte
    void run_transaction(uint8_t mosi_byte, uint8_t expected_miso_byte, const std::string& test_name) {
        std::cout << "\nTest: " << test_name << std::endl;

        // Setup transmit data before asserting CS
        set_transmit_data(expected_miso_byte);

        // Start transaction
        set_cs(true);

        // Transfer byte
        uint8_t received_miso = transfer_byte(mosi_byte);

        // End transaction
        set_cs(false);

        // Check MOSI -> rx_data path (SPI module received correctly)
        if (dut->rx_valid && dut->rx_data == mosi_byte) {
            std::cout << "  MOSI -> rx_data: PASS - SPI module received 0x"
                      << std::hex << (int)mosi_byte << std::dec << std::endl;
        } else {
            std::cout << "  MOSI -> rx_data: FAIL - Expected 0x"
                      << std::hex << (int)mosi_byte << " Got: 0x" << (int)dut->rx_data
                      << " Valid: " << dut->rx_valid << std::dec << std::endl;
        }

        // Check tx_data -> MISO path (SPI module transmitted correctly)
        if (received_miso == expected_miso_byte) {
            std::cout << "  tx_data -> MISO: PASS - Received 0x"
                      << std::hex << (int)received_miso << std::dec << std::endl;
        } else {
            std::cout << "  tx_data -> MISO: FAIL - Expected 0x"
                      << std::hex << (int)expected_miso_byte << " Got: 0x"
                      << (int)received_miso << std::dec << std::endl;
        }

        // Space after test for clean output
        delay(10);
    }
};

int main(int argc, char** argv) {
    // Initialize Verilator
    Verilated::commandArgs(argc, argv);

    // Create an instance of our module under test
    Vspi_interface *dut = new Vspi_interface;

    // Trace initialization
    Verilated::traceEverOn(true);
    VerilatedVcdC* trace = new VerilatedVcdC;
    dut->trace(trace, 99);
    trace->open("spi_interface_test.vcd");

    // Create our SPI tester helper
    SPITester tester(dut, trace);

    // Initialize signals
    dut->rst_n = 0;      // Start in reset
    dut->spi_sck = 0;    // Start with clock low
    dut->spi_cs_n = 1;   // Start with CS inactive (high)
    dut->spi_mosi = 0;   // Default MOSI value
    dut->tx_data = 0;
    dut->tx_ready = 0;

    // Reset sequence
    trace->dump(sim_time++);
    dut->eval();

    // Release reset
    dut->rst_n = 1;
    dut->eval();
    trace->dump(sim_time++);

    std::cout << "=== SPI Interface Bidirectional Tests ===" << std::endl;

    // Test 1: Simple byte transfer (pattern 0xA5)
    tester.run_transaction(0xA5, 0x5A, "Simple byte transfer (0xA5/0x5A)");

    // Test 2: Transfer with all zeros
    tester.run_transaction(0x00, 0xFF, "All zeros/ones transfer");

    // Test 3: Transfer with all ones
    tester.run_transaction(0xFF, 0x00, "All ones/zeros transfer");

    // Test 4: Alternating bit pattern
    tester.run_transaction(0x55, 0xAA, "Alternating bits (0x55/0xAA)");

    // Clean up
    trace->close();
    delete trace;
    delete dut;

    std::cout << "\nSimulation completed after " << sim_time << " ticks" << std::endl;

    return 0;
}
