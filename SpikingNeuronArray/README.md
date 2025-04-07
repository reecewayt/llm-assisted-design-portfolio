# Spiking Neuron Simulation with Verilator

This document provides instructions for setting up and running the simulation of the leaky integrate-and-fire neuron model using Verilator.

## Prerequisites

Before starting, make sure you have the following installed:

- Verilator (version 4.0 or higher)
- A C++ compiler (g++ or clang)
- GTKWave (for viewing waveforms)
- Make

On Debian/Ubuntu-based systems, you can install these dependencies with:

```bash
sudo apt update
sudo apt install verilator g++ make gtkwave
```

## Project Structure

```
.
├── Makefile                  # Build system configuration
├── src
│   └── neuron.sv            # SystemVerilog implementation of the neuron
└── tb
    └── neuron_tb.cpp        # C++ testbench for Verilator
```

## Neuron Implementation

The `neuron.sv` file contains a SystemVerilog implementation of a leaky integrate-and-fire neuron with:
- Configurable bit widths for various parameters
- Leak mechanism
- Threshold-based firing
- Refractory period

## Running the Simulation

1. **Build the simulation**

   ```bash
   make
   ```

   This will compile the SystemVerilog code using Verilator and build the C++ testbench.

2. **Run the simulation**

   ```bash
   make run
   ```

   This will execute the simulation, which will:
   - Apply test stimuli to the neuron
   - Generate a VCD waveform file
   - Print simulation results to the console

3. **View waveforms**

   ```bash
   make wave
   ```

   This will run the simulation and open GTKWave to visualize the resulting waveforms.

4. **Clean up generated files**

   ```bash
   make clean
   ```

## Understanding the Testbench

The C++ testbench (`neuron_tb.cpp`) applies a sequence of current stimuli to the neuron:

1. Reset phase (current = 0)
2. Medium current (20 units) - Should slowly accumulate
3. High current (30 units) - Should trigger spikes more frequently
4. Low current (5 units) - May not trigger spikes or trigger them rarely

The testbench outputs CSV-formatted data showing the time, input current, and spike output for later analysis.

## Modifying the Simulation

- To change the stimulus pattern, modify the `stimulus` vector in `neuron_tb.cpp`
- To adjust neuron parameters (threshold, leak value), modify the corresponding values in `neuron.sv`
- To run a longer simulation, increase the `MAX_SIM_TIME` constant in `neuron_tb.cpp`

## Analyzing Results

After running the simulation, you can view the waveforms with GTKWave to see the detailed behavior


### SPI Interface Description

# Neuromorphic SoC with SPI Interface

This document describes the SPI interface design for the spiking neural network ASIC. The design includes a standard SPI slave interface that allows external devices to configure, control, and read from the neural network.

## System Architecture

The system consists of three main components:

1. **SPI Interface** - Provides communication with external devices
2. **Neural Network** - The 3x3 spiking neural network
3. **Top Module** - Integrates the SPI interface with the neural network

## SPI Interface Protocol

The SPI interface implements a standard 4-wire SPI slave:

- **SCLK** - SPI clock (input)
- **MOSI** - Master Out Slave In (input)
- **MISO** - Master In Slave Out (output)
- **CS_N** - Chip Select, active low (input)

### Command Format

Each SPI transaction consists of:

1. **Command/Address Byte**:
   - Bit 7: Command type (0 = Read, 1 = Write)
   - Bits 6-0: Register address

2. **Data Byte**:
   - For write operations: data sent from master to slave
   - For read operations: data sent from slave to master

### Memory Map

| Address Range | Description                   | Access |
|---------------|-------------------------------|--------|
| 0x00          | Control Register              | R/W    |
| 0x01-0x1F     | Reserved                      | -      |
| 0x20-0x22     | Input Currents (Layer 1)      | R/W    |
| 0x30-0x38     | Weights (3x3 array)           | R/W    |
| 0xF0          | Output Spikes (Layer 2)       | R      |

### Control Register (0x00)

| Bit | Function         | Description                                    |
|-----|------------------|------------------------------------------------|
| 0   | Network Enable   | 0 = Disabled (clock gated), 1 = Enabled        |
| 1   | Network Reset    | 0 = In reset, 1 = Normal operation             |
| 2-7 | Reserved         | Set to 0                                       |

### Weight Matrix Addressing (0x30-0x38)

The weights are stored in row-major order:

| Address | Weight                       |
|---------|------------------------------|
| 0x30    | weights[0][0] (Row 0, Col 0) |
| 0x31    | weights[0][1] (Row 0, Col 1) |
| 0x32    | weights[0][2] (Row 0, Col 2) |
| 0x33    | weights[1][0] (Row 1, Col 0) |
| ...     | ...                          |
| 0x38    | weights[2][2] (Row 2, Col 2) |

### Output Spikes (0xF0)

The output spikes are packed into a single byte:

| Bit | Function      |
|-----|---------------|
| 0   | Layer 2 Neuron 0 Spike |
| 1   | Layer 2 Neuron 1 Spike |
