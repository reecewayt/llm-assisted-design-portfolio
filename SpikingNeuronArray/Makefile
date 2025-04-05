######################################################################
# Makefile for Spiking Neuron Verilator Simulation
######################################################################

# Verilator flags
VERILATOR = verilator
VERILATOR_FLAGS = --trace --trace-params --trace-structs --trace-underscore \
                  -Wall -Wno-fatal \
                  --cc --exe --build

# Directory structure
SRC_DIR = src
TB_DIR = tb
OBJ_DIR = obj_dir

# Module names (matching filenames without extensions)
NEURON_MODULE = neuron
NETWORK_MODULE = network
SYNAPSE_MODULE = multi_input_synapse
SPI_MODULE = spi_interface
ASIC_MODULE = neuromorphic_asic

# Source files
NEURON_SV = $(SRC_DIR)/$(NEURON_MODULE).sv
SYNAPSE_SV = $(SRC_DIR)/$(SYNAPSE_MODULE).sv
NETWORK_SV = $(SRC_DIR)/$(NETWORK_MODULE).sv
SPI_SV = $(SRC_DIR)/$(SPI_MODULE).sv
ASIC_SV = $(SRC_DIR)/$(ASIC_MODULE).sv

NEURON_TB = $(TB_DIR)/$(NEURON_MODULE)_tb.cpp
NETWORK_TB = $(TB_DIR)/$(NETWORK_MODULE)_tb.cpp
ASIC_TB = $(TB_DIR)/spi_tb.cpp
SPI_STANDALONE_TB = $(TB_DIR)/standalone_spi_test.cpp

# All SystemVerilog files for different configurations
NETWORK_SV_FILES = $(NEURON_SV) $(SYNAPSE_SV) $(NETWORK_SV)
ASIC_SV_FILES = $(NEURON_SV) $(SYNAPSE_SV) $(NETWORK_SV) $(SPI_SV) $(ASIC_SV)

# Target executables
NEURON_TARGET = $(OBJ_DIR)/V$(NEURON_MODULE)
NETWORK_TARGET = $(OBJ_DIR)/V$(NETWORK_MODULE)
ASIC_TARGET = $(OBJ_DIR)/V$(ASIC_MODULE)
SPI_STANDALONE_TARGET = $(OBJ_DIR)/V$(SPI_MODULE)

# Default target (build the neuron simulation)
all: neuron

# Neuron target
neuron: $(NEURON_TARGET)

# Network target
network: $(NETWORK_TARGET)

# ASIC target
asic: $(ASIC_TARGET)

# Standalone SPI interface test target
spi-standalone: $(SPI_STANDALONE_TARGET)

# Build the neuron simulation
$(NEURON_TARGET): $(NEURON_SV) $(NEURON_TB)
	$(VERILATOR) $(VERILATOR_FLAGS) $(NEURON_SV) $(NEURON_TB)

# Build the network simulation
$(NETWORK_TARGET): $(NETWORK_SV_FILES) $(NETWORK_TB)
	$(VERILATOR) $(VERILATOR_FLAGS) --top-module $(NETWORK_MODULE) \
		$(NETWORK_SV_FILES) $(NETWORK_TB)

# Build the ASIC simulation
$(ASIC_TARGET): $(ASIC_SV_FILES) $(ASIC_TB)
	$(VERILATOR) $(VERILATOR_FLAGS) --top-module $(ASIC_MODULE) \
		$(ASIC_SV_FILES) $(ASIC_TB)

# Build the standalone SPI interface test
$(SPI_STANDALONE_TARGET): $(SPI_SV) $(SPI_STANDALONE_TB)
	$(VERILATOR) $(VERILATOR_FLAGS) --top-module $(SPI_MODULE) \
		$(SPI_SV) $(SPI_STANDALONE_TB)

# Run the neuron simulation
run-neuron: $(NEURON_TARGET)
	@echo "Running neuron simulation..."
	@./$(NEURON_TARGET)

# Run the network simulation
run-network: $(NETWORK_TARGET)
	@echo "Running network simulation..."
	@./$(NETWORK_TARGET)

# Run the ASIC simulation
run-asic: $(ASIC_TARGET)
	@echo "Running ASIC simulation..."
	@./$(ASIC_TARGET)

# Run the standalone SPI interface test
run-spi-standalone: $(SPI_STANDALONE_TARGET)
	@echo "Running standalone SPI interface test..."
	@./$(SPI_STANDALONE_TARGET)

# Generate VCD output and run GTKWave for neuron simulation
wave-neuron: run-neuron
	@echo "Opening neuron waveform viewer..."
	@gtkwave neuron.vcd &

# Generate VCD output and run GTKWave for network simulation
wave-network: run-network
	@echo "Opening network waveform viewer..."
	@gtkwave network.vcd &

# Generate VCD output and run GTKWave for ASIC simulation
wave-asic: run-asic
	@echo "Opening ASIC waveform viewer..."
	@gtkwave neuromorphic_asic.vcd &

# Generate VCD output and run GTKWave for standalone SPI interface test
wave-spi-standalone: run-spi-standalone
	@echo "Opening SPI interface waveform viewer..."
	@gtkwave spi_interface_test.vcd &

# Clean up generated files
clean:
	rm -rf $(OBJ_DIR)
	rm -f *.vcd
	rm -f *.csv

# Help target
help:
	@echo "Spiking Neural Network Verilator Simulation"
	@echo "----------------------------------------"
	@echo "make               - Build the neuron simulation (default)"
	@echo "make neuron        - Build the neuron simulation"
	@echo "make network       - Build the neural network simulation"
	@echo "make asic          - Build the complete ASIC with SPI interface"
	@echo "make spi-standalone - Build the standalone SPI interface test"
	@echo "make run-neuron    - Run the neuron simulation"
	@echo "make run-network   - Run the network simulation"
	@echo "make run-asic      - Run the ASIC simulation with SPI interface"
	@echo "make run-spi-standalone - Run the standalone SPI interface test"
	@echo "make wave-neuron   - Run the neuron simulation and open waveform viewer"
	@echo "make wave-network  - Run the network simulation and open waveform viewer"
	@echo "make wave-asic     - Run the ASIC simulation and open waveform viewer"
	@echo "make wave-spi-standalone - Run the SPI interface test and open waveform viewer"
	@echo "make clean         - Remove generated files"
	@echo "make help          - Display this help message"

.PHONY: all neuron network asic spi-standalone run-neuron run-network run-asic run-spi-standalone \
        wave-neuron wave-network wave-asic wave-spi-standalone clean help
