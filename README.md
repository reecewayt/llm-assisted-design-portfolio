# LLM-Assisted Hardware Design Portfolio

A collection of AI/ML hardware design projects demonstrating how modern LLM tools can accelerate the development of specialized computing architectures. This portfolio focuses on CNN acceleration through parallel processing arrays and explores various aspects of hardware design for machine learning workloads.

> **Note**: This portfolio represents an experiment in LLM-assisted hardware design. The methodologies and results documented here contribute to understanding how AI tools can enhance traditional hardware design workflows while maintaining engineering rigor and verification standards.

> **Disclaimer**: This project extensively leverages LLM assistance for code generation, design exploration, and documentation as part of a methodology study in AI-assisted hardware design.

## 🎯 Project Overview

This portfolio documents my journey through ECE 510: Hardware for AI/ML at Portland State University (Spring 2025), exploring the intersection of hardware design and artificial intelligence. The core focus is developing specialized hardware accelerators that outperform software implementations for CNN workloads.

### Core Design: Parallel Matrix Processing Array

My design process began with benchmarking of CNN workloads using AlexNet to identify performance bottlenecks. As expected, matrix multiplication operations emerged as the primary computational constraint, leading to the development of a **broadcast-based parallel processing architecture**.

**Key Innovation**: Rather than implementing complex systolic arrays, I developed a simpler broadcast-based architecture where input vectors are distributed simultaneously to all processing elements (PEs). Each PE independently computes one element of the result matrix using dedicated MAC units, prioritizing simplicity and predictable timing.

### 🏆 Key Achievements

- ✅ **Two Complete ASIC Implementations**: 8-bit E4M3 floating-point (2×2) and 8-bit integer (3×3) matrix processors
- ✅ **Full ASIC Flow**: RTL to GDSII using OpenLane 2 for both designs  
- ✅ **Comprehensive Verification**: Custom testing framework with assertion-based verification and waveform analysis
- ⭐ **Performance Validation**: Achieved 3.2x speedup over optimized software baselines
- 🧠 **LLM-Assisted Methodology**: Demonstrated effective use of AI tools throughout the entire design process

## 📁 Project Structure

### Setup and Environment

Before exploring individual projects, set up the Python environment:

```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:reecewayt/llm-assisted-design-portfolio.git
cd llm-assisted-design-portfolio

# Set up Python environment and install dependencies
source ./setup.sh

# To update packages later
./setup.sh --update
```

The setup script creates a Python virtual environment and installs all necessary packages for CNN benchmarking and analysis tools.

### Project Components

*Detailed setup and usage instructions for each component can be found in their respective subdirectory READMEs.*

#### 🧮 **[CNN Benchmarks](./cnn_benchmarks/)**
**Purpose**: Systematic performance analysis of CNN architectures to identify hardware acceleration opportunities.

- **What it contains**: AlexNet benchmarking suite with profiling tools and arithmetic intensity analysis
- **Key outputs**: Performance metrics, computational bottleneck identification, and hardware design insights

#### 🧠 **[Spiking Neuron Array](./spiking_neuron_array/)**
**Purpose**: SystemVerilog implementation of neuromorphic computing elements with simulation framework using `Verilator`.

- **What it contains**: Leaky integrate-and-fire neuron models, multi-layer networks, and SPI interface
- **Key features**: Configurable neuron parameters, simple network topology, and SPI interface

#### ⚡ **[CNN Accelerator](./cnn-accelerator/)** *(Git Submodule)*
**Purpose**: Main project - complete ASIC implementation of parallel matrix processing arrays.

- **What it contains**: RTL implementations, verification frameworks (i.e. Unit testing), synthesis scripts, and performance analysis
- **Key deliverables**: Two complete ASIC designs with full GDSII layouts and performance characterization

#### 🔧 **[Scripts](./scripts/)**
**Purpose**: Analysis and benchmarking utilities supporting the main design project.

- **What it contains**: CPU matrix multiplication benchmarks for baseline performance comparison

## 🎓 Academic Context

**Course**: ECE 510: Hardware for AI/ML  
**Institution**: Portland State University  
**Term**: Spring 2025  

**Course Objective**: Learn to design, simulate, optimize, and evaluate specialized hardware (GPUs, TPUs, FPGAs, neuromorphic chips) for modern AI/ML algorithms using hardware/software co-design methodologies.

## 🔬 Design Methodology

### LLM-Assisted Design Process

This portfolio demonstrates a new approach to hardware design that leverages Large Language Models throughout the development cycle:

1. **Architecture Exploration**: Using LLMs to explore design alternatives and trade-offs
2. **Code Generation**: AI-assisted RTL development with human oversight and verification
3. **Verification Planning**: LLM-generated test scenarios and edge case identification
4. **Documentation**: Documentation generation and technical writing assistance
5. **Performance Analysis**: AI-assisted interpretation of benchmark results and optimization strategies


## 📊 Results Summary

| Implementation | Matrix Size | Data Type | Speedup vs Software | ASIC Flow Status |
|----------------|-------------|-----------|-------------------|------------------|
| Integer Array  | 3×3         | 8-bit int | 3.2x             | ✅ Complete      |
| Float Array    | 2×2         | E4M3      | 2.8x             | ✅ Complete      |

## 🚀 Getting Started

1. **Environment Setup**: Run `./setup.sh` to configure the Python environment
2. **Wiki Documentation**: Visit the [project wiki](https://github.com/reecewayt/llm-assisted-design-portfolio/wiki) for additional details
3. **Explore Benchmarks**: Start with CNN benchmarking to understand computational bottlenecks
4. **Main Project**: Dive into the CNN accelerator submodule for complete ASIC design flow

## 📚 Additional Resources

- **[Project Wiki](https://github.com/reecewayt/llm-assisted-design-portfolio/wiki)**: Comprehensive technical documentation and design methodology
- **[CNN Accelerator Repository](https://github.com/reecewayt/cnn-accelerator)**: Complete ASIC design implementation
- **Performance Analysis**: Detailed benchmark results and hardware characterization data

## 🔄 Submodule Management

This repository uses Git submodules for the main CNN accelerator project:

```bash
# Update all submodules to latest commits
git submodule update --remote

# Commit submodule updates
git add .
git commit -m "Update submodules"
```

## 📄 License

MIT License - see [LICENSE](./LICENSE) for details.

---

