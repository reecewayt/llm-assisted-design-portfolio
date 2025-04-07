module spi_interface #(
    parameter int DATA_WIDTH = 8        // Width of data bits
) (
    // Reset
    input logic rst_n,                  // Active low reset
    // SPI interface
    input logic spi_sck,                // SPI clock
    input logic spi_cs_n,               // SPI chip select (active low)
    input logic spi_mosi,               // SPI main out, subnode in
    output logic spi_miso,              // SPI main in, subnode out
    // Data interface to ASIC logic
    output logic [DATA_WIDTH-1:0] rx_data,   // Received data byte
    output logic rx_valid,                    // Received data valid strobe
    input logic [DATA_WIDTH-1:0] tx_data,    // Data to transmit
    input logic tx_ready                      // Data ready to transmit
);
    // Internal registers
    logic [DATA_WIDTH-1:0] rx_shift_reg;    // Receive shift register
    logic [DATA_WIDTH-1:0] tx_shift_reg;    // Transmit shift register
    logic [2:0] rx_bit_counter;             // Counts received bits (0-7)
    logic [2:0] tx_bit_counter;             // Counts transmitted bits (0-7)
    logic tx_data_latched;                  // Flag to indicate tx data is latched

    // SPI receiver - sample MOSI on rising edge of SCK
    always_ff @(posedge spi_sck or negedge rst_n) begin
        if (!rst_n) begin
            rx_shift_reg <= '0;
            rx_bit_counter <= '0;
            rx_valid <= 1'b0;
            rx_data <= '0;
        end else if (spi_cs_n) begin
            // CS inactive - reset state
            rx_bit_counter <= '0;
            rx_valid <= 1'b0;
        end else begin
            // CS active - shift in data
            rx_valid <= 1'b0;  // Default state

            // Shift in MOSI bit (MSB first)
            rx_shift_reg <= {rx_shift_reg[DATA_WIDTH-2:0], spi_mosi};

            // Increment bit counter
            rx_bit_counter <= rx_bit_counter + 1'b1;

            // When we receive a full byte, set valid and update rx_data
            if (rx_bit_counter == 3'b111) begin
                // Final bit is already shifted in above, so include it in rx_data
                rx_data <= {rx_shift_reg[DATA_WIDTH-2:0], spi_mosi};
                rx_bit_counter <= '0;  // Reset counter
                rx_valid <= 1'b1;
            end
        end
    end

    // TX data latch - capture new data when tx_ready is high
    always_ff @(posedge tx_ready or negedge rst_n) begin
        if (!rst_n) begin
            tx_data_latched <= 1'b0;
        end else begin
            // Latch new data into shift register
            tx_shift_reg <= tx_data;
            tx_data_latched <= 1'b1;
        end
    end

    // MISO control - update on falling edge of SCK
    always_ff @(negedge spi_sck or negedge rst_n) begin
        if (!rst_n) begin
            spi_miso <= 1'b0;
            tx_bit_counter <= '0;
        end else if (spi_cs_n) begin
            // CS inactive - reset state
            spi_miso <= 1'b0;
            tx_bit_counter <= '0;
        end else if (tx_data_latched) begin
            // Output MSB first
            spi_miso <= tx_shift_reg[DATA_WIDTH-1];

            // Shift register for next bit
            tx_shift_reg <= {tx_shift_reg[DATA_WIDTH-2:0], 1'b0};

            // Increment counter
            tx_bit_counter <= tx_bit_counter + 1'b1;

            // Reset latch flag after full byte
            if (tx_bit_counter == 3'b111) begin
                tx_data_latched <= 1'b0;
                tx_bit_counter <= '0;
            end
        end
    end
endmodule
