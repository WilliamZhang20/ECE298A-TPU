`default_nettype none

module mmu_feeder (
    input wire clk,
    input wire rst,
    input wire en,
    input wire [2:0] mmu_cycle,

    /* Memory module interface */
    input wire [7:0] weight0, weight1, weight2, weight3,
    input wire [7:0] input0, input1, input2, input3,

    /* systolic array -> feeder */
    input wire [7:0] c00, c01, c10, c11,

    /* feeder -> mmu */
    output reg clear,
    output reg [7:0] a_data0,
    output reg [7:0] a_data1,
    output reg [7:0] b_data0,
    output reg [7:0] b_data1,

    /* feeder -> rpi */
    output wire done,
    output reg [7:0] host_outdata
);

    // Done signal for output phase
    assign done = en && (mmu_cycle >= 3'b010) && (mmu_cycle <= 3'b101);

    // Output counter for selecting c_out
    reg [1:0] output_count;

    // Sequential logic for control and data outputs
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            clear <= 1;
            a_data0 <= 0;
            a_data1 <= 0;
            b_data0 <= 0;
            b_data1 <= 0;
            output_count <= 0;
        end else begin
            a_data0 <= 0;
            a_data1 <= 0;
            b_data0 <= 0;
            b_data1 <= 0;
            output_count <= 0;
            if (en) begin
                clear <= 0;

                // Update output_count during output phase
                if (mmu_cycle >= 3) begin
                    output_count <= output_count + 1;
                end else begin
                    output_count <= 0;
                end

                // Input assignments based on mmu_cycle
                case (mmu_cycle)
                    3'b000: begin
                        a_data0 <= weight0;
                        b_data0 <= input0;
                    end
                    3'b001: begin
                        a_data0 <= weight1;
                        a_data1 <= weight2;
                        b_data0 <= input2;
                        b_data1 <= input1;
                    end
                    3'b010: begin
                        a_data1 <= weight3;
                        b_data1 <= input3;
                    end
                    // Other cycles (3'b011 to 3'b101) keep defaults (0)
                    default: begin end
                endcase
            end else begin
                clear <= 1;
            end
        end
    end

    // Combinational logic for host_outdata with corrected saturation
    always @(*) begin
        host_outdata = 8'b0; // Default to avoid latch
        if (en) begin
            case (output_count)
                2'b00: host_outdata = c00;
                2'b01: host_outdata = c01;
                2'b10: host_outdata = c10;
                2'b11: host_outdata = c11;
                default: host_outdata = 8'b0;
            endcase
        end
    end

endmodule