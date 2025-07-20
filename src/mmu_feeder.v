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

<<<<<<< HEAD
    // Done signal for output phase
=======
    wire [7:0] weights [0:3];
    wire [7:0] inputs [0:3];
    wire [7:0] c_out [0:3];

    assign weights[0] = weight0;
    assign weights[1] = weight1;
    assign weights[2] = weight2;
    assign weights[3] = weight3;

    assign inputs[0] = input0;
    assign inputs[1] = input1;
    assign inputs[2] = input2;
    assign inputs[3] = input3;

    assign c_out[0] = c00;
    assign c_out[1] = c01;
    assign c_out[2] = c10;
    assign c_out[3] = c11;

>>>>>>> 29f8ba429b77d603255adec93a3e590b7cfed28a
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
                2'b00: host_outdata = (c00[15] && c00[15:8] != 8'hFF) ? -8'sd128 :
                                      (!c00[15] && c00[15:8] != 8'h00) ? 8'sd127 :
                                      c00[7:0];
                2'b01: host_outdata = (c01[15] && c01[15:8] != 8'hFF) ? -8'sd128 :
                                      (!c01[15] && c01[15:8] != 8'h00) ? 8'sd127 :
                                      c01[7:0];
                2'b10: host_outdata = (c10[15] && c10[15:8] != 8'hFF) ? -8'sd128 :
                                      (!c10[15] && c10[15:8] != 8'h00) ? 8'sd127 :
                                      c10[7:0];
                2'b11: host_outdata = (c11[15] && c11[15:8] != 8'hFF) ? -8'sd128 :
                                      (!c11[15] && c11[15:8] != 8'h00) ? 8'sd127 :
                                      c11[7:0];
                default: host_outdata = 8'b0;
            endcase
        end
    end

endmodule