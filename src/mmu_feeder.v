`default_nettype none

module mmu_feeder (
    input wire clk,
    input wire rst,
    input wire en,
    input wire [2:0] mmu_cycle,

    input wire transpose,

    /* Memory module interface */
    input wire [7:0] weight0, weight1, weight2, weight3,
    input wire [7:0] input0, input1, input2, input3,

    /* systolic array -> feeder */
    input wire signed [11:0] c00, c01, c10, c11,

    /* feeder -> mmu */
    output wire clear,
    output wire [7:0] a_data0,
    output wire [7:0] a_data1,
    output wire [7:0] b_data0,
    output wire [7:0] b_data1,

    /* feeder -> rpi */
    output wire done,
    output reg [7:0] host_outdata
);

    // Done signal for output phase
    assign done = en && (mmu_cycle >= 3'b010) && (mmu_cycle <= 3'b101);
    assign clear = (mmu_cycle == 3'b110);

    // Output counter for selecting c_out
    reg [1:0] output_count;

    function [7:0] saturate_to_s8;
        input signed [11:0] val;
        begin
            if (val > 127)
                saturate_to_s8 = 8'sd127;
            else if (val < -128)
                saturate_to_s8 = -8'sd128;
            else
                saturate_to_s8 = val[7:0];
        end
    endfunction

    assign a_data0 = en ?
                     (mmu_cycle == 3'd0) ? weight0 : 
                     (mmu_cycle == 3'd1) ? weight1 : 0 : 0;

    assign a_data1 = en ?
                     (mmu_cycle == 3'd1) ? weight2 : 
                     (mmu_cycle == 3'd2) ? weight3 : 0 : 0;

    assign b_data0 = en ? 
                     (mmu_cycle == 3'd0) ? input0 : 
                     (mmu_cycle == 3'd1) ? 
                     transpose ? input1 : input2 : 0 : 0;

    assign b_data1 = en ?
                     (mmu_cycle == 3'd1) ? 
                     transpose ? input2 : input1 :
                     (mmu_cycle == 3'd2) ? input3 : 0 : 0;


    // Sequential logic for control and data outputs
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            output_count <= 0;
        end else begin
            output_count <= 0;
            if (en) begin
                // Update output_count during output phase
                if (mmu_cycle >= 2) begin
                    output_count <= output_count + 1;
                end else begin
                    output_count <= 0;
                end
            end
        end
    end

    // Combinational logic for host_outdata with corrected saturation
    always @(*) begin
        host_outdata = 8'b0; // Default to avoid latch
        if (en) begin
            case (output_count)
                2'b00: host_outdata = saturate_to_s8(c00);
                2'b01: host_outdata = saturate_to_s8(c01);
                2'b10: host_outdata = saturate_to_s8(c10);
                2'b11: host_outdata = saturate_to_s8(c11);
                default: host_outdata = 8'b0;
            endcase
        end
    end

endmodule