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

    assign done = en && (mmu_cycle >= 3'b010) && (mmu_cycle <= 3'b101);

    reg [1:0] output_count;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            clear <= 1;
            a_data0 <= 0;
            a_data1 <= 0;
            b_data0 <= 0;
            b_data1 <= 0;
            output_count <= 0;
        end else begin
            if (en) begin
                clear <= 0;

                if (mmu_cycle >= 3) begin
                    output_count <= output_count + 1;
                end else begin
                    output_count <= 0;
                end
            
                case (mmu_cycle)
                    3'b000: begin
                        a_data0  <= weights[0];   
                        a_data1  <= 8'b0;         
                        b_data0  <= inputs[0];   
                        b_data1  <= 8'b0;
                        output_count <= 0; 
                    end

                    3'b001: begin
                        a_data0  <= weights[1];   
                        a_data1  <= weights[2];   
                        b_data0  <= inputs[2];   
                        b_data1  <= inputs[1];
                    end

                    3'b010: begin
                        a_data0  <= 0;         
                        a_data1  <= weights[3];   
                        b_data0  <= 0;          
                        b_data1  <= inputs[3];
                    end

                    3'b011: begin
                        a_data0 <= 0;      
                        a_data1 <= 0;
                        b_data0 <= 0;       
                        b_data1 <= 0;  
                    end

                    3'b100: begin
                        a_data0 <= 0;      
                        a_data1 <= 0;
                        b_data0 <= 0;       
                        b_data1 <= 0;
                    end

                    3'b101: begin
                        a_data0 <= 0;      
                        a_data1 <= 0;
                        b_data0 <= 0;       
                        b_data1 <= 0;
                    end

                    default: begin
                        a_data0 <= 8'b0;
                        a_data1 <= 8'b0;
                        b_data0 <= 8'b0;
                        b_data1 <= 8'b0;
                    end
                    
                endcase
            end else begin
                clear <= 1;
                a_data0 <= 0;
                a_data1 <= 0;
                b_data0 <= 0;
                b_data1 <= 0;
                output_count <= 0;
            end
        end
    end

    always @(*) begin
        host_outdata = 0;
        if (en) begin
            host_outdata = c_out[output_count];
        end
    end

endmodule