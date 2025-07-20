module systolic_array_2x2 #(
    parameter WIDTH = 8
)(
    input wire clk,
    input wire rst,
    input wire clear,

    input wire [WIDTH-1:0] a_data0,
    input wire [WIDTH-1:0] a_data1,
    input wire [WIDTH-1:0] b_data0,
    input wire [WIDTH-1:0] b_data1,

    output wire signed [WIDTH-1:0] c00,
    output wire signed [WIDTH-1:0] c01,
    output wire signed [WIDTH-1:0] c10,
    output wire signed [WIDTH-1:0] c11
);

    // Internal signals between PEs
    wire [WIDTH-1:0] a_wire [0:1][0:2];
    wire [WIDTH-1:0] b_wire [0:2][0:1];
    wire signed [2*WIDTH-1:0] c_array [0:1][0:1];

    // Clamped/truncated output values
    wire signed [WIDTH-1:0] c_clamped [0:1][0:1];

    // Input loading at top-left
    assign a_wire[0][0] = a_data0;
    assign a_wire[1][0] = a_data1;
    assign b_wire[0][0] = b_data0;
    assign b_wire[0][1] = b_data1;

    genvar i, j;
    generate
        for (i = 0; i < 2; i = i + 1) begin : row
            for (j = 0; j < 2; j = j + 1) begin : col
                PE #(.WIDTH(8)) pe_inst (
                    .clk(clk),
                    .rst(rst),
                    .clear(clear),
                    .a_in(a_wire[i][j]),
                    .b_in(b_wire[i][j]),
                    .a_out(a_wire[i][j+1]),
                    .b_out(b_wire[i+1][j]),
                    .c_out(c_array[i][j])
                );
            end
        end
    endgenerate

    localparam signed [2*WIDTH-1:0] MAX_VAL = (1 <<< (WIDTH-1)) - 1;
    localparam signed [2*WIDTH-1:0] MIN_VAL = -(1 <<< (WIDTH-1));

    generate
        for (i = 0; i < 2; i = i + 1) begin : clamp_row
            for (j = 0; j < 2; j = j + 1) begin : clamp_col
                assign c_clamped[i][j] = (c_array[i][j] > MAX_VAL) ? MAX_VAL[WIDTH-1:0] :
                                         (c_array[i][j] < MIN_VAL) ? MIN_VAL[WIDTH-1:0] :
                                         c_array[i][j][WIDTH-1:0];
            end
        end
    endgenerate

    // Assign clamped values to outputs
    assign c00 = c_clamped[0][0];
    assign c01 = c_clamped[0][1];
    assign c10 = c_clamped[1][0];
    assign c11 = c_clamped[1][1];
endmodule
