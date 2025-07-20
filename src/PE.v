module PE #(
    parameter WIDTH = 8
)(
    input wire clk,
    input wire rst,
    input wire clear,
    input wire relu_en,
    input wire signed [WIDTH-1:0] a_in,
    input wire signed [WIDTH-1:0] b_in,

    output reg signed [WIDTH-1:0] a_out,
    output reg signed [WIDTH-1:0] b_out,

    output reg signed [WIDTH-1:0] c_out
);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            a_out <= 0;
            b_out <= 0;
            c_out <= 0;
        end else if (clear) begin
            a_out <= 0;
            b_out <= 0;
            c_out <= 0;
        end else begin
            a_out <= a_in;
            b_out <= b_in;

            c_out <= c_out + (a_in * b_in);
        end
    end

endmodule
