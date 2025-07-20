module PE #(
    parameter WIDTH = 8
)(
    input wire clk,
    input wire rst,
    input wire clear,
    input wire transpose_en,  // Enables fused transpose
    input wire relu_en,       // Enables fused ReLU

    input wire signed [WIDTH-1:0] a_in,
    input wire signed [WIDTH-1:0] b_in,

    output reg signed [WIDTH-1:0] a_out,
    output reg signed [WIDTH-1:0] b_out,

    output reg signed [WIDTH-1:0] c_out
);

    reg signed [WIDTH-1:0] product;

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
            // Fused transpose
            a_out <= transpose_en ? b_in : a_in;
            b_out <= transpose_en ? a_in : b_in;

            // Compute product
            product = a_in * b_in;

            // Apply ReLU if enabled
            if (relu_en && product < 0)
                c_out <= 0;
            else
                c_out <= c_out + product;
        end
    end

endmodule
