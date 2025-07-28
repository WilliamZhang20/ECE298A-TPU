`default_nettype none
`timescale 1ns / 1ps

module systolic_array_tb;

    initial begin
`ifdef VCD_PATH
        $dumpfile(`VCD_PATH);
`else
		$dumpfile("systolic_array_tb.vcd");
`endif
		$dumpvars(0, systolic_array_tb);
        #1;
	end

  // Inputs
  reg clk;
  reg rst;
  reg clear;
  reg activation;
  reg signed [7:0] a_data0;
  reg signed [7:0] a_data1;
  reg signed [7:0] b_data0;
  reg signed [7:0] b_data1;

  // Outputs
  wire signed [15:0] c00;
  wire signed [15:0] c01;
  wire signed [15:0] c10;
  wire signed [15:0] c11;

  // Instantiate the systolic array module
  systolic_array_2x2 #(.WIDTH(8)) dut (
      .clk(clk),
      .rst(rst),
      .clear(clear),
      .activation(activation),
      .a_data0(a_data0),
      .a_data1(a_data1),
      .b_data0(b_data0),
      .b_data1(b_data1),
      .c00(c00),
      .c01(c01),
      .c10(c10),
      .c11(c11)
  );
  
endmodule
